#!/usr/bin/env python3
"""Polynomial exponent tuning for multivariate interpolants.

This script follows the exact settings from checker-multivariateSI-2D.ipynb:
- base_lr_v = 1e-3
- N_epoch = 250
- n_inner = 500
- batch_size = 1000
- base_variance = 3.0
- target_type = "rect"
- scheduler = CosineAnnealingWarmRestarts
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import interflow.fabrics_extra as fabrics_extra
import interflow.prior as prior
import interflow.stochastic_interpolant as stochastic_interpolant
from interflow.unet_velocity import make_lightweight_unet_velocity_2d


@dataclass
class TrainingConfig:
    exp_name: str
    num_epochs: int = 250
    batch_size: int = 1000
    n_inner: int = 500
    base_lr_v: float = 1e-3
    v_sq_samples: int = 50_000
    v_sq_time_points: int = 100
    v_sq_batch_size: int = 2_000
    plot_batch_size: int = 10_000
    ode_steps: int = 100
    seed: int = 0
    base_variance: float = 3.0
    target_type: str = "rect"
    results_root: Path = Path("results_exponent_tuning")


@dataclass
class MatrixConfig:
    matrix_type: str = "diagonal"
    coefficient_type: str = "polynomial"
    exponent_p: List[float] = None
    exponent_q: List[float] = None

    def __post_init__(self):
        if self.coefficient_type == "polynomial":
            if self.exponent_p is None:
                self.exponent_p = [1.0, 1.0]
            if self.exponent_q is None:
                self.exponent_q = [1.0, 1.0]


# --------------------------------------------------------------------------------------
# Target samplers
# --------------------------------------------------------------------------------------

def sample_rect(
    bs: int,
    device: torch.device,
    dtype: torch.dtype,
    w: float = 10.0,
    h: float = 0.1,
    layers_x: int = 2,
    layers_y: int = 4,
    scale: float = 1.0,
) -> torch.Tensor:
    """Sample from rectangular checkerboard pattern."""
    col = torch.randint(layers_x, (bs,), device=device)
    base_row = torch.randint(layers_y, (bs,), device=device)

    x_in = torch.rand(bs, device=device, dtype=dtype) * w
    x = (col.to(dtype) + x_in / w) * w

    y_in = torch.rand(bs, device=device, dtype=dtype) * h
    y = (base_row.to(dtype) * 2.0 * h) + ((col % 2).to(dtype) * h) + y_in

    x = x - (layers_x * w) / 2.0
    y = y - (layers_y * h)
    return torch.stack((x, y), dim=1) * scale


TARGET_BUILDERS = {
    "rect": sample_rect,
}


# --------------------------------------------------------------------------------------
# MixedInterpolant wrapper
# --------------------------------------------------------------------------------------

class MixedInterpolant(torch.nn.Module):
    """Wrapper for multivariate interpolant with API compatibility."""

    def __init__(self, interpolant_base, interpolant_aux, mode='base', mixed_steps=1000):
        super().__init__()
        self.interpolant_base = interpolant_base
        self.interpolant_aux = interpolant_aux
        self.mode = mode
        self.mixed_steps = mixed_steps
        self.current_step = 0

        # Check if base interpolant has a flow_model (for compatibility)
        self.has_flow_model = hasattr(interpolant_base, 'flow_model')

        # Create dummy flow_model if base doesn't have one (for multivariate interpolant)
        if not self.has_flow_model:
            self.flow_model = torch.nn.Module()
        else:
            self.flow_model = interpolant_base.flow_model

    def __getattr__(self, name):
        if name in ['interpolant_base','interpolant_aux', 'mode', 'mixed_steps', 'current_step', 'has_flow_model', 'flow_model']:
            return super().__getattr__(name)
        if name in ['calc_xt', 'dtIt']:
            return super().__getattr__(name)
        if self.mode == 'base':
            return getattr(self.interpolant_base, name)
        elif self.mode == 'aux':
            return getattr(self.interpolant_aux, name)
        else:
            return getattr(self.interpolant_base, name)

    def set_step(self, step):
        self.current_step = step

    def set_mode(self, mode):
        if mode not in ['base', 'aux', 'mixed']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'base', 'aux', or 'mixed'")
        self.mode = mode

    def calc_xt(self, t, x0, x1):
        if self.mode == 'base':
            return self.interpolant_base.calc_xt(t, x0, x1)
        elif self.mode == 'aux':
            return self.interpolant_aux.calc_xt(t, x0, x1)
        else:  # mixed mode
            xt_complex = self.interpolant_base.calc_xt(t, x0, x1)
            xt_simple = self.interpolant_aux.calc_xt(t, x0, x1)
            alpha = self.get_mixed_factor()
            return alpha * xt_complex + (1 - alpha) * xt_simple

    def dtIt(self, t, x0, x1):
        if self.mode == 'base':
            return self.interpolant_base.dtIt(t, x0, x1)
        elif self.mode == 'aux':
            return self.interpolant_aux.dtIt(t, x0, x1)
        else:  # mixed mode
            dtIt_complex = self.interpolant_base.dtIt(t, x0, x1)
            dtIt_simple = self.interpolant_aux.dtIt(t, x0, x1)
            alpha = self.get_mixed_factor()
            return alpha * dtIt_complex + (1 - alpha) * dtIt_simple


# --------------------------------------------------------------------------------------
# Metric utilities
# --------------------------------------------------------------------------------------

def total_grad_norm(params: Iterable[torch.nn.Parameter]) -> float:
    sq_norm = 0.0
    for p in params:
        if p.grad is not None:
            sq_norm += p.grad.detach().pow(2).sum().item()
    return math.sqrt(sq_norm) if sq_norm > 0 else 0.0


def _as_float_list(values):
    if values is None:
        return []
    if torch.is_tensor(values):
        return values.detach().cpu().flatten().tolist()
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    return [float(values)]


def _compute_kink_locations(exponent_p_list, exponent_q_list):
    """
    Compute critical time points where velocity field has kinks.

    For A(t) = (1-t^p): kink at t ≈ (p-2)/(p-1) when p > 2
    For B(t) = t^q:     kink at t ≈ (q-2)/(q-1) when q > 2

    Returns:
        List of (t_center, importance_weight) tuples
    """
    kink_locations = []

    # Analyze A(t) = (1-t^p) kinks
    max_p = max(exponent_p_list) if exponent_p_list else 1.0
    if max_p > 2.0:
        # Primary kink at t = (p-2)/(p-1)
        t_kink_A = (max_p - 2.0) / (max_p - 1.0)
        importance = max_p - 1.0  # Higher exponent = sharper kink
        kink_locations.append((t_kink_A, importance))
    elif max_p > 1.0:
        # Rapid change near t=1
        kink_locations.append((0.95, max_p - 1.0))

    # Analyze B(t) = t^q kinks
    max_q = max(exponent_q_list) if exponent_q_list else 1.0
    if max_q > 2.0:
        # Primary kink at t = (q-2)/(q-1)
        t_kink_B = (max_q - 2.0) / (max_q - 1.0)
        importance = max_q - 1.0
        # Only add if far from existing kinks
        if not kink_locations or abs(t_kink_B - kink_locations[0][0]) > 0.15:
            kink_locations.append((t_kink_B, importance))
    elif max_q > 1.0:
        # Rapid change near t=0
        kink_locations.append((0.05, max_q - 1.0))

    return kink_locations


def _infer_beta_params(matrix_cfg):
    """
    ENHANCED: Compute both alpha and beta parameters for better kink coverage.

    Returns (alpha, beta) where:
    - alpha: Used in Beta(α,1) to concentrate near t=1 (A(t) kinks)
    - beta:  Used in Beta(1,β) to concentrate near t=0 (B(t) kinks)
    """
    if not isinstance(matrix_cfg, dict):
        return 1.0, 1.0
    if matrix_cfg.get('coefficient_type', 'polynomial') != 'polynomial':
        return 1.0, 1.0

    exponents_p = _as_float_list(matrix_cfg.get('exponent_p'))
    exponents_q = _as_float_list(matrix_cfg.get('exponent_q'))

    if not exponents_p:
        exponents_p = [1.0]
    if not exponents_q:
        exponents_q = [1.0]

    # For Beta(α, 1): concentrates samples near t=1 (for A(t) kinks)
    alpha = float(max(max(exponents_p), 1.0))

    # For Beta(1, β): concentrates samples near t=0 (for B(t) kinks)
    beta = float(max(max(exponents_q), 1.0))

    return alpha, beta


def _sample_times_with_importance(
    bs: int,
    device: torch.device,
    dtype: torch.dtype,
    exponents_p: list,
    exponents_q: list,
    strategy: str = 'enhanced_beta_mixture',
    keepdim: bool = False
):
    """
    ENHANCED: Sample time points with better kink coverage.

    Strategy:
    - 'enhanced_beta_mixture': Mixture of Beta(α,1), Beta(1,β), and Uniform
      for comprehensive coverage of both A(t) and B(t) kinks
    """
    alpha, beta = _infer_beta_params({
        'coefficient_type': 'polynomial',
        'exponent_p': exponents_p,
        'exponent_q': exponents_q,
    })

    if strategy == 'enhanced_beta_mixture' and (alpha > 1.0 or beta > 1.0):
        # Three-component mixture for better kink coverage
        # Adjust component weights based on exponents
        if alpha > 2.0 and beta > 2.0:
            # Both have strong kinks
            component_weights = [0.35, 0.35, 0.30]
        elif alpha > 2.0:
            # Mainly A(t) kinks near t=1
            component_weights = [0.50, 0.20, 0.30]
        elif beta > 2.0:
            # Mainly B(t) kinks near t=0
            component_weights = [0.20, 0.50, 0.30]
        else:
            # Both low exponents
            component_weights = [0.30, 0.30, 0.40]

        # Sample component assignments
        component_choices = torch.multinomial(
            torch.tensor(component_weights, dtype=dtype, device=device),
            bs,
            replacement=True,
        )

        samples = torch.zeros(bs, dtype=dtype, device=device)

        # Component 0: Beta(α, 1) - concentrates near t=1
        mask0 = component_choices == 0
        n0 = mask0.sum().item()
        if n0 > 0:
            beta_dist0 = torch.distributions.Beta(
                torch.tensor(alpha, dtype=dtype, device=device),
                torch.tensor(1.0, dtype=dtype, device=device),
            )
            samples[mask0] = beta_dist0.sample((n0,)).to(device)

        # Component 1: Beta(1, β) - concentrates near t=0
        mask1 = component_choices == 1
        n1 = mask1.sum().item()
        if n1 > 0:
            beta_dist1 = torch.distributions.Beta(
                torch.tensor(1.0, dtype=dtype, device=device),
                torch.tensor(beta, dtype=dtype, device=device),
            )
            samples[mask1] = beta_dist1.sample((n1,)).to(device)

        # Component 2: Uniform
        mask2 = component_choices == 2
        n2 = mask2.sum().item()
        if n2 > 0:
            samples[mask2] = torch.rand(n2, dtype=dtype, device=device)

        # Compute importance weights
        eps = torch.finfo(dtype).eps
        clamped = samples.clamp(min=eps, max=1 - eps)

        beta_dist0 = torch.distributions.Beta(
            torch.tensor(alpha, dtype=dtype, device=device),
            torch.tensor(1.0, dtype=dtype, device=device),
        )
        beta_dist1 = torch.distributions.Beta(
            torch.tensor(1.0, dtype=dtype, device=device),
            torch.tensor(beta, dtype=dtype, device=device),
        )

        pdf = (
            component_weights[0] * torch.exp(beta_dist0.log_prob(clamped))
            + component_weights[1] * torch.exp(beta_dist1.log_prob(clamped))
            + component_weights[2] * 1.0
        )
        weights = (1.0 / (pdf + eps)).detach()
    else:
        # Fallback to uniform or simple Beta
        if alpha <= 1.0 + 1e-6:
            samples = torch.rand(bs, device=device, dtype=dtype)
            weights = torch.ones_like(samples)
        else:
            concentration1 = torch.tensor(alpha, device=device, dtype=dtype)
            concentration0 = torch.tensor(1.0, device=device, dtype=dtype)
            beta_dist = torch.distributions.Beta(concentration1, concentration0)
            samples = beta_dist.sample((bs,)).to(device=device, dtype=dtype)
            eps = torch.finfo(dtype).eps
            clamped = samples.clamp(min=eps, max=1 - eps)
            log_pdf = beta_dist.log_prob(clamped)
            weights = torch.exp(-log_pdf).detach()

    if keepdim:
        samples = samples.unsqueeze(-1)
        weights = weights.unsqueeze(-1)

    return samples, weights


def estimate_v_squared_norm_grid(
    v: torch.nn.Module,
    interpolant,
    base_sampler,
    target_sampler,
    n_samples: int,
    n_time_points: int,
    batch_size: int,
    device: torch.device,
) -> float:
    """Estimate E[|v|^2] over space and time."""
    v_mode = v.training
    v.eval()

    times = torch.linspace(0.0, 1.0, n_time_points, device=device)
    samples_per_time = max(1, n_samples // n_time_points)

    total_sq = 0.0
    total_count = 0

    with torch.no_grad():
        for t_scalar in times:
            remaining = samples_per_time
            while remaining > 0:
                current_bs = min(batch_size, remaining)
                x0s = base_sampler(current_bs)
                x1s = target_sampler(current_bs)

                # Create time tensor that broadcasts correctly
                t = torch.tensor([t_scalar.item()], device=device, dtype=x0s.dtype)
                xt = interpolant.calc_xt(t, x0s, x1s)
                if isinstance(xt, tuple):
                    xt = xt[0]
                # Pass scalar time to match network expectations
                vt = v(xt, t)
                sq = vt.pow(2).sum(dim=-1)
                total_sq += sq.sum().item()
                total_count += sq.numel()
                remaining -= current_bs

    if v_mode:
        v.train()

    return total_sq / max(total_count, 1)


def rollout_samples(
    v: torch.nn.Module,
    interpolant,
    base_sampler,
    n_step: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate samples using PFlow integrator."""
    v.eval()
    with torch.no_grad():
        base_batch = base_sampler(batch_size)
        integrator = stochastic_interpolant.PFlowIntegrator(
            b=v,
            method="dopri5",
            interpolant=interpolant,
            n_step=n_step,
        )
        traj, _ = integrator.rollout(base_batch)
        xf = traj[-1].detach().cpu()
    v.train()
    return xf


# --------------------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------------------

def save_epoch_plot(
    epoch: int,
    samples: torch.Tensor,
    metrics: List[Dict],
    out_path: Path,
    width: float = 10.0,
    height: float = 0.4,
) -> None:
    """Save epoch visualization with samples and metrics."""
    epochs = np.arange(1, epoch + 2)
    losses_v = [m["loss_v"] for m in metrics]
    v_sq = [m["v_sq_norm"] for m in metrics]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Samples histogram
    axes[0].hist2d(samples[:, 0], samples[:, 1], bins=100,
                   range=[[-width, width], [-height, height]])
    axes[0].set_title("Samples from PFlow", fontsize=14)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # Plot 2: Losses
    axes[1].plot(epochs, losses_v, label='v_loss', color='blue', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training Loss', fontsize=14)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Plot 3: v_squared_norm
    axes[2].plot(epochs, v_sq, color='green', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel(r'$\mathbb{E}[|v_t|^2]$')
    axes[2].set_title('Velocity Field Squared Norm', fontsize=14)
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(f"Epoch {epoch+1}", fontsize=16, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------

def train(cfg: TrainingConfig, matrix_cfg: MatrixConfig, device: torch.device) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ndim = 2
    dtype = torch.float32

    # Base distribution: N(0, base_variance*I)
    base_loc = torch.zeros(ndim, device=device, dtype=dtype)
    base_scale = torch.ones(ndim, device=device, dtype=dtype) * cfg.base_variance
    base = prior.SimpleNormal(base_loc, base_scale)

    def sample_base(bs: int) -> torch.Tensor:
        return base(bs).to(device=device, dtype=dtype)

    target_builder = TARGET_BUILDERS.get(cfg.target_type, sample_rect)

    def sample_target(bs: int) -> torch.Tensor:
        return target_builder(bs, device=device, dtype=dtype)

    # Build multivariate interpolant
    matrix_config_dict = {
        'matrix_type': matrix_cfg.matrix_type,
        'coefficient_type': matrix_cfg.coefficient_type,
        'exponent_p': matrix_cfg.exponent_p,
        'exponent_q': matrix_cfg.exponent_q,
    }

    base_interpolant = stochastic_interpolant.Interpolant(
        path="multivariate",
        gamma_type=None,
        data_type="vector",
        data_dim=ndim,
        matrix_config=matrix_config_dict
    ).to(device)

    # Auxiliary interpolant
    aux_interpolant = stochastic_interpolant.Interpolant(
        path="one-sided-trig",
        gamma_type=None,
    ).to(device)

    interpolant = MixedInterpolant(base_interpolant, aux_interpolant, mode='base').to(device)

    # Analyze kink locations
    kink_locs = _compute_kink_locations(matrix_cfg.exponent_p, matrix_cfg.exponent_q)
    print(f"\nKink analysis:")
    if kink_locs:
        for i, (t_center, importance) in enumerate(kink_locs, 1):
            print(f"  {i}. t ≈ {t_center:.3f} (importance: {importance:.2f})")
    else:
        print("  No kinks detected (linear interpolation)")

    # Velocity network - using lightweight UNet (from notebook)
    v_model = make_lightweight_unet_velocity_2d(
        hidden_dims=[128, 256, 256, 128],
        time_emb_dim=128,
        dropout=0.0,
        activation='gelu',
        wrapped=True
    ).to(device)

    loss_fn_sample = stochastic_interpolant.make_batch_loss(
        stochastic_interpolant.losses['one-sided-v'],
        method='shared',
    )

    # Optimizer and scheduler (following notebook)
    opt_v = torch.optim.AdamW(v_model.parameters(), lr=cfg.base_lr_v, betas=(0.9, 0.999), weight_decay=1e-4)

    # CosineAnnealingWarmRestarts as in notebook
    sched_v = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt_v,
        T_0=50 * cfg.n_inner,  # Restart every 50 epochs
        T_mult=2,
        eta_min=1e-6
    )

    results_dir = cfg.results_root / cfg.exp_name
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Persist config
    (results_dir / "config.json").write_text(json.dumps({
        "training": {k: str(v) if isinstance(v, Path) else v for k, v in cfg.__dict__.items()},
        "matrix_config": matrix_cfg.__dict__,
    }, indent=2))

    metrics_rows = []

    def train_step(epoch: int):
        if hasattr(interpolant, "set_step"):
            interpolant.set_step(epoch)

        v_model.train()
        v_model.requires_grad_(True)

        grad_v_norm = 0.0
        last_loss = torch.tensor(0.0, device=device)

        for _ in range(cfg.n_inner):
            opt_v.zero_grad(set_to_none=True)
            x0s = sample_base(cfg.batch_size)
            x1s = sample_target(cfg.batch_size)
            ts_batch, weights_batch = _sample_times_with_importance(
                cfg.batch_size, device, dtype,
                matrix_config_dict['exponent_p'],
                matrix_config_dict['exponent_q'],
                strategy='enhanced_beta_mixture',
                keepdim=False
            )
            per_sample_loss = loss_fn_sample(v_model, x0s, x1s, ts_batch, interpolant)
            weights = weights_batch.to(per_sample_loss.device, per_sample_loss.dtype)
            loss_v = (per_sample_loss * weights).mean()
            loss_v.backward()
            grad_v_norm = total_grad_norm(v_model.parameters())
            opt_v.step()
            sched_v.step()
            last_loss = loss_v.detach()

        return last_loss.item(), grad_v_norm

    print(f"Starting training: {cfg.exp_name}")
    print(f"Exponents: p={matrix_cfg.exponent_p}, q={matrix_cfg.exponent_q}")
    alpha, beta = _infer_beta_params(matrix_config_dict)
    print(f"Using enhanced_beta_mixture sampling: Beta({alpha:.2f},1) + Beta(1,{beta:.2f}) + Uniform")
    print("="*70)

    for epoch in range(cfg.num_epochs):
        start = time.time()
        loss_v, grad_v = train_step(epoch)
        v_sq = estimate_v_squared_norm_grid(
            v_model,
            interpolant,
            sample_base,
            sample_target,
            cfg.v_sq_samples,
            cfg.v_sq_time_points,
            cfg.v_sq_batch_size,
            device,
        )

        metrics_rows.append({
            "epoch": epoch + 1,
            "loss_v": loss_v,
            "grad_v": grad_v,
            "v_sq_norm": v_sq,
            "elapsed_sec": time.time() - start,
        })

        # Generate samples and save plot every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == cfg.num_epochs - 1:
            samples = rollout_samples(
                v_model,
                interpolant,
                sample_base,
                cfg.ode_steps,
                cfg.plot_batch_size,
                device,
            )

            plot_path = plots_dir / f"epoch_{epoch+1:04d}.png"
            save_epoch_plot(epoch, samples, metrics_rows, plot_path)

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} | v_loss={loss_v:.4f} | grad_v={grad_v:.4f} | "
            f"E[|v|^2]={v_sq:.4f} | time={metrics_rows[-1]['elapsed_sec']:.1f}s",
            flush=True,
        )

    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_rows, indent=2))
    print(f"Saved metrics to {metrics_path}")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polynomial exponent tuning for multivariate interpolants")
    parser.add_argument("--exp-name", required=True, help="Experiment name")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--exponent-p", nargs=2, type=float, required=True, help="Exponents for A(t)")
    parser.add_argument("--exponent-q", nargs=2, type=float, required=True, help="Exponents for B(t)")
    parser.add_argument("--scheduler", default="cosine-warm-restarts", help="Scheduler type")
    parser.add_argument("--num-epochs", type=int, default=250, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size")
    parser.add_argument("--n-inner", type=int, default=500, help="Inner optimization steps")
    parser.add_argument("--results-root", default="results_exponent_tuning", help="Results root")
    parser.add_argument("--device", default=None, help="Device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    training_cfg = TrainingConfig(
        exp_name=args.exp_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        n_inner=args.n_inner,
        base_lr_v=args.lr,
        results_root=Path(args.results_root),
        seed=args.seed,
    )

    matrix_cfg = MatrixConfig(
        matrix_type="diagonal",
        coefficient_type="polynomial",
        exponent_p=args.exponent_p,
        exponent_q=args.exponent_q,
    )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print(f"[exponent-tuning] Using device: {device}")
    print(f"[exponent-tuning] Learning rate: {args.lr}")
    print(f"[exponent-tuning] Batch size: {args.batch_size}")
    print(f"[exponent-tuning] Inner steps: {args.n_inner}")
    print(f"[exponent-tuning] Exponents p: {args.exponent_p}, q: {args.exponent_q}")
    print(f"[exponent-tuning] Scheduler: {args.scheduler}")

    train(training_cfg, matrix_cfg, device)


if __name__ == "__main__":
    main()
