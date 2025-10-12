#!/usr/bin/env python3
"""Command-line training harness for multivariate interpolant experiments.

This script mirrors the logic in `notebooks/checker-multivariateSI-2D.ipynb` for
cluster execution with configurable matrix coefficients as hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple, List

import matplotlib
matplotlib.use("Agg")  # headless plotting for cluster jobs
import matplotlib.pyplot as plt
import numpy as np
import torch

import interflow as itf
import interflow.fabrics_extra as fabrics_extra
import interflow.prior as prior
import interflow.stochastic_interpolant as stochastic_interpolant


# --------------------------------------------------------------------------------------
# Helper dataclasses
# --------------------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    exp_name: str
    num_epochs: int = 500
    batch_size: int = 1000
    n_inner: int = 1000
    base_lr_v: float = 2e-3
    step_gamma: float = 0.8
    step_interval_inner: int = 20
    v_sq_samples: int = 50_000
    v_sq_time_points: int = 100
    v_sq_batch_size: int = 2_000
    plot_batch_size: int = 10_000
    ode_steps: int = 100
    n_save: int = 10
    seed: int = 0
    base_variance: float = 3.0
    target_type: str = "rect"
    results_root: Path = Path("results_multivariate")


@dataclass
class MatrixConfig:
    matrix_type: str = "diagonal"
    exponent_p: List[float] = None
    exponent_q: List[float] = None

    def __post_init__(self):
        if self.exponent_p is None:
            self.exponent_p = [1.0, 1.0]
        if self.exponent_q is None:
            self.exponent_q = [1.0, 1.0]


# --------------------------------------------------------------------------------------
# Target samplers (ported from notebook)
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
# MixedInterpolant wrapper (ported from notebook)
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

    def get_mixed_factor(self):
        if self.mode == 'base':
            return 1.0
        elif self.mode == 'aux':
            return 0.0
        else:  # mixed mode
            if self.current_step >= self.mixed_steps:
                return 1.0
            return 0.5 * (1 - np.cos(np.pi * self.current_step / self.mixed_steps))

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

def visualize_multivariate_trajectories(
    interpolant,
    base_sampler,
    target_sampler,
    matrix_config: MatrixConfig,
    out_path: Path,
    n_trajectories: int = 20,
    seed: int = 42,
    device: torch.device = None,
):
    """Visualize multivariate interpolant trajectories (save once)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Use provided device or detect from CUDA availability
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get base (multivariate) interpolant
    if hasattr(interpolant, 'interpolant_base'):
        multivar_interpolant = interpolant.interpolant_base
    else:
        multivar_interpolant = interpolant

    # Generate fixed trajectory endpoints
    x0_viz = base_sampler(n_trajectories)
    x1_viz = target_sampler(n_trajectories)

    # Compute axis limits
    x_min = min(x0_viz[:, 0].min().item(), x1_viz[:, 0].min().item())
    x_max = max(x0_viz[:, 0].max().item(), x1_viz[:, 0].max().item())
    y_min = min(x0_viz[:, 1].min().item(), x1_viz[:, 1].min().item())
    y_max = max(x0_viz[:, 1].max().item(), x1_viz[:, 1].max().item())

    x_range = x_max - x_min
    y_range = y_max - y_min
    x_lim = (x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    y_lim = (y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    t_dense = torch.linspace(0, 1, 50, device=device)
    colors = plt.cm.viridis(np.linspace(0, 1, len(t_dense)))

    with torch.no_grad():
        for i in range(n_trajectories):
            trajectory = []
            for t in t_dense:
                t_tensor = t.unsqueeze(0)
                x_mv = multivar_interpolant.calc_xt(t_tensor, x0_viz[i:i+1], x1_viz[i:i+1])
                if isinstance(x_mv, tuple):
                    x_mv = x_mv[0]
                trajectory.append(x_mv[0].cpu().numpy())

            trajectory = np.array(trajectory)

            # Plot with gradient color
            for j in range(len(trajectory) - 1):
                ax.plot(trajectory[j:j+2, 0], trajectory[j:j+2, 1],
                       '-', color=colors[j], alpha=0.7, linewidth=2)

            # Mark endpoints
            ax.plot(trajectory[0, 0], trajectory[0, 1], 'o',
                   color='green', markersize=8, alpha=0.8,
                   label='Start (t=0)' if i == 0 else '')
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], 's',
                   color='red', markersize=8, alpha=0.8,
                   label='End (t=1)' if i == 0 else '')

    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title(f'Multivariate Interpolant: p={matrix_config.exponent_p}, q={matrix_config.exponent_q}', fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal', adjustable='box')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_epoch_plot(
    epoch: int,
    samples: torch.Tensor,
    metrics: List[Dict],
    out_path: Path,
    width: float = 10.0,
    height: float = 0.4,
) -> None:
    epochs = np.arange(1, epoch + 2)
    losses_v = [m["loss_v"] for m in metrics]
    v_sq = [m["v_sq_norm"] for m in metrics]
    grads_v = [m["grad_v"] for m in metrics]

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

    base_loc = torch.zeros(ndim, device=device, dtype=dtype)
    base_scale = torch.ones(ndim, device=device, dtype=dtype) * cfg.base_variance
    base = prior.SimpleNormal(base_loc, base_scale)

    def sample_base(bs: int) -> torch.Tensor:
        return base(bs).to(device=device, dtype=dtype)

    target_builder = TARGET_BUILDERS.get(cfg.target_type, sample_rect)

    def sample_target(bs: int) -> torch.Tensor:
        return target_builder(bs, device=device, dtype=dtype)

    # Build multivariate interpolant
    base_interpolant = stochastic_interpolant.Interpolant(
        path="multivariate",
        gamma_type=None,
        data_type="vector",
        data_dim=ndim,
        matrix_config={
            'matrix_type': matrix_cfg.matrix_type,
            'exponent_p': matrix_cfg.exponent_p,
            'exponent_q': matrix_cfg.exponent_q,
        }
    ).to(device)

    # Auxiliary interpolant for comparison
    aux_interpolant = stochastic_interpolant.Interpolant(
        path="one-sided-trig",
        gamma_type=None,
    ).to(device)

    interpolant = MixedInterpolant(base_interpolant, aux_interpolant, mode='base').to(device)

    # Velocity network
    v_model = fabrics_extra.make_resnet(
        hidden_sizes=[256, 256, 256, 256],
        in_size=ndim + 1,
        out_size=ndim,
        inner_act='gelu',
        use_layernorm=True,
        dropout=0.0,
    ).to(device)

    loss_fn_v = stochastic_interpolant.make_loss(method="shared", interpolant=interpolant, loss_type="one-sided-v")

    opt_v = torch.optim.AdamW(v_model.parameters(), lr=cfg.base_lr_v, betas=(0.5, 0.9), weight_decay=1e-4)
    sched_v = torch.optim.lr_scheduler.StepLR(opt_v, step_size=cfg.step_interval_inner * cfg.n_inner, gamma=cfg.step_gamma)

    results_dir = cfg.results_root / cfg.exp_name
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Save interpolant trajectories once at the beginning
    traj_path = results_dir / "multivariate_trajectories.png"
    visualize_multivariate_trajectories(
        interpolant, sample_base, sample_target, matrix_cfg, traj_path, n_trajectories=20, device=device
    )
    print(f"Saved interpolant trajectories to {traj_path}")

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

        # Only train velocity (no flow parameters)
        v_model.requires_grad_(True)

        grad_v_norm = 0.0
        for _ in range(cfg.n_inner):
            opt_v.zero_grad(set_to_none=True)
            x0s = sample_base(cfg.batch_size)
            x1s = sample_target(cfg.batch_size)
            ts = torch.rand(cfg.batch_size, device=device, dtype=dtype)
            loss_v = loss_fn_v(v_model, x0s, x1s, ts, interpolant)
            loss_v.backward()
            grad_v_norm = total_grad_norm(v_model.parameters())
            opt_v.step()
            sched_v.step()

        return loss_v.detach().item(), grad_v_norm

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
    parser = argparse.ArgumentParser(description="Train multivariate interpolant with matrix coefficients")
    parser.add_argument("--exp-name", required=True, help="Experiment name")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate for velocity network")
    parser.add_argument("--exponent-p", nargs=2, type=float, required=True, help="Exponents for A(t) (2 values)")
    parser.add_argument("--exponent-q", nargs=2, type=float, required=True, help="Exponents for B(t) (2 values)")
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--results-root", default="results_multivariate", help="Results root directory")
    parser.add_argument("--device", default=None, help="Force device (cpu or cuda)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()

    training_cfg = TrainingConfig(
        exp_name=args.exp_name,
        num_epochs=args.num_epochs,
        base_lr_v=args.lr,
        results_root=Path(args.results_root),
        seed=args.seed,
    )

    matrix_cfg = MatrixConfig(
        matrix_type="diagonal",
        exponent_p=args.exponent_p,
        exponent_q=args.exponent_q,
    )

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[multivariate-tuning] Using device: {device}")
    print(f"[multivariate-tuning] Learning rate: {args.lr}")
    print(f"[multivariate-tuning] Exponents p: {args.exponent_p}, q: {args.exponent_q}")

    train(training_cfg, matrix_cfg, device)


if __name__ == "__main__":
    main()
