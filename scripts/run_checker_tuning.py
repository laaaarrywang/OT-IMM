#!/usr/bin/env python3
"""Command-line training harness for the checkerboard nonlinear interpolant workflow.

This script mirrors the logic in `notebooks/checker-nonlinear-2D.ipynb` while making it
config-driven so it can be scheduled on the cluster. Each run logs per-epoch metrics and
saves side-by-side figures (generated samples + loss/E[|v|^2] curves) into
`results0929/<exp_name>/plots/`.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

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
    num_epochs: int = 100
    batch_size: int = 1000
    n_inner: int = 500
    n_outer: int = 10
    base_lr_v: float = 2e-3
    base_lr_flow: float = 1e-4
    step_gamma: float = 0.8
    step_interval_inner: int = 10
    step_interval_outer: int = 10
    v_sq_samples: int = 50_000
    v_sq_time_points: int = 100
    v_sq_batch_size: int = 2_000
    plot_batch_size: int = 10_000
    ode_steps: int = 100
    n_save: int = 10
    warmup_steps: int = 50
    seed: int = 0
    target_type: str = "checker"
    results_root: Path = Path("results0929/checker_tuning")


@dataclass
class FlowConfig:
    num_layers: int
    hidden: int
    mlp_blocks: int
    time_embed_dim: int = 128
    activation: str = "gelu"
    use_layernorm: bool = False
    use_permutation: bool = True
    fourier_min_freq: float = 1.0
    fourier_max_freq: float = 10.0
    vector_use_spectral_norm: bool = True
    vector_log_scale_clamp: float = 1.0
    vector_use_soft_clamp: bool = True


# --------------------------------------------------------------------------------------
# Target samplers (ported from notebook)
# --------------------------------------------------------------------------------------

def sample_checker(bs: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    x1 = torch.rand(bs, device=device, dtype=dtype) * 4 - 2
    parity = torch.randint(2, (bs,), device=device, dtype=torch.int64).to(dtype)
    x2 = torch.rand(bs, device=device, dtype=dtype) - 2 * parity
    x2 = x2 + torch.remainder(torch.floor(x1), torch.tensor(2.0, device=device, dtype=dtype))
    return torch.stack((x1, x2), dim=1) * 2.0


def sample_rect(
    bs: int,
    device: torch.device,
    dtype: torch.dtype,
    w: float = 1.0,
    h: float = 0.1,
    layers_x: int = 2,
    layers_y: int = 4,
    scale: float = 10.0,
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


def sample_rhombus(
    bs: int,
    device: torch.device,
    dtype: torch.dtype,
    s: float = 1.0,
    angle_deg: float = 60.0,
    layers_x: int = 4,
    layers_y: int = 4,
    scale: float = 2.0,
) -> torch.Tensor:
    i = torch.randint(layers_x, (bs,), device=device)
    j = torch.randint(layers_y, (bs,), device=device)

    u_in = torch.rand(bs, device=device, dtype=dtype)
    v_in = torch.rand(bs, device=device, dtype=dtype)

    u = i.to(dtype) + u_in
    v = (2 * j + (i % 2)).to(dtype) + v_in

    theta = math.radians(angle_deg)
    a = torch.tensor([s, 0.0], device=device, dtype=dtype)
    b = torch.tensor([s * math.cos(theta), s * math.sin(theta)], device=device, dtype=dtype)

    u_c = layers_x / 2.0
    v_c = layers_y
    pts = (u - u_c).unsqueeze(1) * a + (v - v_c).unsqueeze(1) * b
    return pts * scale


TARGET_BUILDERS = {
    "checker": sample_checker,
    "rect": sample_rect,
    "rhombus": sample_rhombus,
}


# --------------------------------------------------------------------------------------
# Warmup interpolant wrapper (ported from notebook)
# --------------------------------------------------------------------------------------

class WarmupInterpolant(torch.nn.Module):
    def __init__(self, interpolant_base: torch.nn.Module, interpolant_aux: torch.nn.Module, warmup_steps: int = 1000):
        super().__init__()
        self.interpolant_base = interpolant_base
        self.interpolant_aux = interpolant_aux
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def __getattr__(self, name: str):
        if name in {"interpolant_base", "interpolant_aux", "warmup_steps", "current_step"}:
            return super().__getattr__(name)
        if name in {"calc_xt", "dtIt"}:
            return super().__getattr__(name)
        return getattr(self.interpolant_base, name)

    def set_step(self, step: int) -> None:
        self.current_step = step

    def get_warmup_factor(self) -> float:
        if self.current_step >= self.warmup_steps:
            return 1.0
        return 1.0  # cosine warmup disabled as in notebook

    def calc_xt(self, t, x0, x1):
        xt_complex = self.interpolant_base.calc_xt(t, x0, x1)
        xt_simple = self.interpolant_aux.calc_xt(t, x0, x1)
        alpha = self.get_warmup_factor()
        return alpha * xt_complex + (1.0 - alpha) * xt_simple

    def dtIt(self, t, x0, x1):
        dt_complex = self.interpolant_base.dtIt(t, x0, x1)
        dt_simple = self.interpolant_aux.dtIt(t, x0, x1)
        alpha = self.get_warmup_factor()
        return alpha * dt_complex + (1.0 - alpha) * dt_simple


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
    interpolant: WarmupInterpolant,
    base_sampler,
    target_sampler,
    n_samples: int,
    n_time_points: int,
    batch_size: int,
    device: torch.device,
) -> float:
    v_mode = v.training
    flow_model = getattr(interpolant, "flow_model", None)
    flow_mode = flow_model.training if flow_model is not None else None

    v.eval()
    if flow_model is not None:
        flow_model.eval()

    times = torch.linspace(0.0, 1.0, n_time_points, device=device)
    samples_per_time = max(1, n_samples // n_time_points)

    total_sq = 0.0
    total_count = 0

    for t_scalar in times:
        remaining = samples_per_time
        while remaining > 0:
            current_bs = min(batch_size, remaining)
            x0s = base_sampler(current_bs)
            x1s = target_sampler(current_bs)

            t = torch.full((current_bs,), t_scalar.item(), device=device, dtype=x0s.dtype)
            xt = interpolant.calc_xt(t, x0s, x1s)
            if isinstance(xt, tuple):
                xt = xt[0]
            vt = v(xt, t)
            sq = vt.pow(2).sum(dim=-1)
            total_sq += sq.sum().item()
            total_count += sq.numel()
            remaining -= current_bs

    if v_mode:
        v.train()
    if flow_model is not None:
        if flow_mode:
            flow_model.train()
        else:
            flow_model.eval()

    return total_sq / max(total_count, 1)


def rollout_samples(
    v: torch.nn.Module,
    interpolant: WarmupInterpolant,
    base_sampler,
    n_save: int,
    n_step: int,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    base_batch = base_sampler(batch_size)
    logp0 = base_sampler.base.log_prob(base_batch)

    integrator = stochastic_interpolant.PFlowIntegrator(
        b=v,
        method="dopri5",
        interpolant=interpolant,
        n_step=n_step,
    )
    traj, dlogp = integrator.rollout(base_batch)
    xf = traj[-1].detach().cpu()
    logpx = (logp0 + dlogp[-1].detach()).cpu()
    return xf, logpx


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------

def save_epoch_plot(
    epoch: int,
    samples: torch.Tensor,
    logpx: torch.Tensor,
    metrics: Dict[str, torch.Tensor],
    out_path: Path,
) -> None:
    epochs = np.arange(1, epoch + 2)
    losses_v = [m["loss_v"] for m in metrics]
    losses_flow = [m["loss_flow"] for m in metrics]
    grads_v = [m["grad_v"] for m in metrics]
    grads_flow = [m["grad_flow"] for m in metrics]
    v_sq = [m["v_sq_norm"] for m in metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc = axes[0].scatter(samples[:, 0], samples[:, 1], c=logpx.exp().numpy(), s=3, alpha=0.4)
    axes[0].set_title("Generated samples (PFlow)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.2)
    fig.colorbar(sc, ax=axes[0], shrink=0.8, label="Density")

    ax_metrics = axes[1]
    ax_metrics.plot(epochs, losses_v, label="v loss", color="#1f77b4")
    ax_metrics.plot(epochs, losses_flow, label="flow loss", color="#ff7f0e")
    ax_metrics.plot(epochs, grads_v, label="||∇v||", color="#2ca02c", linestyle="--")
    ax_metrics.plot(epochs, grads_flow, label="||∇flow||", color="#d62728", linestyle="--")
    ax_metrics.set_xlabel("Epoch")
    ax_metrics.set_ylabel("Loss / Grad")
    ax_metrics.grid(alpha=0.3)

    ax_vsq = ax_metrics.twinx()
    ax_vsq.plot(epochs, v_sq, label="E[|v|²]", color="#9467bd")
    ax_vsq.set_ylabel("E[|v|²]")

    lines, labels = ax_metrics.get_legend_handles_labels()
    lines2, labels2 = ax_vsq.get_legend_handles_labels()
    axes[1].legend(lines + lines2, labels + labels2, loc="upper right")
    axes[1].set_title("Metrics")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# --------------------------------------------------------------------------------------
# Training loop
# --------------------------------------------------------------------------------------

def train(cfg: TrainingConfig, flow_cfg: FlowConfig, device: torch.device) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    ndim = 2
    dtype = torch.float32

    base_loc = torch.zeros(ndim, device=device, dtype=dtype)
    base_scale = torch.ones(ndim, device=device, dtype=dtype) * 3.0
    base = prior.SimpleNormal(base_loc, base_scale)

    def sample_base(bs: int) -> torch.Tensor:
        return base(bs).to(device=device, dtype=dtype)

    sample_base.base = base  # for log_prob reuse in rollout

    target_builder = TARGET_BUILDERS.get(cfg.target_type, sample_checker)

    def sample_target(bs: int) -> torch.Tensor:
        return target_builder(bs, device=device, dtype=dtype)

    # Build interpolants
    base_interpolant = stochastic_interpolant.Interpolant(
        path="nonlinear",
        gamma_type=None,
        flow_config=dict(
            num_layers=flow_cfg.num_layers,
            hidden=flow_cfg.hidden,
            mlp_blocks=flow_cfg.mlp_blocks,
            time_embed_dim=flow_cfg.time_embed_dim,
            activation=flow_cfg.activation,
            use_layernorm=flow_cfg.use_layernorm,
            use_permutation=flow_cfg.use_permutation,
            fourier_min_freq=flow_cfg.fourier_min_freq,
            fourier_max_freq=flow_cfg.fourier_max_freq,
            vector_use_spectral_norm=flow_cfg.vector_use_spectral_norm,
            vector_log_scale_clamp=flow_cfg.vector_log_scale_clamp,
            vector_use_soft_clamp=flow_cfg.vector_use_soft_clamp,
        ),
        data_type="vector",
        data_dim=ndim,
    ).to(device)

    aux_interpolant = stochastic_interpolant.Interpolant(
        path="one-sided-trig",
        gamma_type=None,
    ).to(device)

    interpolant = WarmupInterpolant(base_interpolant, aux_interpolant, warmup_steps=cfg.warmup_steps).to(device)

    v_model = fabrics_extra.make_resnet(
        hidden_sizes=[flow_cfg.hidden] * flow_cfg.mlp_blocks,
        in_size=ndim + 1,
        out_size=ndim,
        inner_act=flow_cfg.activation,
        use_layernorm=True,
        dropout=0.0,
    ).to(device)

    loss_fn_v = stochastic_interpolant.make_loss(method="shared", interpolant=interpolant, loss_type="one-sided-v")

    opt_v = torch.optim.AdamW(v_model.parameters(), lr=cfg.base_lr_v, betas=(0.5, 0.9), weight_decay=1e-4)
    opt_flow = torch.optim.AdamW(interpolant.flow_model.parameters(), lr=cfg.base_lr_flow, betas=(0.5, 0.9), weight_decay=1e-4)

    sched_v = torch.optim.lr_scheduler.StepLR(opt_v, step_size=cfg.step_interval_inner * cfg.n_inner, gamma=cfg.step_gamma)
    sched_flow = torch.optim.lr_scheduler.StepLR(opt_flow, step_size=cfg.step_interval_outer * cfg.n_outer, gamma=cfg.step_gamma)

    results_dir = cfg.results_root / cfg.exp_name
    plots_dir = results_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Persist config copy
    (results_dir / "config.json").write_text(json.dumps({
        "training": cfg.__dict__,
        "flow_config": flow_cfg.__dict__,
    }, indent=2))

    metrics_rows = []

    def train_step(epoch: int):
        if hasattr(interpolant, "set_step"):
            interpolant.set_step(epoch)

        v_model.train()
        interpolant.flow_model.train()

        # Inner minimization (v)
        interpolant.flow_model.requires_grad_(False)
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

        # Outer maximization (flow)
        interpolant.flow_model.requires_grad_(True)
        v_model.requires_grad_(False)

        grad_flow_norm = 0.0
        for _ in range(cfg.n_outer):
            opt_flow.zero_grad(set_to_none=True)
            x0s = sample_base(cfg.batch_size)
            x1s = sample_target(cfg.batch_size)
            ts = torch.rand(cfg.batch_size, device=device, dtype=dtype)
            loss_flow = loss_fn_v(v_model, x0s, x1s, ts, interpolant)
            (-loss_flow).backward()
            grad_flow_norm = total_grad_norm(interpolant.flow_model.parameters())
            opt_flow.step()
            sched_flow.step()

        v_model.requires_grad_(True)
        interpolant.flow_model.requires_grad_(True)

        return loss_v.detach().item(), loss_flow.detach().item(), grad_v_norm, grad_flow_norm

    for epoch in range(cfg.num_epochs):
        start = time.time()
        loss_v, loss_flow, grad_v, grad_flow = train_step(epoch)
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
            "loss_flow": loss_flow,
            "grad_v": grad_v,
            "grad_flow": grad_flow,
            "v_sq_norm": v_sq,
            "elapsed_sec": time.time() - start,
        })

        samples, logpx = rollout_samples(
            v_model,
            interpolant,
            sample_base,
            cfg.n_save,
            cfg.ode_steps,
            cfg.plot_batch_size,
            device,
        )

        plot_path = plots_dir / f"epoch_{epoch+1:04d}.png"
        save_epoch_plot(epoch, samples, logpx, metrics_rows, plot_path)

        print(
            f"Epoch {epoch+1}/{cfg.num_epochs} | v_loss={loss_v:.4f} | flow_loss={loss_flow:.4f} "
            f"| grad_v={grad_v:.4f} | grad_flow={grad_flow:.4f} | E[|v|^2]={v_sq:.4f} | "
            f"time={metrics_rows[-1]['elapsed_sec']:.1f}s",
            flush=True,
        )

    metrics_path = results_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_rows, indent=2))


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune checkerboard interpolant parameters")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--exp-name", default=None, help="Override experiment name")
    parser.add_argument("--results-root", default=None, help="Override results root directory")
    parser.add_argument("--device", default=None, help="Force device (cpu or cuda)")
    return parser.parse_args()


def load_configs(config_path: Path) -> Tuple[TrainingConfig, FlowConfig]:
    cfg_dict = json.loads(config_path.read_text())

    train_cfg_dict = cfg_dict.get("training", {})
    flow_cfg_dict = cfg_dict.get("flow_config", {})

    exp_name = cfg_dict.get("exp_name") or train_cfg_dict.get("exp_name") or config_path.stem
    train_cfg_dict.setdefault("exp_name", exp_name)

    training_cfg = TrainingConfig(**train_cfg_dict)
    flow_cfg = FlowConfig(**flow_cfg_dict)
    return training_cfg, flow_cfg


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    training_cfg, flow_cfg = load_configs(config_path)

    if args.exp_name:
        training_cfg.exp_name = args.exp_name
    if args.results_root:
        training_cfg.results_root = Path(args.results_root)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[checker-tuning] Using device: {device}")

    # Ensure results root exists before training
    training_cfg.results_root = Path(training_cfg.results_root)
    training_cfg.results_root.mkdir(parents=True, exist_ok=True)

    train(training_cfg, flow_cfg, device)


if __name__ == "__main__":
    main()
