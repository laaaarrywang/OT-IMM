#!/usr/bin/env python
"""
Run checker-nonlinear-2D experiments with configurable hyperparameters.
Saves plots and metrics at each epoch to specified output directory.
"""

import os
import sys
import json
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import math
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import interflow as itf
import interflow.prior as prior
import interflow.fabrics
import interflow.fabrics_extra
import interflow.stochastic_interpolant as stochastic_interpolant
from torch import autograd

# Set up device
if torch.cuda.is_available():
    print('CUDA available, setting default tensor residence to GPU.')
    itf.util.set_torch_device('cuda')
    device = torch.device("cuda")
else:
    print('No CUDA device found!')
    device = torch.device("cpu")

print(f"Device: {device}")
print(f"Torch version: {torch.__version__}")

# Utility functions
def grab(var):
    """Take a tensor off the gpu and convert it to a numpy array on the CPU."""
    return var.detach().cpu().numpy()

def total_grad_norm(params):
    norms = []
    for p in params:
        if p.grad is not None:
            norms.append(p.grad.detach().pow(2).sum())
    if not norms:
        return 0.0
    return (torch.stack(norms).sum().sqrt()).item()

def set_requires_grad(module, flag: bool):
    for p in module.parameters():
        p.requires_grad_(flag)

def estimate_v_squared_norm(v, interpolant, base, target, n_samples=1000, batch_size=100, device=None):
    """Estimate E_{t,x_0,x_1}[|v_t(I_t(x0,x1))|^2] using Monte Carlo."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v.eval()
    if hasattr(interpolant, 'flow_model'):
        interpolant.flow_model.eval()

    v_squared_norms = []

    with torch.no_grad():
        n_time_samples = max(1, n_samples // batch_size)
        samples_per_t = min(batch_size, n_samples)

        ts = torch.linspace(0, 1, n_time_samples, device=device)
        for t in ts:
            t = t.unsqueeze(0)
            x0s = base(samples_per_t).to(device)
            x1s = target(samples_per_t).to(device)

            xts = interpolant.calc_xt(t, x0s, x1s)
            if isinstance(xts, tuple):
                xts = xts[0]

            vts = v(xts, t)
            v_squared = torch.sum(vts ** 2, dim=-1)
            v_squared_norms.extend(v_squared.cpu().numpy())

    v.train()
    if hasattr(interpolant, 'flow_model'):
        interpolant.flow_model.train()

    return float(np.mean(v_squared_norms))

# Target functions
def target(bs):
    """Square checkerboard target."""
    x1 = torch.rand(bs, dtype=torch.float32) * 4 - 2
    x2_ = torch.rand(bs, dtype=torch.float32) - torch.randint(2, (bs,), dtype=torch.float32) * 2
    x2 = x2_ + (torch.floor(x1) % 2)
    return (torch.cat([x1[:, None], x2[:, None]], 1) * 2)

def target_rect(bs, w=1.0, h=0.1, layers_x=2, layers_y=4, scale=10.0, device=None, dtype=None):
    """Rectangular checkerboard with adjustable aspect ratio."""
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()

    col = torch.randint(layers_x, (bs,), device=device)
    base_row = torch.randint(layers_y, (bs,), device=device)

    x_in = torch.rand(bs, dtype=dtype, device=device) * w
    x = (col.to(dtype) + x_in / w) * w

    y_in = torch.rand(bs, dtype=dtype, device=device) * h
    y = (base_row.to(dtype) * 2.0 * h) + ((col % 2).to(dtype) * h) + y_in

    x = x - (layers_x * w) / 2.0
    y = y - (layers_y * h)
    out = torch.stack([x, y], dim=1) * scale
    return out

def target_rhombus(bs,
                   s=1.0,                 # side length of the rhombus (both edges equal)
                   angle_deg=60.0,        # interior angle between edges a and b (0<angle<180, != 0/180)
                   layers_x=4,            # how many tiles along the 'a' direction
                   layers_y=4,            # how many (checker) pairs along the 'b' direction
                   scale=2.0,
                   device=None, dtype=None):
    """
    True rhombus checkerboard sampler.

    Construction:
      - Lattice edges: a and b with |a|=|b|=s and angle `angle_deg` between them.
      - Checker staggering: along b, every odd a-column is shifted by +b/2 (i%2).
      - Indices: u in [0, layers_x), v in [0, 2*layers_y) with parity offset.
      - We sample uniformly inside a single rhombus cell by adding u_in, v_in ∈ [0,1).

    Output:
      Tensor of shape [bs, 2].
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.get_default_dtype()

    # Integer tile indices for the checker tiling
    i = torch.randint(layers_x, (bs,), device=device)           # column along a
    j = torch.randint(layers_y, (bs,), device=device)           # row-pair along b

    # Uniform offsets inside a rhombus cell
    u_in = torch.rand(bs, device=device, dtype=dtype)           # along a
    v_in = torch.rand(bs, device=device, dtype=dtype)           # along b

    # Staggering to make a checker: v steps by 2, with +1 for odd columns
    u = i.to(dtype) + u_in                                      # [0, layers_x)
    v = (2 * j + (i % 2)).to(dtype) + v_in                      # [0, 2*layers_y)

    # Build rhombus lattice basis (equal side lengths, non-orthogonal)
    theta = math.radians(angle_deg)
    a = torch.tensor([s, 0.0], device=device, dtype=dtype)                      # |a|=s
    b = torch.tensor([s * math.cos(theta), s * math.sin(theta)],                # |b|=s
                     device=device, dtype=dtype)

    # Center the cloud approximately at the origin
    u_c = layers_x / 2.0
    v_c = layers_y       # because v spans 2*layers_y "b-steps"
    pts = (u - u_c).unsqueeze(1) * a + (v - v_c).unsqueeze(1) * b

    return pts * scale

# WarmupInterpolant class
class WarmupInterpolant(torch.nn.Module):
    def __init__(self, interpolant_base, interpolant_aux, warmup_steps=1000):
        super().__init__()
        self.interpolant_base = interpolant_base
        self.interpolant_aux = interpolant_aux
        self.warmup_steps = warmup_steps
        self.current_step = 0

    def __getattr__(self, name):
        if name in ['interpolant_base','interpolant_aux', 'warmup_steps', 'current_step']:
            return super().__getattr__(name)
        if name in ['calc_xt', 'dtIt']:
            return super().__getattr__(name)
        return getattr(self.interpolant_base, name)

    def set_step(self, step):
        self.current_step = step

    def get_warmup_factor(self):
        return 1.0  

    def calc_xt(self, t, x0, x1):
        bs = x0.shape[0]
        xt_complex = self.interpolant_base.calc_xt(t, x0, x1)
        xt_simple = self.interpolant_aux.calc_xt(t, x0, x1)
        alpha = self.get_warmup_factor()
        return alpha * xt_complex + (1 - alpha) * xt_simple

    def dtIt(self, t, x0, x1):
        dtIt_complex = self.interpolant_base.dtIt(t, x0, x1)
        dtIt_simple = self.interpolant_aux.dtIt(t, x0, x1)
        alpha = self.get_warmup_factor()
        return alpha * dtIt_complex + (1 - alpha) * dtIt_simple

# Training functions
def train_step(bs, interpolant, v, opt_v, opt_flow, sched_v, sched_flow, loss_fn_v,
               n_inner=1, n_outer=1, reuse_batch=True, clip_v=1.0, clip_flow=1.0,
               device=None, train_v_only=False, warmup_step=0, base_fn=None, target_fn=None):
    """Take a single step of adversarial training."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(interpolant, 'set_step'):
        interpolant.set_step(warmup_step)

    v.train()
    interpolant.flow_model.train()

    if reuse_batch:
        x0s = base_fn(bs)
        x1s = target_fn(bs)
        ts = torch.rand(size=(bs,1))

    # Inner minimization: minimize over v
    set_requires_grad(interpolant.flow_model, False)
    set_requires_grad(v, True)

    v_grad_val = 0.0
    for _ in range(n_inner):
        if not reuse_batch:
            x0s = base_fn(bs).to(device)
            x1s = target_fn(bs).to(device)
            ts = torch.rand(bs, device=device).reshape(bs, 1)

        opt_v.zero_grad(set_to_none=True)
        loss_v = loss_fn_v(v, x0s, x1s, ts, interpolant)
        loss_v.backward()

        if clip_v is not None and clip_v < float('inf'):
            torch.nn.utils.clip_grad_norm_(v.parameters(), clip_v)
        v_grad_val = total_grad_norm(v.parameters())

        opt_v.step()
        sched_v.step()

    if not train_v_only:
        # Outer maximization: maximize over interpolant.flow_model
        set_requires_grad(v, False)
        set_requires_grad(interpolant.flow_model, True)

        flow_grad_val = 0.0
        for _ in range(n_outer):
            if not reuse_batch:
                x0s = base_fn(bs).to(device)
                x1s = target_fn(bs).to(device)
                ts = torch.rand(bs, device=device).reshape(bs, 1)

            opt_flow.zero_grad(set_to_none=True)
            loss_flow = loss_fn_v(v, x0s, x1s, ts, interpolant)
            (-loss_flow).backward()  # Negative for maximization

            if clip_flow is not None and clip_flow < float('inf'):
                torch.nn.utils.clip_grad_norm_(interpolant.flow_model.parameters(), clip_flow)
            flow_grad_val = total_grad_norm(interpolant.flow_model.parameters())

            opt_flow.step()
            sched_flow.step()

        set_requires_grad(v, True)
        set_requires_grad(interpolant.flow_model, True)
        return loss_v.detach(), loss_flow.detach(), torch.tensor(v_grad_val), torch.tensor(flow_grad_val)
    else:
        set_requires_grad(v, True)
        set_requires_grad(interpolant.flow_model, True)
        return loss_v.detach(), torch.tensor(v_grad_val)

def make_plots(v, interpolant, base, target, epoch, output_dir, data_dict):
    """Generate and save plots for current epoch."""
    from interflow.stochastic_interpolant import PFlowIntegrator

    # Sample from probability flow
    pflow = PFlowIntegrator(b=v, method='dopri5', interpolant=interpolant, n_step=3)

    bs = 10000
    x0_tests = base(bs)
    logp0 = base.log_prob(x0_tests)

    with torch.no_grad():
        xfs_pflow, dlogp_pflow = pflow.rollout(x0_tests)
        logpx_pflow = logp0 + dlogp_pflow[-1].squeeze()
        xf_pflow = grab(xfs_pflow[-1].squeeze())

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot samples from PFlow
    axes[0].scatter(xf_pflow[:, 0], xf_pflow[:, 1], vmin=0.0, vmax=0.05, alpha=0.2,
                   c=grab(torch.exp(logpx_pflow).detach()))
    axes[0].set_xlim(-12, 12)
    axes[0].set_ylim(-6.5, 6.5)
    axes[0].set_title(f"PFlow Samples (Epoch {epoch})", fontsize=14)
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")

    # Plot loss history
    if len(data_dict['v_losses']) > 0:
        epochs = np.arange(len(data_dict['v_losses']))
        axes[1].plot(epochs, data_dict['v_losses'], label='v loss')
        axes[1].plot(epochs, data_dict['flow_losses'], label='flow loss')
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Training Losses")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    # Plot E[|v|^2]
    if len(data_dict['v_squared_norm']) > 0:
        epochs = np.arange(len(data_dict['v_squared_norm']))
        axes[2].plot(epochs, data_dict['v_squared_norm'])
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel(r"$E[|v|^2]$")
        axes[2].set_title(r"Velocity Squared Norm")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(output_dir, f'epoch_{epoch:04d}.png')
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Saved plot to {plot_path}")

def main(config):
    """Main training function."""
    # Create output directory
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Setup
    ndim = 2

    # Base distribution
    base_loc = torch.zeros(ndim, device=device, dtype=torch.float32)
    base_var = torch.ones(ndim, device=device, dtype=torch.float32)
    base = prior.SimpleNormal(base_loc, 3 * base_var)

    # Create interpolants
    flow_config = {
        "num_layers": config.get('num_layers', 2),
        "time_embed_dim": config.get('time_embed_dim', 128),
        "hidden": config.get('hidden', 128),
        "mlp_blocks": config.get('mlp_blocks', 2),
        "activation": config.get('activation', 'gelu'),
        "use_layernorm": config.get('use_layernorm', False),
        "use_permutation": config.get('use_permutation', True),
        "fourier_min_freq": config.get('fourier_min_freq', 1.0),
        "fourier_max_freq": config.get('fourier_max_freq', 10.0),
        "vector_use_spectral_norm": config.get('vector_use_spectral_norm', True),
        "vector_log_scale_clamp": config.get('vector_log_scale_clamp', 1.0),
        "vector_use_soft_clamp": config.get('vector_use_soft_clamp', True),
    }

    # Main nonlinear interpolant
    interpolant = stochastic_interpolant.Interpolant(
        path="nonlinear",
        gamma_type=None,
        flow_config=flow_config,
        data_type="vector",
        data_dim=ndim
    )

    # Auxiliary linear interpolant
    diagonal_scale = torch.tensor([1.0, 1.0], device=device, dtype=torch.float32)
    interpolant_aux = stochastic_interpolant.Interpolant(
        path='one-sided-linear',
        gamma_type=None,
        diagonal_scale=diagonal_scale
    )

    # Warmup interpolant
    N_warmup = config.get('N_warmup', 50)
    warmup_interpolant = WarmupInterpolant(interpolant, interpolant_aux, warmup_steps=N_warmup)

    # Create velocity network v
    v_hidden = config.get('v_hidden_sizes', [256, 256, 256, 256])
    in_size = ndim + 1
    out_size = ndim

    v = itf.fabrics_extra.make_resnet(
        hidden_sizes=v_hidden,
        in_size=in_size,
        out_size=out_size,
        inner_act='gelu',
        use_layernorm=True,
        dropout=0.0
    )

    # Optimizers
    base_lr_v = config.get('lr_v', 2e-3)
    base_lr_flow = config.get('lr_flow', 2e-4)
    opt_v = torch.optim.AdamW(v.parameters(), lr=base_lr_v, betas=(0.5, 0.9), weight_decay=1e-4)
    opt_flow = torch.optim.AdamW(warmup_interpolant.flow_model.parameters(),
                                 lr=base_lr_flow, betas=(0.5, 0.9), weight_decay=1e-4)

    

    # Training parameters
    N_epoch = config.get('N_epoch', 50)
    bs = config.get('batch_size', 1000)
    n_inner = config.get('n_inner', 500)
    n_outer = config.get('n_outer', 5)
    plot_freq = config.get('plot_freq', 1)
    
    # Schedulers
    gamma = config.get('scheduler_gamma', 0.8)
    sched_v = torch.optim.lr_scheduler.StepLR(optimizer=opt_v, step_size=10*n_inner, gamma=gamma)
    sched_flow = torch.optim.lr_scheduler.StepLR(optimizer=opt_flow, step_size=10*n_outer, gamma=gamma)

    # Data dictionary for metrics
    data_dict = {
        'v_losses': [],
        'flow_losses': [],
        'v_grads': [],
        'flow_grads': [],
        'v_squared_norm': [],
    }

    # Loss function
    loss_fn_v = stochastic_interpolant.make_loss(
        method='shared',
        interpolant=warmup_interpolant,
        loss_type='one-sided-v'
    )

    # Training loop
    print(f"\nStarting training for {N_epoch} epochs...")
    print(f"Output directory: {output_dir}")

    for epoch in range(N_epoch):
        # Train step
        loss_v, loss_flow, v_grad, flow_grad = train_step(
            bs=bs,
            interpolant=warmup_interpolant,
            v=v,
            opt_v=opt_v,
            opt_flow=opt_flow,
            sched_v=sched_v,
            sched_flow=sched_flow,
            loss_fn_v=loss_fn_v,
            n_inner=n_inner,
            n_outer=n_outer,
            reuse_batch=False,
            clip_v=float('inf'),
            clip_flow=float('inf'),
            train_v_only=False,
            warmup_step=epoch,
            base_fn=base,
            target_fn=target_rhombus,
            device=device
        )

        # Log metrics
        data_dict['v_losses'].append(grab(loss_v).mean())
        data_dict['flow_losses'].append(grab(loss_flow).mean())
        data_dict['v_grads'].append(grab(v_grad).mean())
        data_dict['flow_grads'].append(grab(flow_grad).mean())

        # Estimate E[|v|^2]
        v_sq_norm = estimate_v_squared_norm(v, warmup_interpolant, base, target_rhombus,
                                           n_samples=10000, batch_size=500, device=device)
        data_dict['v_squared_norm'].append(v_sq_norm)

        # Print progress
        print(f"Epoch {epoch+1}/{N_epoch}: v_loss={loss_v.item():.4f}, "
              f"flow_loss={loss_flow.item():.4f}, v_grad={v_grad.item():.4f}, "
              f"flow_grad={flow_grad.item():.4f}, E[|v|²]={v_sq_norm:.4f}")

        # Generate plots
        if (epoch + 1) % plot_freq == 0:
            make_plots(v, warmup_interpolant, base, target_rhombus, epoch + 1, output_dir, data_dict)

    # Save final metrics
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(data_dict, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")

    # Save model checkpoints
    checkpoint = {
        'epoch': N_epoch,
        'v_state_dict': v.state_dict(),
        'flow_model_state_dict': warmup_interpolant.flow_model.state_dict(),
        'opt_v_state_dict': opt_v.state_dict(),
        'opt_flow_state_dict': opt_flow.state_dict(),
        'config': config
    }
    checkpoint_path = output_dir / 'checkpoint.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run checker-nonlinear-2D experiments")
    parser.add_argument('--config', type=str, required=True, help='Path to config JSON file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    main(config)