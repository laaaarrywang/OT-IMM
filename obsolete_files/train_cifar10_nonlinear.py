#!/usr/bin/env python
"""
Train nonlinear stochastic interpolant on CIFAR-10 data.
This script runs training without generating plots for headless execution.
"""

import os
import sys
import torch
import numpy as np
import argparse
import json
import time
from datetime import datetime
import pickle

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torchvision
import torchvision.transforms as transforms
import interflow as itf
import interflow.prior as prior
import interflow.fabrics
import interflow.fabrics_extra
import interflow.stochastic_interpolant as stochastic_interpolant
import interflow.realnvp as realnvp

# Set device
if torch.cuda.is_available():
    print('CUDA available, setting default tensor residence to GPU.')
    itf.util.set_torch_device('cuda')
    device = torch.device("cuda")
else:
    print('No CUDA device found, using CPU.')
    device = torch.device("cpu")

print(f"Device: {itf.util.get_torch_device()}")
print(f"Torch version: {torch.__version__}")


class CIFAR10Target:
    """CIFAR-10 target distribution."""
    def __init__(self, dataset, device='cuda'):
        self.dataset = dataset
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.indices = list(range(len(dataset)))

    def __call__(self, batch_size, flatten=True):
        batch_indices = np.random.choice(self.indices, size=batch_size, replace=True)
        samples = []
        for idx in batch_indices:
            img, _ = self.dataset[idx]
            samples.append(img)
        samples = torch.stack(samples).to(self.device)
        if flatten:
            samples = samples.view(batch_size, -1)
        return samples


def setup_data(cifar_dir="/scratch/gautschi/wang6559/cifar10"):
    """Setup CIFAR-10 dataset and distributions."""
    print("Loading CIFAR-10 dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=cifar_dir,
        train=True,
        download=True,
        transform=transform
    )

    target = CIFAR10Target(trainset, device=device)

    # Base distribution
    ndim = 3 * 32 * 32  # 3072
    base_loc = torch.zeros(ndim, device=device, dtype=torch.float32)
    base_var = torch.ones(ndim, device=device, dtype=torch.float32)
    base = prior.SimpleNormal(base_loc, base_var)

    return base, target, ndim


def setup_models(config, ndim):
    """Setup interpolant and velocity models."""
    print("Setting up models...")

    # Interpolant configuration
    flow_config = config.get('flow_config', {
        "num_layers": 2,
        "time_embed_dim": 256,
        "img_base_channels": 128,
        "img_blocks": 2,
        "img_groups": 32,
        "img_log_scale_clamp": 5.0,
        "img_use_soft_clamp": True,
        "use_permutation": True,
        "fourier_min_freq": 1.0,
        "fourier_max_freq": 150.0,
    })

    interpolant = stochastic_interpolant.Interpolant(
        path="nonlinear",
        gamma_type=None,
        flow_config=flow_config,
        data_type="cifar10"
    )

    # Velocity model
    hidden_sizes = config.get('hidden_sizes', [256, 256, 256, 256])
    v = itf.fabrics_extra.make_resnet(
        hidden_sizes=hidden_sizes,
        in_size=ndim + 1,
        out_size=ndim,
        inner_act='gelu',
        use_layernorm=True,
        dropout=0.0
    )

    trainable_params = sum(p.numel() for p in interpolant.flow_model.parameters())
    print(f"Flow model parameters: {trainable_params:,}")

    v_params = sum(p.numel() for p in v.parameters())
    print(f"Velocity model parameters: {v_params:,}")

    return interpolant, v


def train_step(bs, interpolant, v, opt_v, opt_flow, loss_fn_v,
               base, target, n_inner, n_outer, device):
    """Single training step."""

    def set_requires_grad(module, flag):
        for p in module.parameters():
            p.requires_grad_(flag)

    v.train()
    interpolant.flow_model.train()

    # Inner minimization: minimize over v
    set_requires_grad(interpolant.flow_model, False)
    set_requires_grad(v, True)

    loss_v_total = 0
    for _ in range(n_inner):
        x0s = base(bs).to(device)
        x1s = target(bs, flatten=True).to(device)
        ts = torch.rand(bs, device=device)

        opt_v.zero_grad(set_to_none=True)
        loss_v = loss_fn_v(v, x0s, x1s, ts, interpolant)
        loss_v.backward()
        opt_v.step()
        loss_v_total += loss_v.item()

    # Outer maximization: maximize over interpolant.flow_model
    set_requires_grad(v, False)
    set_requires_grad(interpolant.flow_model, True)

    loss_flow_total = 0
    for _ in range(n_outer):
        x0s = base(bs).to(device)
        x1s = target(bs, flatten=True).to(device)
        ts = torch.rand(bs, device=device)

        opt_flow.zero_grad(set_to_none=True)
        loss_flow = loss_fn_v(v, x0s, x1s, ts, interpolant)
        (-loss_flow).backward()  # Negative for maximization
        opt_flow.step()
        loss_flow_total += loss_flow.item()

    set_requires_grad(v, True)
    set_requires_grad(interpolant.flow_model, True)

    return loss_v_total / n_inner, loss_flow_total / n_outer


def save_checkpoint(epoch, interpolant, v, opt_v, opt_flow, metrics, save_dir):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save({
        'epoch': epoch,
        'interpolant_state': interpolant.flow_model.state_dict(),
        'v_state': v.state_dict(),
        'opt_v_state': opt_v.state_dict(),
        'opt_flow_state': opt_flow.state_dict(),
        'metrics': metrics
    }, checkpoint_path)
    print(f"  Checkpoint saved: {checkpoint_path}")

    # Also save as latest
    latest_path = os.path.join(save_dir, "checkpoint_latest.pt")
    torch.save({
        'epoch': epoch,
        'interpolant_state': interpolant.flow_model.state_dict(),
        'v_state': v.state_dict(),
        'opt_v_state': opt_v.state_dict(),
        'opt_flow_state': opt_flow.state_dict(),
        'metrics': metrics
    }, latest_path)


def main(config_file=None):
    """Main training function."""

    parser = argparse.ArgumentParser(description='Train CIFAR-10 nonlinear stochastic interpolant')
    parser.add_argument('--config', type=str, default=None, help='Config JSON file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=100, help='Batch size')
    parser.add_argument('--n_inner', type=int, default=200, help='Inner optimization steps')
    parser.add_argument('--n_outer', type=int, default=5, help='Outer optimization steps')
    parser.add_argument('--lr_v', type=float, default=2e-3, help='Learning rate for velocity')
    parser.add_argument('--lr_flow', type=float, default=2e-4, help='Learning rate for flow')
    parser.add_argument('--save_dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--checkpoint_freq', type=int, default=10, help='Checkpoint frequency')

    args = parser.parse_args()

    # Load config if provided
    if config_file or args.config:
        config_path = config_file or args.config
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"Loaded config from {config_path}")
    else:
        config = {}

    # Override config with command line args
    n_epochs = config.get('n_epochs', args.epochs)
    batch_size = config.get('batch_size', args.batch_size)
    n_inner = config.get('n_inner', args.n_inner)
    n_outer = config.get('n_outer', args.n_outer)
    lr_v = config.get('lr_v', args.lr_v)
    lr_flow = config.get('lr_flow', args.lr_flow)
    checkpoint_freq = config.get('checkpoint_freq', args.checkpoint_freq)

    # Setup save directory
    if args.save_dir:
        save_dir = args.save_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"results/cifar10_nonlinear_{timestamp}"

    os.makedirs(save_dir, exist_ok=True)
    print(f"Results will be saved to: {save_dir}")

    # Save config
    config_save = {
        'n_epochs': n_epochs,
        'batch_size': batch_size,
        'n_inner': n_inner,
        'n_outer': n_outer,
        'lr_v': lr_v,
        'lr_flow': lr_flow,
        'checkpoint_freq': checkpoint_freq,
        'timestamp': datetime.now().isoformat()
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config_save, f, indent=2)

    # Setup data
    base, target, ndim = setup_data()

    # Setup models
    interpolant, v = setup_models(config, ndim)

    # Setup optimizers
    opt_v = torch.optim.AdamW(v.parameters(), lr=lr_v, betas=(0.5, 0.9), weight_decay=1e-4)
    opt_flow = torch.optim.AdamW(interpolant.flow_model.parameters(),
                                  lr=lr_flow, betas=(0.5, 0.9), weight_decay=1e-4)

    # Setup loss
    loss_fn_v = stochastic_interpolant.make_loss(
        method='shared',
        interpolant=interpolant,
        loss_type='one-sided-v'
    )

    # Training metrics
    metrics = {
        'v_losses': [],
        'flow_losses': [],
        'epochs': [],
        'wall_time': []
    }

    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    print(f"Epochs: {n_epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Inner steps: {n_inner}")
    print(f"Outer steps: {n_outer}")
    print(f"Learning rates: v={lr_v}, flow={lr_flow}")
    print("="*50 + "\n")

    start_time = time.time()

    # Training loop
    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Train step
        loss_v, loss_flow = train_step(
            bs=batch_size,
            interpolant=interpolant,
            v=v,
            opt_v=opt_v,
            opt_flow=opt_flow,
            loss_fn_v=loss_fn_v,
            base=base,
            target=target,
            n_inner=n_inner,
            n_outer=n_outer,
            device=device
        )

        # Log metrics
        metrics['v_losses'].append(loss_v)
        metrics['flow_losses'].append(loss_flow)
        metrics['epochs'].append(epoch + 1)
        metrics['wall_time'].append(time.time() - start_time)

        epoch_time = time.time() - epoch_start

        # Print progress
        print(f"Epoch {epoch+1}/{n_epochs} | "
              f"v_loss: {loss_v:.4f} | "
              f"flow_loss: {loss_flow:.4f} | "
              f"time: {epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % checkpoint_freq == 0 or epoch == n_epochs - 1:
            save_checkpoint(epoch + 1, interpolant, v, opt_v, opt_flow, metrics, save_dir)

    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("Training Complete!")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Results saved to: {save_dir}")
    print("="*50)

    # Save final metrics
    with open(os.path.join(save_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)

    return save_dir


if __name__ == "__main__":
    main()