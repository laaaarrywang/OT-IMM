# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an implementation of the **Stochastic Interpolants** framework for building normalizing flows and diffusion models. The codebase provides tools for learning velocity fields and score functions through interpolation between two distributions.

## Common Development Commands

### Running Experiments

**Single experiment:**
```bash
python run_checker_experiment.py configs/baseline.json
```

**Batch submission on SLURM cluster:**
```bash
./submit_jobs.sh [batch_size]  # Default batch size is 3
```

**Generate hyperparameter configurations:**
```bash
python hyperparameter_configs.py
```

**Quick notebook testing:**
```bash
python quick_notebook_test.py
```

### Monitoring SLURM Jobs

```bash
squeue -u $USER                    # Check job status
tail -f slurm_logs/checker_*.out   # Monitor specific job output
scancel -u $USER                   # Cancel all jobs
```

## Architecture Overview

### Core Components

**interflow/stochastic_interpolant.py**
- `Interpolant` class: Main class implementing stochastic interpolants x_t = I_t(x_0, x_1) + ≥(t)z
- Supports multiple interpolation paths: linear, trigonometric, encoding-decoding, one-sided, nonlinear
- Provides ODE/SDE integration methods for generation
- Key methods: `calc_xt()`, `calc_It()`, `calc_dtIt()`, `calc_path_parallel_tvel()`

**interflow/fabrics.py**
- Defines interpolation path functions (±(t), ≤(t)) and their derivatives
- `make_It()`: Factory function for creating interpolants
- `make_gamma()`: Factory function for noise schedules (brownian, zero, sines, linear)
- Network builders: `make_fc_net()`, `make_mlp_blocks()`
- Nonlinear flow support via RealNVP integration

**interflow/realnvp.py**
- RealNVP implementation for nonlinear interpolants
- `TimeIndexedRealNVP`: Time-conditioned normalizing flow
- Factory functions: `create_vector_flow()`, `create_image_flow()`, etc.
- Supports various data types: vector, MNIST, CIFAR-10, ImageNet

### Key Concepts

1. **Interpolants**: Define trajectories between source distribution ¡Ä and target distribution ¡Å
2. **Velocity field v(x,t)**: Learned to match the time derivative of interpolant
3. **Score function s(x,t)**: Related to denoiser ∑(x,t) = -≥(t)s(x,t)
4. **Warmup mechanism**: Uses auxiliary linear interpolant during early training

### Experiment Structure

Experiments are configured via JSON files specifying:
- Network architecture (depth, width, activation)
- Training dynamics (n_inner, n_outer iterations)
- Learning rates (lr_v for velocity, lr_flow for nonlinear flow)
- Fourier embedding parameters
- Regularization (spectral_norm, log_scale_clamp)

Results are saved in `results/<config_name>/` with:
- Model checkpoints
- Training metrics (losses, grad norms)
- Visualization plots at each epoch

## Important Implementation Details

- Device handling: Automatically uses CUDA if available
- Mixed precision: Supports float32/float64 based on configuration
- Gradient accumulation: Inner/outer optimization loops for better stability
- Fourier embeddings: Maps inputs to higher dimensional space for sharp transitions
- Path types:
  - `linear`: Standard linear interpolation (1-t)xÄ + txÅ
  - `trig`: Trigonometric path with controlled curvature
  - `one-sided`: ±(t)xÄ + ≤(t)xÅ where xÄ ~ N(0,1)
  - `nonlinear`: Uses learned normalizing flow for interpolation

## Data Generation

Target distributions in notebooks include:
- Checkerboard patterns (square, rectangular, rhombus)
- Gaussian mixtures
- Image datasets (MNIST, CIFAR-10)
- Mirror interpolation (dataset to itself)

Base distributions typically use:
- Standard Gaussian N(0,1)
- Mixture of Gaussians
- Learned prior distributions