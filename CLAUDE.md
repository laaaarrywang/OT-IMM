# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an implementation of the **Stochastic Interpolants** framework for building normalizing flows and diffusion models. The codebase provides tools for learning velocity fields and score functions through interpolation between two distributions.

## Common Development Commands

### Running Experiments

**Nonlinear interpolant experiments:**
```bash
# Single experiment with config file
python scripts/run_checker_tuning.py --config configs/baseline.json

# Override specific parameters
python scripts/run_checker_tuning.py --config configs/baseline.json --exp-name custom_run --device cuda
```

**Multivariate interpolant experiments:**
```bash
# Polynomial coefficients
python scripts/run_multivariate_tuning.py \
  --exp-name test_poly \
  --lr 1e-4 \
  --coefficient-type polynomial \
  --exponent-p 1.0 1.0 \
  --exponent-q 1.0 1.0 \
  --num-epochs 500

# Trigonometric coefficients
python scripts/run_multivariate_tuning.py \
  --exp-name test_trig \
  --lr 1e-4 \
  --coefficient-type trigonometric \
  --freq-a 1.0 1.0 \
  --freq-b 1.0 1.0

# Fourier series coefficients
python scripts/run_multivariate_tuning.py \
  --exp-name test_fourier \
  --lr 1e-4 \
  --coefficient-type fourier \
  --fourier-m 5 \
  --fourier-alpha 1.0 \
  --fourier-beta 1.0
```

**Batch submission on SLURM cluster:**
```bash
./scripts/submit_checker_tuning_jobs.sh      # Submit checkerboard experiments
./scripts/submit_multivariate_jobs.sh        # Submit multivariate experiments
```

**Quick notebook testing:**
```bash
jupyter lab notebooks/checker-nonlinear-2D.ipynb      # Nonlinear interpolant
jupyter lab notebooks/checker-multivariateSI-2D.ipynb # Multivariate interpolant
```

### SLURM Cluster Usage

**Job submission scripts** (`scripts/submit_*.sh`):
- Use `sbatch` to submit jobs with GPU allocation
- Common parameters: `--partition=smallgpu`, `--gpus-per-node=1`
- Load conda environment: `module load conda && conda activate OT_IMM`
- Set PYTHONPATH before running scripts

**Monitoring jobs**:
```bash
squeue -u $USER                    # Check job status
tail -f slurm_logs/checker_*.out   # Monitor specific job output
scancel -u $USER                   # Cancel all jobs
scancel <job_id>                   # Cancel specific job
./scripts/watch_and_cancel_job.sh <job_id> <results_dir> <expected_epochs>
```

**Checking SLURM logs**:
- Logs are saved in `slurm_logs/<experiment_type>_<timestamp>/`
- Each job gets its own `.out` file with stdout/stderr
- Monitor training progress: `tail -f slurm_logs/*/exp_name.out`

### Testing

```bash
python interflow/test_nonlinear_interpolant.py  # Test boundary conditions
python interflow/test_differentiability.py      # Check gradients
python test_fourier_boundaries.py               # Test Fourier network boundaries
python test_multivariate_fix.py                 # Test multivariate interpolant
```

## Architecture Overview

### Core Components

**interflow/stochastic_interpolant.py**
- `Interpolant` class: Main class implementing stochastic interpolants x_t = I_t(x_0, x_1) + γ(t)z
- Supports multiple interpolation paths: linear, trigonometric, encoding-decoding, one-sided, nonlinear, multivariate, mirror
- Provides ODE/SDE integration methods for generation
- Key methods: `calc_xt()`, `calc_It()`, `calc_dtIt()`, `calc_path_parallel_tvel()`, `calc_antithetic_xts()`

**interflow/fabrics.py**
- Defines interpolation path functions (α(t), β(t)) and their derivatives
- `make_It()`: Factory function for creating interpolants
- `make_gamma()`: Factory function for noise schedules (brownian, zero, sines, linear)
- Network builders: `make_fc_net()`, `make_mlp_blocks()`
- Nonlinear flow support via RealNVP integration

**interflow/fabrics_extra.py**
- Advanced neural architectures: Fourier networks, ResNets, SIREN
- `make_fourier_net()`: Critical for sharp boundaries in checkerboard experiments
- `make_resnet()`: ResNet blocks with layer normalization and dropout
- Supports spectral normalization and layer normalization

**interflow/unet_velocity.py**
- UNet-based velocity field architectures for 2D problems
- `UNetVelocityField2D`: Full UNet with attention blocks and time conditioning
- `LightweightUNetVelocity2D`: Simplified UNet for low-dimensional problems
- Factory functions: `make_unet_velocity_2d()`, `make_lightweight_unet_velocity_2d()`
- Used primarily in multivariate interpolant experiments

**interflow/realnvp.py**
- RealNVP implementation for nonlinear interpolants
- `TimeIndexedRealNVP`: Time-conditioned normalizing flow
- Factory functions: `create_vector_flow()`, `create_image_flow()`, etc.
- Supports various data types: vector, MNIST, CIFAR-10, ImageNet

### Key Concepts

1. **Interpolants**: Define trajectories between source distribution ρ₀ and target distribution ρ₁
2. **Velocity field v(x,t)**: Learned to match the time derivative of interpolant
3. **Score function s(x,t)**: Related to denoiser η(x,t) = -γ(t)s(x,t)
4. **Warmup mechanism**: Uses auxiliary linear interpolant during early training

### Experiment Structure

### Configuration

**Nonlinear interpolant configs** (JSON files in `configs/`):
- `training`: TrainingConfig parameters (epochs, batch_size, n_inner, n_outer, learning rates)
- `flow_config`: FlowConfig parameters (num_layers, hidden, mlp_blocks, fourier frequencies)
- Network architecture (depth, width, activation)
- Fourier embedding parameters (fourier_min_freq, fourier_max_freq)
- Regularization (spectral_norm, log_scale_clamp, use_soft_clamp)

**Multivariate interpolant configs** (command-line arguments):
- Learning rate (--lr)
- Coefficient type (polynomial, trigonometric, fourier)
- Matrix parameters (exponent-p, exponent-q for polynomial; freq-a, freq-b for trig)
- Scheduler type (cosine, step, cosine-warm-restarts, onecycle)
- Number of epochs (--num-epochs)

### Results Structure

Results are saved in experiment-specific directories:

**Nonlinear experiments**: `results0929/checker_tuning/<exp_name>/`
- `config.json`: Saved configuration
- `metrics.json`: Per-epoch training metrics (losses, grad norms, v_sq_norm, elapsed time)
- `plots/epoch_XXXX.png`: Generated samples + metric curves side-by-side

**Multivariate experiments**: `results_multivariate/<exp_name>/`
- `config.json`: Training and matrix configuration
- `metrics.json`: Per-epoch metrics (loss_v, grad_v, v_sq_norm, elapsed_sec)
- `multivariate_trajectories.png`: Visualization of interpolant paths (generated once)
- `plots/epoch_XXXX.png`: Sample histograms + loss/velocity curves (every 10 epochs)

## Important Implementation Details

### General

- Device handling: Automatically uses CUDA if available (or specify with `--device`)
- Mixed precision: Supports float32/float64 based on configuration
- Gradient accumulation: Inner/outer optimization loops for better stability
- Fourier embeddings: Maps inputs to higher dimensional space for sharp transitions

### Interpolant Path Types

- `linear`: Standard linear interpolation (1-t)x₀ + tx₁
- `trig`: Trigonometric path with controlled curvature
- `one-sided-linear` / `one-sided-trig`: α(t)x₀ + β(t)x₁ where x₀ ~ N(0,1)
- `nonlinear`: Uses learned RealNVP flow T_θ for interpolation: I_t = α(t)T_θ(x₀,t) + β(t)x₁
- `multivariate`: Matrix-coefficient interpolation A(t)x₀ + B(t)x₁
  - `diagonal`: A(t) and B(t) are diagonal matrices
  - `full`: A(t) and B(t) are full matrices (not yet implemented)
- `mirror`: Dataset to itself interpolation

### Critical Design Patterns

1. **Interpolant Wrappers**:
   - `WarmupInterpolant`: Blends complex and simple interpolants during early training
   - `MixedInterpolant`: Supports mode switching between base/aux/mixed interpolants

2. **Network Architecture Selection**:
   - Fourier networks: Best for sharp boundaries (checkerboard, rectangular patterns)
   - ResNets: General-purpose with layer normalization
   - UNets: Best for multivariate interpolants with complex spatial dependencies

3. **Time Broadcasting**:
   - Interpolants expect time as scalar tensor that broadcasts to batch
   - Networks typically take (x, t) where x is [batch, dim] and t is [batch] or [1]

4. **Metric Computation**:
   - E[|v|²] computed via grid sampling across time and space
   - Used to diagnose training stability and convergence

## Data Generation and Target Distributions

Target distributions include:
- **Checkerboard patterns**:
  - `sample_checker()`: Square checkerboard (2x2 cells)
  - `sample_rect()`: Rectangular checkerboard (configurable width/height)
  - `sample_rhombus()`: Rhombus-shaped checkerboard (configurable angle)
- **Gaussian mixtures**: Multiple modes with various separations
- **Image datasets**: MNIST, CIFAR-10 (via `image-nonlinear-cifar10.ipynb`)
- **Mirror interpolation**: Dataset to itself (via `checker-mirror.ipynb`)

Base distributions:
- **Standard Gaussian**: N(0, I) with configurable variance
- **SimpleNormal** (prior.py): Diagonal Gaussian with per-dimension scale
- **Mixture of Gaussians**: For non-Gaussian base experiments

## Key Notebook Workflows

1. **checker-nonlinear-2D.ipynb**:
   - Trains nonlinear RealNVP interpolant on checkerboard
   - Demonstrates inner/outer loop training
   - Visualizes learned trajectories and velocity fields

2. **checker-multivariateSI-2D.ipynb**:
   - Trains multivariate (matrix-coefficient) interpolant
   - Compares polynomial, trigonometric, and Fourier coefficients
   - Visualizes interpolant paths and boundary conditions

3. **checker-with-score.ipynb**:
   - Score matching approach instead of velocity matching
   - Useful for denoising and score-based generation

4. **image-nonlinear-cifar10.ipynb**:
   - Scales interpolants to high-dimensional image data
   - Uses convolutional architectures in RealNVP

5. **checker-mirror.ipynb**:
   - Self-interpolation (dataset to itself)
   - Different dynamics than base → target interpolation

## Training Strategy

### Nonlinear Interpolants (checker tuning)

1. **Inner/Outer Loops**:
   - Inner loop (n_inner): Updates velocity network with flow frozen
   - Outer loop (n_outer): Updates flow network with velocity frozen
   - Adversarial-style training where flow tries to maximize loss, velocity minimizes

2. **Learning Rate Hierarchy**:
   - `lr_v`: Velocity network learning rate (typically 2e-3)
   - `lr_flow`: Flow network learning rate, 10-50x smaller than lr_v (typically 1e-4)
   - Flow parameters learn slower for stability

3. **Warmup Mechanism**:
   - `WarmupInterpolant` wrapper blends nonlinear and auxiliary (linear) interpolants
   - First `warmup_steps` epochs gradually transition to complex interpolant
   - Prevents early training collapse with nonlinear flows

4. **Schedulers**:
   - StepLR with gamma decay at fixed intervals
   - Separate schedulers for velocity and flow networks

### Multivariate Interpolants

1. **Single Optimization Loop**:
   - Only velocity network is trained (no flow parameters)
   - Uses n_inner iterations per epoch

2. **Learning Rate Schedulers**:
   - Cosine annealing: Smooth decay from base_lr to eta_min
   - Step: Periodic decay by gamma factor
   - CosineAnnealingWarmRestarts: Periodic restarts with cosine decay
   - OneCycleLR: Warmup + cosine annealing strategy

3. **Importance Sampling for Time Points**:
   - **When to use**: Polynomial coefficients with exponents > 1
   - **Algorithm**:
     - Compute `alpha = max(max(exponent_p), 1.0)`
     - If alpha ≤ 1: Use uniform sampling t ~ U(0,1)
     - If alpha > 1: Use Beta distribution t ~ Beta(alpha, 1)
   - **Why it works**:
     - Beta(alpha, 1) concentrates samples near t=1
     - When p>1, (1-t^p) changes rapidly near t=1
     - More samples where gradient magnitude is largest
   - **Loss correction**: Reweight loss by `w = exp(-log p(t))` to maintain unbiasedness
   - **Implementation**: `_sample_times_with_importance()` in run_multivariate_tuning.py:233

4. **Matrix Coefficient Types**:
   - **Polynomial**: A(t) = M_A ⊙ (1 - t^p), B(t) = M_B ⊙ t^q
     - Satisfies boundary conditions: A(0) = M_A, A(1) = 0, B(0) = 0, B(1) = M_B
     - Derivatives: dA/dt = -M_A ⊙ p·t^(p-1), dB/dt = M_B ⊙ q·t^(q-1)
     - When p > 1: A(t) changes rapidly near t=1 (needs importance sampling)
     - When q > 1: B(t) changes rapidly near t=1 (needs importance sampling)
   - **Trigonometric**: A(t) = M_A ⊙ cos^(ω_A)(πt/2), B(t) = M_B ⊙ sin^(ω_B)(πt/2)
   - **Fourier series**: Base trigonometric + sum of sinusoidal Fourier components

### Common Loss Functions

- `loss_per_sample_v`: Velocity matching (||v - ∂I/∂t||²)
- `loss_per_sample_one_sided_v`: One-sided velocity matching (no noise term)
- `loss_per_sample_s`: Score matching
- `loss_per_sample_eta`: Denoiser matching
- `loss_per_sample_b`: Drift matching
- `loss_per_sample_mirror`: Mirror interpolation (dataset to itself)