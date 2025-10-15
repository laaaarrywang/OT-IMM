# Polynomial Exponent Tuning Summary

## Overview
This tuning sweeps polynomial exponent configurations where `exponent_p = exponent_q` (always equal), with components >= 1.0, focusing on cases where one component is 1.0.

## Settings (from checker-multivariateSI-2D.ipynb)
- **Learning rate**: 1e-3
- **Epochs**: 250
- **Batch size**: 1000
- **Inner steps (n_inner)**: 500
- **Base variance**: 3.0
- **Target**: rectangular checkerboard (w=10.0, h=0.1)
- **Scheduler**: CosineAnnealingWarmRestarts (T_0=50*n_inner, T_mult=2)
- **Network**: Lightweight UNet with hidden_dims=[128, 256, 256, 128]
- **Importance sampling**: Beta distribution with alpha=max(exponent_p)

## 8 Exponent Configurations

| # | Config | exponent_p | exponent_q | Description |
|---|--------|------------|------------|-------------|
| 1 | baseline | [1.0, 1.0] | [1.0, 1.0] | Baseline - both components at 1.0 |
| 2 | p1.0_p1.5 | [1.0, 1.5] | [1.0, 1.5] | One at 1.0, other at 1.5 |
| 3 | p1.0_p2.0 | [1.0, 2.0] | [1.0, 2.0] | One at 1.0, other at 2.0 |
| 4 | p1.0_p3.0 | [1.0, 3.0] | [1.0, 3.0] | One at 1.0, other at 3.0 |
| 5 | p1.0_p5.0 | [1.0, 5.0] | [1.0, 5.0] | One at 1.0, other at 5.0 |
| 6 | p1.0_p7.0 | [1.0, 7.0] | [1.0, 7.0] | One at 1.0, other at 7.0 |
| 7 | p1.5_p1.5 | [1.5, 1.5] | [1.5, 1.5] | Both at 1.5 |
| 8 | p2.0_p2.0 | [2.0, 2.0] | [2.0, 2.0] | Both at 2.0 |

## Key Features

### Importance Sampling
- For exponents > 1.0, uses Beta(alpha, 1) distribution for time sampling
- Alpha = max(exponent_p components)
- Reweights loss by exp(-log_pdf) to maintain unbiasedness
- Focuses training on regions where coefficients change rapidly (near t=1)

### Matrix Coefficients
- **A(t) = M_A ⊙ (1 - t^p)**: Source coefficient (decays to 0 at t=1)
- **B(t) = M_B ⊙ t^q**: Target coefficient (grows to 1 at t=1)
- **Derivatives**:
  - dA/dt = -M_A ⊙ p·t^(p-1)
  - dB/dt = M_B ⊙ q·t^(q-1)

### Why Focus on One Component = 1.0?
The rectangular target has extreme anisotropy (100:1 ratio with w=10.0, h=0.1). By keeping one component at 1.0:
- Tests how different exponents handle the narrow dimension
- Isolates the effect of increased exponents on one direction
- Provides baseline comparison with uniform exponents

## Files Created
- `scripts/submit_exponent_tuning.sh`: SLURM job submission script
- `scripts/run_exponent_tuning.py`: Training script with notebook settings
- Results saved to: `results_exponent_tuning/poly_<config>/`

## Usage

### Submit all 8 jobs:
```bash
cd /home/wang6559/Desktop/stochastic-interpolants
./scripts/submit_exponent_tuning.sh
```

### Monitor jobs:
```bash
squeue -u $USER
tail -f slurm_logs/exponent_tuning_*/poly_*.out
```

### Test single job locally:
```bash
export PYTHONPATH=/home/wang6559/Desktop/stochastic-interpolants:$PYTHONPATH
conda activate OT_IMM
python scripts/run_exponent_tuning.py \
    --exp-name test \
    --lr 1e-3 \
    --exponent-p 1.0 2.0 \
    --exponent-q 1.0 2.0 \
    --num-epochs 10 \
    --batch-size 1000 \
    --n-inner 500 \
    --results-root test_results
```

## Expected Runtime
- ~6-8 hours per job (250 epochs, 500 inner steps)
- All 8 jobs submitted simultaneously (within cluster limits)

## Analysis
After completion, compare:
1. Final E[|v|^2] values across configs
2. Loss convergence curves
3. Sample quality from generated plots
4. Gradient norms during training

Look for:
- Optimal exponent for narrow dimension (h=0.1)
- Training stability with high exponents
- Impact of importance sampling on convergence
