# How to Launch Hyperparameter Tuning Experiments

## Quick Start

```bash
cd /home/wang6559/Desktop/stochastic-interpolants
./submit_tuning_batch.sh
```

## What Gets Run

**10 configs total, all with fixed n_inner=500, n_outer=10**

Focus: Network structure and learning rate exploration

### Configs Overview

1. baseline_500vs10: lr_flow=2e-4, 2L/128H (spectral norm)
2. flow_lr_2x: lr_flow=4e-4, 2L/128H (spectral norm)
3. flow_lr_5x: lr_flow=1e-3, 2L/128H (spectral norm)
4. flow_lr_match: lr_flow=2e-3, 2L/128H (spectral norm)
5. deeper_4layer: lr_flow=1e-3, 4L/128H (spectral norm)
6. wider_256h: lr_flow=1e-3, 2L/256H (spectral norm)
7. deeper_wider: lr_flow=1e-3, 4L/256H, 3 MLP blocks (spectral norm)
8. no_specnorm: lr_flow=1e-3, 4L/256H, 3 MLP blocks (no spectral norm)
9. relaxed_constraints: lr_flow=1e-3, 4L/256H, freq=50, clamp=3.0 (no spectral norm)
10. most_expressive: lr_flow=1e-3, 6L/256H, 4 MLP blocks, freq=50, clamp=3.0 (no spectral norm)

**Batching**: 3 + 3 + 2 + 2 jobs
**Total time**: ~40-50 minutes

## Monitor

```bash
squeue -u $USER
tail -f slurm_logs/tune_*.out
```

## Analyze Results

```bash
python analyze_tuning_results.py
```

Check:
- `results0929/comparison.png`
- `results0929/summary.csv`
- `results0929/<config>/samples_epoch_*.png`

## Troubleshoot

```bash
# Cancel all
scancel -u $USER

# Check errors
cat slurm_logs/tune_*_<job_id>.err
```
