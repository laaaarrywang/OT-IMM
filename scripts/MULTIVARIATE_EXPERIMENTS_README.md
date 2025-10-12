# Multivariate Interpolant Hyperparameter Sweep

This directory contains scripts for running large-scale hyperparameter sweeps for multivariate interpolant experiments on the SLURM cluster.

## Overview

The multivariate interpolant uses matrix coefficients `A(t)` and `B(t)` as hyperparameters:
```
x_t = A(t) * x_0 + B(t) * x_1
```

Where:
- `A(t) = diag((1-t)^p)` with `p = [p1, p2]`
- `B(t) = diag(t^q)` with `q = [q1, q2]`

## Files

1. **`run_multivariate_tuning.py`**: Standalone training script
   - Trains velocity network for multivariate interpolant
   - Matrix coefficients are fixed hyperparameters (not trained)
   - Saves interpolant trajectories once at the beginning
   - Saves training plots every 10 epochs

2. **`submit_multivariate_jobs.sh`**: SLURM batch submission script
   - Loops over all hyperparameter combinations
   - Submits jobs to the `smallgpu` partition
   - Organizes logs and results automatically

## Hyperparameter Grid

### Learning Rates
- `1e-5`, `1e-4`, `1e-3`

### Exponent Configurations (p = q)
1. `[1.0, 1.0]` - Baseline (linear interpolation for both dimensions)
2. `[1.0, 0.5]` - Fast Y-dimension (good for wide rectangular targets)
3. `[1.0, 0.2]` - Very fast Y-dimension
4. `[0.2, 1.0]` - Fast X-dimension
5. `[0.5, 1.0]` - Moderately fast X-dimension

**Total: 3 × 5 = 15 experiments**

## Usage

### Test Single Job First (Recommended)

Before submitting all jobs, test with a single short job:

```bash
cd /home/wang6559/Desktop/stochastic-interpolants
mkdir -p slurm_logs
sbatch scripts/test_single_job.sh
```

This will run a 5-epoch test to verify:
- Conda environment loads correctly
- All imports work (interflow, matplotlib, torch)
- PYTHONPATH is set correctly
- Training script runs without errors

Check the log: `slurm_logs/test_multivar_*.out`

If the test job succeeds, you'll see "Script finished with exit code: 0" at the end of the log.

### Submit All Jobs

Once the test job succeeds:

```bash
cd /home/wang6559/Desktop/stochastic-interpolants
./scripts/submit_multivariate_jobs.sh
```

This will:
- Submit 15 SLURM jobs (one for each hyperparameter combination)
- Create timestamped log directory: `slurm_logs/multivariate_YYYYMMDD_HHMMSS/`
- Save results to: `results_multivariate_sweep/`

### Run Single Experiment Manually

```bash
python scripts/run_multivariate_tuning.py \
    --exp-name test_experiment \
    --lr 1e-3 \
    --exponent-p 1.0 0.5 \
    --exponent-q 1.0 0.5 \
    --num-epochs 100 \
    --results-root results_multivariate \
    --seed 42
```

### Monitor Jobs

```bash
# Check job status
squeue -u $USER

# Watch specific log
tail -f slurm_logs/multivariate_*/lr1e-3_p1.0_0.5_q1.0_0.5.out

# Cancel all jobs
scancel -u $USER
```

## Output Structure

```
results_multivariate_sweep/
├── lr1e-5_p1.0_1.0_q1.0_1.0/
│   ├── config.json                       # Experiment configuration
│   ├── metrics.json                      # Training metrics per epoch
│   ├── multivariate_trajectories.png     # Interpolant paths (saved once)
│   └── plots/
│       ├── epoch_0010.png               # Training progress plots
│       ├── epoch_0020.png
│       └── ...
├── lr1e-5_p1.0_0.5_q1.0_0.5/
│   └── ...
└── ...
```

### Key Files

- **`config.json`**: Full experiment configuration (learning rate, exponents, etc.)
- **`metrics.json`**: Per-epoch metrics (loss, gradient norms, E[|v|²])
- **`multivariate_trajectories.png`**: Fixed trajectory visualization (saved once)
- **`plots/epoch_XXXX.png`**: Training progress (samples + loss curves)

## Configuration Details

### Training Settings
- **Epochs**: 100 (default)
- **Batch size**: 1000
- **Inner iterations**: 1000 (velocity network updates per epoch)
- **Optimizer**: AdamW with betas=(0.5, 0.9), weight_decay=1e-4
- **Scheduler**: StepLR with gamma=0.8 every 20 inner steps

### Target Distribution
- **Type**: Rectangular checkerboard
- **Parameters**: width=10.0, height=0.1, layers_x=2, layers_y=4
- **Aspect ratio**: 100:1 (very wide rectangles)

### Base Distribution
- **Type**: SimpleNormal (Gaussian)
- **Variance**: 3.0

### Network Architecture
- **Type**: ResNet
- **Hidden sizes**: [256, 256, 256, 256]
- **Activation**: GELU
- **Layer normalization**: True

## Expected Results

### Key Metrics to Compare

1. **Final E[|v|²]**: Lower is better (indicates learned velocity field quality)
2. **Training loss convergence**: Faster convergence indicates better hyperparameters
3. **Sample quality**: Visual inspection of generated checkerboard patterns
4. **Stability**: Consistent gradient norms throughout training

### Hypotheses

- **[1.0, 0.5]** (fast Y): Should work well for 100:1 aspect ratio targets
  - Narrow dimension reaches target quickly
  - Wide dimension takes its time

- **[0.5, 1.0]** (fast X): May struggle
  - Large movements in X happen too quickly
  - Potential instability early in training

- **[1.0, 1.0]** (baseline): Solid baseline but may be suboptimal
  - Equal treatment of both dimensions
  - May not exploit aspect ratio

## Resource Requirements

- **Time**: ~4-6 hours per job (100 epochs)
- **GPU**: 1 × L40S (or similar)
- **Memory**: 187.5 GB
- **CPUs**: 64

## Analysis

After jobs complete, analyze results with:

```python
import json
from pathlib import Path

results_root = Path("results_multivariate_sweep")

# Collect metrics from all experiments
for exp_dir in sorted(results_root.iterdir()):
    if exp_dir.is_dir():
        metrics_file = exp_dir / "metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            final_v_sq = metrics[-1]["v_sq_norm"]
            final_loss = metrics[-1]["loss_v"]
            print(f"{exp_dir.name}: v²={final_v_sq:.4f}, loss={final_loss:.4f}")
```

## Notes

- Matrix coefficients are **hyperparameters**, not trained
- No outer optimization loop (n_outer is effectively 0)
- Interpolant trajectories saved once to reduce I/O
- Training plots saved every 10 epochs to balance disk usage and monitoring

## Troubleshooting

### Common Issues

**1. "ModuleNotFoundError: No module named 'interflow'"**

**Cause**: PYTHONPATH not set correctly

**Fix**: The submission script now includes `export PYTHONPATH=/home/wang6559/Desktop/stochastic-interpolants:$PYTHONPATH`. If you still see this error, verify:
- You're using the updated `submit_multivariate_jobs.sh`
- The path `/home/wang6559/Desktop/stochastic-interpolants` exists
- You're running from the repo root directory

**2. "Non-zero exit code" in SLURM logs**

**Cause**: Various runtime errors

**Fix**:
- Read the actual error in the log file: `cat slurm_logs/multivariate_*/lr*.out`
- Run the test job first: `sbatch scripts/test_single_job.sh`
- Check if conda environment OT_IMM is activated correctly

**3. Jobs queue but never start**

**Cause**: Resource availability or queue limits

**Fix**:
- Check queue: `squeue -u $USER`
- Check partition status: `sinfo -p smallgpu`
- Verify account access: `sacctmgr show assoc user=$USER`

### Debugging Tips

1. **Run test job**: Always start with `sbatch scripts/test_single_job.sh`
2. **Check logs immediately**: `tail -f slurm_logs/test_multivar_*.out`
3. **Test locally** (if on login node with GPU):
   ```bash
   module load conda
   conda activate OT_IMM
   export PYTHONPATH=/home/wang6559/Desktop/stochastic-interpolants:$PYTHONPATH
   python scripts/run_multivariate_tuning.py \
       --exp-name debug \
       --lr 1e-3 \
       --exponent-p 1.0 0.5 \
       --exponent-q 1.0 0.5 \
       --num-epochs 2 \
       --results-root results_test
   ```

## Future Work

When matrix coefficients become trainable:
1. Update `run_multivariate_tuning.py` to include outer optimization loop
2. Add `lr_flow` parameter for flow learning rate
3. Modify to train M_A and M_B matrices
