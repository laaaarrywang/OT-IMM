# Hyperparameter Sweep Instructions

## Files Created

1. **run_checker_experiment.py** - Main Python script that runs the experiments
2. **hyperparameter_configs.py** - Generates 10 different hyperparameter configurations
3. **run_single_job.sbatch** - SBATCH script to run a single configuration
4. **submit_all_jobs.sh** - Master script to submit all jobs

## How to Run

### Option 1: Submit all jobs at once
```bash
./submit_all_jobs.sh
```
This will:
- Generate all config files in `configs/`
- Submit each config as a separate SBATCH job
- Save results in `results/<config_name>/`
- Save logs in `slurm_logs/`

### Option 2: Submit individual jobs
First generate configs:
```bash
python hyperparameter_configs.py
```

Then submit individual configs:
```bash
sbatch run_single_job.sbatch configs/baseline.json
sbatch run_single_job.sbatch configs/enhanced_rect.json
# etc...
```

## Configurations

10 configurations are generated with varying:
- Network depth (num_layers: 2-6)
- Network width (hidden: 128-512)
- MLP blocks (2-4)
- Fourier frequencies (min: 0.1-1.0, max: 10-100)
- Learning rates (lr_v: 1e-3 to 2e-3, lr_flow: 5e-5 to 2e-4)
- Training dynamics (n_inner: 50-500, n_outer: 1-50)
- Constraints (spectral_norm, log_scale_clamp)

### Config Details:
1. **baseline** - Original settings from notebook
2. **enhanced_rect** - Optimized for rectangular target (recommended)
3. **high_freq** - Very high frequency (50) for sharp transitions
4. **deep_net** - 6 layers with more outer steps
5. **wide_net** - 512 hidden units
6. **balanced_outer** - Focus on outer optimization (20 steps)
7. **conservative** - Low learning rates, high regularization
8. **aggressive** - No constraints, high capacity
9. **very_high_freq** - Extreme frequencies (0.1-100)
10. **outer_focused** - 50 inner/50 outer steps balance

## Output Structure

Each job creates:
```
results/
├── <config_name>/
│   ├── config.json          # Configuration used
│   ├── metrics.json         # Training metrics
│   ├── checkpoint.pt        # Model checkpoint
│   ├── epoch_0001.png       # Plot at epoch 1
│   ├── epoch_0002.png       # Plot at epoch 2
│   └── ...                  # Plot for each epoch
```

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Check specific job output
tail -f slurm_logs/checker_<config_name>_<job_id>.out

# Cancel all jobs if needed
scancel -u $USER
```

## After Jobs Complete

Analyze results with:
```python
import json
import matplotlib.pyplot as plt

# Load metrics from all configs
configs = ['baseline', 'enhanced_rect', 'high_freq', ...]
for config in configs:
    with open(f'results/{config}/metrics.json') as f:
        metrics = json.load(f)
    # Plot v_losses, flow_losses, v_squared_norm, etc.
```

## Notes

- Each job requests 1 GPU, 64 CPUs, 187GB RAM for 12 hours
- Plots are saved at every epoch (50 plots per job)
- The WarmupInterpolant always uses the auxiliary linear interpolant (warmup_factor=0.0)
- Target is rectangular checkerboard (target_rect)