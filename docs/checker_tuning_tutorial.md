# Checker Tuning Workflow

This guide explains how to run the cluster sweep that tunes the nonlinear interpolant parameters using the `checker-nonlinear-2D` workflow.

## 1. Environment
1. Connect to the cluster head node.
2. Load the shared conda module and activate the project environment:
   ```bash
   module load conda
   conda activate OT_IMM
   ```
3. Move to the repository root (update the path if you cloned elsewhere):
   ```bash
   cd /home/wang6559/Desktop/stochastic-interpolants
   ```

## 2. Configure the Sweep
- Editable JSON configs live in `configs/checker_tuning/`. Each file sets the flow depth (`num_layers`, `hidden`, `mlp_blocks`), learning rates, and inner/outer loop counts.
- Results for every run are written to `results0929/checker_tuning/<exp_name>/` with per-epoch plots in the `plots/` subfolder.
- The training harness is `scripts/run_checker_tuning.py`; it reproduces the notebook logic (losses, gradient norms, E[|v|²], sample plots) headlessly.

## 3. Submit Jobs
Submit all provided configs in one shot:
```bash
./scripts/submit_checker_tuning_jobs.sh
```
The helper script expands every training config (`*.json` excluding `manifest*`) in `configs/checker_tuning/`, normalises the path, and calls `sbatch` with the bundled template. If you only want one job, run:
```bash
sbatch --job-name=checker_baseline \
  scripts/checker_tuning_template.sbatch \
  configs/checker_tuning/checker_baseline.json
```
The sbatch template already includes the requested resources:
```
#SBATCH --job-name=your_job
#SBATCH --account=mathdept
#SBATCH --partition=smallgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=187500M
#SBATCH --time=12:00:00
```
Logs are written to `slurm_logs/<job>_<id>.out|err`.

## 4. Monitor
```bash
squeue -u $USER
watch -n 30 squeue -u $USER
```
Tail a specific log:
```bash
tail -f slurm_logs/checker_baseline_<jobid>.out
```
Each epoch prints: velocity loss, flow loss, gradient norms, E[|v|²], runtime.

### Auto-cancel when outputs finish
To stop a job as soon as every planned configuration has emitted the expected number of epochs, launch the watcher from another shell.

**Single config** (results live directly in one directory):
```bash
./scripts/watch_and_cancel_job.sh <jobid> \
  results0929/checker_tuning/<exp_name> \
  <expected_epochs>
```

**Multiple configs per job** (preferred for sweeps): create a manifest describing each experiment and its epoch count, for example (`configs/checker_tuning/manifest_example.json` ships with defaults):
```json
{
  "checker_baseline": 100,
  "checker_deep_slow": 120
}
```
Save it as `checker_manifest.json`, then run:
```bash
./scripts/watch_and_cancel_job.sh <jobid> \
  results0929/checker_tuning \
  --manifest checker_manifest.json
```
The helper polls every `metrics.json` listed in the manifest and triggers `scancel <jobid>` only after all configurations reach their respective epoch counts.

## 5. Review Outputs
For every experiment `E`:
- `results0929/checker_tuning/E/config.json` – frozen copy of the config.
- `results0929/checker_tuning/E/metrics.json` – per-epoch metrics (losses, grads, E[|v|²]).
- `results0929/checker_tuning/E/plots/epoch_XXXX.png` – side-by-side scatter of generated samples and metric curves.

To compare runs, open the metrics JSON in a notebook or drop the paths into `python analyze_tuning_results.py` once jobs finish.

## 6. Extend the Sweep
1. Copy an existing config, adjust hyperparameters, and save it in `configs/checker_tuning/`.
2. Resubmit with `./scripts/submit_checker_tuning_jobs.sh` or the explicit `sbatch` call above.
3. Keep results organised by giving each new config a unique `exp_name`.

Tip: if you tweak plotting density for quick smoke tests, reduce `v_sq_samples`, `v_sq_time_points`, and `plot_batch_size` locally, but restore the larger values before launching production sweeps.
