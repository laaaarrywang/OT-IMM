#!/bin/bash

#SBATCH --job-name=checker_exp
#SBATCH --account=mathdept
#SBATCH --partition=smallgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=187500M
#SBATCH --time=7:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Script to run a single hyperparameter configuration with fallback on failure
# Usage: sbatch run_single_job_with_fallback.sbatch <config_file>

mkdir -p slurm_logs

CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file provided"
    exit 1
fi

echo "==========================================="
echo "Starting job on $(hostname) at $(date)"
echo "Config file: $CONFIG_FILE"
echo "Job ID: $SLURM_JOB_ID"
echo "==========================================="

nvidia-smi

export PYTHONPATH=/home/wang6559/Desktop/stochastic-interpolants:$PYTHONPATH

# Try to run the experiment
echo "Running experiment with config: $CONFIG_FILE"
python run_checker_experiment.py --config "$CONFIG_FILE"
EXIT_CODE=$?

# If it failed, try with reduced batch size
if [ $EXIT_CODE -ne 0 ]; then
    echo "==========================================="
    echo "First attempt failed with exit code $EXIT_CODE"
    echo "Retrying with reduced batch size..."
    echo "==========================================="

    # Create a modified config with smaller batch size
    CONFIG_NAME=$(basename "$CONFIG_FILE" .json)
    FALLBACK_CONFIG="/tmp/fallback_${CONFIG_NAME}_${SLURM_JOB_ID}.json"

    python -c "
import json
with open('$CONFIG_FILE', 'r') as f:
    config = json.load(f)
# Reduce memory-intensive parameters
config['batch_size'] = 500  # Half the batch size
config['N_epoch'] = 30  # Fewer epochs
config['plot_freq'] = 5  # Less frequent plotting
with open('$FALLBACK_CONFIG', 'w') as f:
    json.dump(config, f, indent=2)
print('Created fallback config with batch_size=500')
"

    echo "Running with fallback config: $FALLBACK_CONFIG"
    python run_checker_experiment.py --config "$FALLBACK_CONFIG"
    EXIT_CODE=$?

    # Clean up temp file
    rm -f "$FALLBACK_CONFIG"
fi

echo "==========================================="
echo "Job finished at $(date) with exit code $EXIT_CODE"
echo "==========================================="

exit $EXIT_CODE