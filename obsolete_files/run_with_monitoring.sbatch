#!/bin/bash

#SBATCH --job-name=checker_exp
#SBATCH --account=mathdept
#SBATCH --partition=smallgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=187500M
#SBATCH --time=0:30:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --signal=B:USR1@60

# Script to run a single hyperparameter configuration with auto-cancellation
# This script monitors the Python process and cancels the job if no progress is made
# Usage: sbatch run_with_monitoring.sbatch <config_file>

# Load necessary modules (adjust based on your cluster)
module load conda
conda activate OT_IMM

# Create slurm logs directory if it doesn't exist
mkdir -p slurm_logs

# Get the config file from command line argument
CONFIG_FILE=$1

if [ -z "$CONFIG_FILE" ]; then
    echo "Error: No config file provided"
    echo "Usage: sbatch run_with_monitoring.sbatch <config_file>"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file $CONFIG_FILE does not exist"
    exit 1
fi

echo "==========================================="
echo "Starting job on $(hostname) at $(date)"
echo "Config file: $CONFIG_FILE"
echo "Job ID: $SLURM_JOB_ID"
echo "Time limit: 30 minutes"
echo "Auto-cancellation enabled"
echo "==========================================="

# Print GPU information
nvidia-smi

# Set up environment (adjust paths as needed)
export PYTHONPATH=/home/wang6559/Desktop/stochastic-interpolants:$PYTHONPATH

# Function to handle timeout
handle_timeout() {
    echo "Warning: Job is approaching time limit, sending signal to Python process..."
    kill -TERM $PYTHON_PID 2>/dev/null
    wait $PYTHON_PID 2>/dev/null
    echo "Job terminated due to approaching time limit"
    exit 0
}

# Set up signal handler for SLURM timeout warning
trap 'handle_timeout' USR1

# Create a monitoring script that will run in the background
cat > /tmp/monitor_${SLURM_JOB_ID}.sh << 'EOF'
#!/bin/bash
PYTHON_PID=$1
CONFIG_FILE=$2
LAST_SIZE=0
NO_PROGRESS_COUNT=0
MAX_NO_PROGRESS=5  # 5 minutes of no progress

# Extract config name for result file monitoring
CONFIG_NAME=$(basename "$CONFIG_FILE" .json)
RESULT_DIR="results/${CONFIG_NAME}"

while kill -0 $PYTHON_PID 2>/dev/null; do
    sleep 60  # Check every minute

    # Check if result files are being updated
    if [ -d "$RESULT_DIR" ]; then
        CURRENT_SIZE=$(du -sb "$RESULT_DIR" 2>/dev/null | cut -f1)
        if [ -z "$CURRENT_SIZE" ]; then
            CURRENT_SIZE=0
        fi

        if [ "$CURRENT_SIZE" -eq "$LAST_SIZE" ]; then
            NO_PROGRESS_COUNT=$((NO_PROGRESS_COUNT + 1))
            echo "No progress detected for $NO_PROGRESS_COUNT minute(s)"

            if [ $NO_PROGRESS_COUNT -ge $MAX_NO_PROGRESS ]; then
                echo "No progress for $MAX_NO_PROGRESS minutes, terminating job..."
                kill -TERM $PYTHON_PID
                sleep 5
                kill -KILL $PYTHON_PID 2>/dev/null
                exit 1
            fi
        else
            NO_PROGRESS_COUNT=0
            echo "Progress detected, resetting counter"
        fi

        LAST_SIZE=$CURRENT_SIZE
    fi
done
EOF

chmod +x /tmp/monitor_${SLURM_JOB_ID}.sh

# Run the experiment in background to get its PID
echo "Running experiment with config: $CONFIG_FILE"
python run_checker_experiment.py --config "$CONFIG_FILE" &
PYTHON_PID=$!

# Start the monitoring script
/tmp/monitor_${SLURM_JOB_ID}.sh $PYTHON_PID "$CONFIG_FILE" &
MONITOR_PID=$!

# Wait for the Python process to finish
wait $PYTHON_PID
PYTHON_EXIT_CODE=$?

# Kill the monitor if it's still running
kill $MONITOR_PID 2>/dev/null

# Clean up
rm -f /tmp/monitor_${SLURM_JOB_ID}.sh

echo "==========================================="
echo "Job finished at $(date)"
echo "Exit code: $PYTHON_EXIT_CODE"
echo "==========================================="

exit $PYTHON_EXIT_CODE