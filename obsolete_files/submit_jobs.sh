#!/bin/bash

# Robust job submission script with configurable batch size and monitoring
# Usage: ./submit_jobs.sh [batch_size]
# Default batch size is 3 (run 3 jobs simultaneously)

echo "==========================================="
echo "Robust Batched Job Submission Script"
echo "==========================================="

# Get batch size from command line or use default
BATCH_SIZE=${1:-3}
echo "Batch size: $BATCH_SIZE jobs at a time"
echo "Using 30-minute time limit with auto-cancellation"

# Create necessary directories
echo "Creating directories..."
mkdir -p configs
mkdir -p results
mkdir -p slurm_logs

# Generate all configuration files
echo "Generating hyperparameter configurations..."
python hyperparameter_configs.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate configurations"
    exit 1
fi

# Count number of configs
CONFIGS=(configs/*.json)
NUM_CONFIGS=${#CONFIGS[@]}
echo "Found $NUM_CONFIGS configurations to run"

# Submit jobs in batches
echo ""
echo "Submitting jobs in batches of $BATCH_SIZE..."
echo "-------------------"

JOB_COUNT=0
BATCH_COUNT=0
PREV_BATCH_IDS=""

for ((i=0; i<$NUM_CONFIGS; i+=$BATCH_SIZE)); do
    BATCH_COUNT=$((BATCH_COUNT + 1))
    CURRENT_BATCH_IDS=""

    echo ""
    echo "Batch $BATCH_COUNT:"

    # Submit up to BATCH_SIZE jobs
    for ((j=i; j<i+$BATCH_SIZE && j<$NUM_CONFIGS; j++)); do
        config=${CONFIGS[$j]}
        CONFIG_NAME=$(basename "$config" .json)
        JOB_COUNT=$((JOB_COUNT + 1))
        echo -n "  [$JOB_COUNT/$NUM_CONFIGS] Submitting $CONFIG_NAME... "

        # Submit job with dependency on previous batch
        if [ -z "$PREV_BATCH_IDS" ]; then
            # First batch - no dependency
            JOB_ID=$(sbatch --job-name="checker_${CONFIG_NAME}" run_with_monitoring.sbatch "$config" | awk '{print $4}')
        else
            # Subsequent batches - wait for ALL previous batch jobs to complete
            DEPENDENCY=""
            for prev_id in $PREV_BATCH_IDS; do
                if [ -z "$DEPENDENCY" ]; then
                    DEPENDENCY="afterany:$prev_id"
                else
                    DEPENDENCY="$DEPENDENCY:$prev_id"
                fi
            done
            JOB_ID=$(sbatch --dependency=$DEPENDENCY --job-name="checker_${CONFIG_NAME}" run_with_monitoring.sbatch "$config" | awk '{print $4}')
        fi

        if [ $? -eq 0 ]; then
            echo "Job ID: $JOB_ID"
            CURRENT_BATCH_IDS="$CURRENT_BATCH_IDS $JOB_ID"
        else
            echo "FAILED"
            exit 1
        fi

        # Small delay to avoid overwhelming the scheduler
        sleep 0.2
    done

    # Update previous batch IDs for next iteration
    PREV_BATCH_IDS="$CURRENT_BATCH_IDS"
done

echo ""
echo "==========================================="
echo "All jobs submitted in batches of $BATCH_SIZE!"
echo "Total batches: $BATCH_COUNT"
echo ""
echo "Features:"
echo "  - 30-minute time limit per job"
echo "  - Auto-cancellation if no progress for 5 minutes"
echo "  - Batched execution to avoid queue overload"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs in:"
echo "  slurm_logs/"
echo ""
echo "Results will be saved in:"
echo "  results/<config_name>/"
echo "==========================================="