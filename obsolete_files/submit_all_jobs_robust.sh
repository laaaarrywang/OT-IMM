#!/bin/bash

# Robust script to submit all hyperparameter sweep jobs with automatic resubmission on failure
# This version monitors jobs and can resubmit failed ones

echo "==========================================="
echo "Robust Hyperparameter Sweep Submission"
echo "==========================================="

# Create necessary directories
echo "Creating directories..."
mkdir -p configs
mkdir -p results
mkdir -p slurm_logs
mkdir -p job_status

# Generate all configuration files
echo "Generating hyperparameter configurations..."
python hyperparameter_configs.py

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate configurations"
    exit 1
fi

# Function to check job status
check_job_status() {
    local job_id=$1
    sacct -j "$job_id" --format=State --noheader | tail -n 1 | tr -d ' '
}

# Submit each configuration
echo ""
echo "Submitting jobs..."
echo "-------------------"

# Store job IDs and config mapping
declare -A JOB_CONFIG_MAP

for config in configs/*.json; do
    CONFIG_NAME=$(basename "$config" .json)
    echo -n "Submitting $CONFIG_NAME... "

    # Submit job with the fallback script
    JOB_ID=$(sbatch --job-name="checker_${CONFIG_NAME}" run_single_job_with_fallback.sbatch "$config" 2>/dev/null | awk '{print $4}')

    if [ $? -eq 0 ] && [ -n "$JOB_ID" ]; then
        echo "Job ID: $JOB_ID"
        JOB_CONFIG_MAP[$JOB_ID]="$config"
        echo "$JOB_ID:$config" >> job_status/submitted_jobs.txt
    else
        echo "FAILED - will retry later"
        echo "$config" >> job_status/failed_to_submit.txt
    fi

    sleep 0.5
done

echo ""
echo "==========================================="
echo "Initial submission complete!"
echo ""
echo "Submitted jobs: ${#JOB_CONFIG_MAP[@]}"
echo ""
echo "Monitor with:"
echo "  watch -n 60 'squeue -u $USER'"
echo ""
echo "Check individual job status with:"
echo "  sacct -j <job_id>"
echo ""
echo "Results in: results/<config_name>/"
echo "Logs in: slurm_logs/"
echo "==========================================="

# Optional: Monitor and resubmit failed jobs
echo ""
read -p "Do you want to monitor jobs and auto-resubmit failures? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Monitoring jobs... Press Ctrl+C to stop"

    while true; do
        sleep 300  # Check every 5 minutes

        echo "Checking job statuses at $(date)..."

        for job_id in "${!JOB_CONFIG_MAP[@]}"; do
            STATUS=$(check_job_status "$job_id")

            if [[ "$STATUS" == "FAILED" ]] || [[ "$STATUS" == "TIMEOUT" ]]; then
                CONFIG="${JOB_CONFIG_MAP[$job_id]}"
                CONFIG_NAME=$(basename "$CONFIG" .json)

                echo "Job $job_id ($CONFIG_NAME) failed with status: $STATUS"
                echo "Resubmitting..."

                NEW_JOB_ID=$(sbatch --job-name="retry_${CONFIG_NAME}" run_single_job_with_fallback.sbatch "$CONFIG" 2>/dev/null | awk '{print $4}')

                if [ $? -eq 0 ] && [ -n "$NEW_JOB_ID" ]; then
                    echo "Resubmitted as job $NEW_JOB_ID"
                    unset JOB_CONFIG_MAP[$job_id]
                    JOB_CONFIG_MAP[$NEW_JOB_ID]="$CONFIG"
                fi
            fi
        done

        # Check if all jobs are done
        RUNNING=$(squeue -u "$USER" --noheader | wc -l)
        if [ "$RUNNING" -eq 0 ]; then
            echo "All jobs completed!"
            break
        fi
    done
fi