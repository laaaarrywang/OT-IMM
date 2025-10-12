#!/bin/bash
# Submit SLURM jobs for multivariate interpolant hyperparameter sweep

# Hyperparameter ranges
LEARNING_RATES=(1e-5 1e-4 1e-3)

# Exponent configurations (p and q are the same)
# Format: "p1 p2 q1 q2"
EXPONENT_CONFIGS=(
    "1.0 1.0 1.0 1.0"
    "1.0 0.5 1.0 0.5"
    "1.0 0.2 1.0 0.2"
    "0.2 1.0 0.2 1.0"
    "0.5 1.0 0.5 1.0"
)

# Number of epochs
NUM_EPOCHS=500  # Change to 10 for quick test, 500 for full training

# Results root directory
RESULTS_ROOT="results_multivariate_sweep"

# Timestamp for this sweep
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Log directory
LOG_DIR="slurm_logs/multivariate_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Submitting multivariate interpolant hyperparameter sweep"
echo "Timestamp: ${TIMESTAMP}"
echo "Log directory: ${LOG_DIR}"
echo "================================================================"
echo ""

# Counter for jobs
JOB_COUNT=0

# Loop over all hyperparameter combinations
for LR in "${LEARNING_RATES[@]}"; do
    for EXP_CONFIG in "${EXPONENT_CONFIGS[@]}"; do
        # Parse exponent configuration
        read -r P1 P2 Q1 Q2 <<< "$EXP_CONFIG"

        # Create experiment name
        # Format: lr1e-3_p1.0_0.5_q1.0_0.5
        LR_STR=$(echo "$LR" | sed 's/e-/e-/g')
        P_STR="${P1}_${P2}"
        Q_STR="${Q1}_${Q2}"
        EXP_NAME="lr${LR_STR}_p${P_STR}_q${Q_STR}"

        # Log file
        LOG_FILE="${LOG_DIR}/${EXP_NAME}.out"

        # Submit job
        JOB_ID=$(sbatch \
            --job-name="multivar_${EXP_NAME}" \
            --account=mathdept \
            --partition=smallgpu \
            --nodes=1 \
            --ntasks-per-node=1 \
            --gpus-per-node=1 \
            --cpus-per-task=64 \
            --mem=187500M \
            --time=08:00:00 \
            --output="${LOG_FILE}" \
            --wrap="module load conda && conda activate OT_IMM && \
                    cd /home/wang6559/Desktop/stochastic-interpolants && \
                    export PYTHONPATH=/home/wang6559/Desktop/stochastic-interpolants:\$PYTHONPATH && \
                    python scripts/run_multivariate_tuning.py \
                        --exp-name ${EXP_NAME} \
                        --lr ${LR} \
                        --exponent-p ${P1} ${P2} \
                        --exponent-q ${Q1} ${Q2} \
                        --num-epochs ${NUM_EPOCHS} \
                        --results-root ${RESULTS_ROOT} \
                        --seed 42" \
            | awk '{print $4}')

        ((JOB_COUNT++))

        echo "[$JOB_COUNT] Submitted job ${JOB_ID}: ${EXP_NAME}"
        echo "    LR: ${LR}, p: [${P1}, ${P2}], q: [${Q1}, ${Q2}]"
        echo "    Log: ${LOG_FILE}"
        echo ""
    done
done

echo "================================================================"
echo "Submitted ${JOB_COUNT} jobs total"
echo "================================================================"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Cancel all jobs with:"
echo "  scancel -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f ${LOG_DIR}/*.out"
echo ""
echo "Results will be saved to: ${RESULTS_ROOT}/"
