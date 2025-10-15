#!/bin/bash
# Submit SLURM jobs for polynomial exponent tuning
#
# This script tests polynomial exponent configurations where:
# - exponent_p = exponent_q (always equal)
# - Components >= 1.0
# - Focus on cases where one component is 1.0
# - Settings follow checker-multivariateSI-2D.ipynb

# Fixed settings from notebook
LEARNING_RATE="1e-3"
SCHEDULER="cosine-warm-restarts"
NUM_EPOCHS=250
BATCH_SIZE=1000
N_INNER=500

# Results root directory
RESULTS_ROOT="results_exponent_tuning"

# Timestamp for this sweep
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Log directory
LOG_DIR="slurm_logs/exponent_tuning_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Polynomial Exponent Tuning: p = q with one component at 1.0"
echo "Fixed learning rate: ${LEARNING_RATE}"
echo "Scheduler: ${SCHEDULER}"
echo "Epochs: ${NUM_EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Inner steps: ${N_INNER}"
echo "Timestamp: ${TIMESTAMP}"
echo "Log directory: ${LOG_DIR}"
echo "================================================================"
echo ""

# Counter for jobs
JOB_COUNT=0

# Polynomial exponent configurations
# Format: "p1 p2 q1 q2 description"
# We keep p = q (same exponents for both A(t) and B(t))
# Focus on cases where one component is 1.0
EXPONENT_CONFIGS=(
    "1.0 1.0 1.0 1.0 baseline"
    "2.0 1.0 2.0 1.0 p2.0_p1.0"
    "1.0 2.0 1.0 2.0 p1.0_p2.0"
    "5.0 1.0 5.0 1.0 p5.0_p1.0"
    "1.0 5.0 1.0 5.0 p1.0_p5.0"
    "1.0 10.0 1.0 10.0 p1.0_p10.0"
    "10.0 1.0 10.0 1.0 p10.0_p1.0"
    "2.0 2.0 2.0 2.0 p2.0_p2.0"
)

for CONFIG in "${EXPONENT_CONFIGS[@]}"; do
    # Parse configuration
    read -r P1 P2 Q1 Q2 DESC <<< "$CONFIG"

    # Create experiment name
    EXP_NAME="poly_${DESC}"

    # Log file
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.out"

    # Submit job
    JOB_ID=$(sbatch \
        --job-name="exp_${EXP_NAME}" \
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
                python scripts/run_exponent_tuning.py \
                    --exp-name ${EXP_NAME} \
                    --lr ${LEARNING_RATE} \
                    --exponent-p ${P1} ${P2} \
                    --exponent-q ${Q1} ${Q2} \
                    --scheduler ${SCHEDULER} \
                    --num-epochs ${NUM_EPOCHS} \
                    --batch-size ${BATCH_SIZE} \
                    --n-inner ${N_INNER} \
                    --results-root ${RESULTS_ROOT} \
                    --seed 42" \
        | awk '{print $4}')

    ((JOB_COUNT++))

    echo "[$JOB_COUNT] Submitted job ${JOB_ID}: ${EXP_NAME}"
    echo "    exponent_p: [${P1}, ${P2}], exponent_q: [${Q1}, ${Q2}]"
    echo "    Log: ${LOG_FILE}"
    echo ""
done

echo "================================================================"
echo "Summary"
echo "================================================================"
echo "Submitted ${JOB_COUNT} jobs total"
echo "- Fixed learning rate: ${LEARNING_RATE}"
echo "- Scheduler: ${SCHEDULER}"
echo "- Epochs per job: ${NUM_EPOCHS}"
echo "- Batch size: ${BATCH_SIZE}"
echo "- Inner steps: ${N_INNER}"
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
echo "================================================================"
