#!/bin/bash
# Test submission with just 3 jobs (1 learning rate, 3 exponent configs)

LEARNING_RATE=1e-3
EXPONENT_CONFIGS=(
    "1.0 1.0 1.0 1.0"
    "1.0 0.5 1.0 0.5"
    "0.5 1.0 0.5 1.0"
)

NUM_EPOCHS=10  # Quick 10-epoch test
RESULTS_ROOT="results_test_parallel"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="slurm_logs/test_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "Submitting 3 test jobs to verify parallel execution..."
JOB_COUNT=0

for EXP_CONFIG in "${EXPONENT_CONFIGS[@]}"; do
    read -r P1 P2 Q1 Q2 <<< "$EXP_CONFIG"
    EXP_NAME="test_p${P1}_${P2}_q${Q1}_${Q2}"
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.out"

    JOB_ID=$(sbatch \
        --job-name="test_${EXP_NAME}" \
        --account=mathdept \
        --partition=smallgpu \
        --nodes=1 \
        --gpus-per-node=1 \
        --cpus-per-task=64 \
        --mem=187500M \
        --time=00:30:00 \
        --output="${LOG_FILE}" \
        --wrap="module load conda && conda activate OT_IMM && \
                cd /home/wang6559/Desktop/stochastic-interpolants && \
                export PYTHONPATH=/home/wang6559/Desktop/stochastic-interpolants:\$PYTHONPATH && \
                date && \
                python scripts/run_multivariate_tuning.py \
                    --exp-name ${EXP_NAME} \
                    --lr ${LEARNING_RATE} \
                    --exponent-p ${P1} ${P2} \
                    --exponent-q ${Q1} ${Q2} \
                    --num-epochs ${NUM_EPOCHS} \
                    --results-root ${RESULTS_ROOT} \
                    --seed 42 && \
                date" \
        | awk '{print $4}')

    ((JOB_COUNT++))
    echo "[$JOB_COUNT] Submitted job ${JOB_ID}: ${EXP_NAME}"
done

echo ""
echo "Submitted ${JOB_COUNT} test jobs (should run in parallel)"
echo "Check with: squeue -u \$USER"
echo "They should all show 'R' (running) state if GPUs are available"