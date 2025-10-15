#!/bin/bash
# Submit SLURM jobs for tuning trigonometric and Fourier series coefficients
#
# This script tests:
# 1. Trigonometric coefficient frequencies for 100:1 anisotropic data
# 2. Fourier series expansion with different M values (fixed α_m = β_m = 1.0)

# Fixed learning rate
LEARNING_RATE="1e-4"

# Use best scheduler from previous experiments
SCHEDULER="cosine-warm-restarts"

# Number of epochs
NUM_EPOCHS=250  # Reduced for faster experiments

# Results root directory
RESULTS_ROOT="results_coefficient_tuning"

# Timestamp for this sweep
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Log directory
LOG_DIR="slurm_logs/coefficient_tuning_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo "Coefficient Tuning Experiments for 100:1 Anisotropic Data"
echo "Fixed learning rate: ${LEARNING_RATE}"
echo "Scheduler: ${SCHEDULER}"
echo "Timestamp: ${TIMESTAMP}"
echo "Log directory: ${LOG_DIR}"
echo "================================================================"
echo ""

# Counter for jobs
JOB_COUNT=0

# ==========================================
# PART 1: Trigonometric Frequency Tuning
# ==========================================

echo "--- Part 1: Trigonometric Frequency Tuning ---"
echo "Testing different frequency ratios for 100:1 anisotropic data"
echo ""

# Trigonometric configurations - systematic frequency ratio testing
# Format: "freq_a1 freq_a2 freq_b1 freq_b2 description"
# x-dim (wide): fixed at freq=2 (standard cos^2, sin^2)
# y-dim (narrow): vary frequency to find optimal anisotropy handling
TRIG_CONFIGS=(
    "2.0 2.0 2.0 2.0 ratio_1to1"       # baseline uniform
    "2.0 4.0 2.0 4.0 ratio_1to2"       # 2x faster in y
    "2.0 10.0 2.0 10.0 ratio_1to5"     # 5x faster in y
    "2.0 20.0 2.0 20.0 ratio_1to10"    # 10x faster in y
    "2.0 40.0 2.0 40.0 ratio_1to20"    # 20x faster in y
    "2.0 60.0 2.0 60.0 ratio_1to30"    # 30x faster in y
    "2.0 80.0 2.0 80.0 ratio_1to40"    # 40x faster in y
    "2.0 100.0 2.0 100.0 ratio_1to50"  # 50x faster (half of data ratio)
    "2.0 200.0 2.0 200.0 ratio_1to100" # 100x faster (matching data ratio)
)

for CONFIG in "${TRIG_CONFIGS[@]}"; do
    # Parse configuration
    read -r FA1 FA2 FB1 FB2 DESC <<< "$CONFIG"

    # Create experiment name
    EXP_NAME="trig_${DESC}"

    # Log file
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.out"

    # Submit job
    JOB_ID=$(sbatch \
        --job-name="coef_${EXP_NAME}" \
        --account=mathdept \
        --partition=smallgpu \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gpus-per-node=1 \
        --cpus-per-task=64 \
        --mem=187500M \
        --time=06:00:00 \
        --output="${LOG_FILE}" \
        --wrap="module load conda && conda activate OT_IMM && \
                cd /home/wang6559/Desktop/stochastic-interpolants && \
                export PYTHONPATH=/home/wang6559/Desktop/stochastic-interpolants:\$PYTHONPATH && \
                python scripts/run_multivariate_tuning.py \
                    --exp-name ${EXP_NAME} \
                    --lr ${LEARNING_RATE} \
                    --coefficient-type trigonometric \
                    --freq-a ${FA1} ${FA2} \
                    --freq-b ${FB1} ${FB2} \
                    --scheduler ${SCHEDULER} \
                    --num-epochs ${NUM_EPOCHS} \
                    --results-root ${RESULTS_ROOT} \
                    --seed 42" \
        | awk '{print $4}')

    ((JOB_COUNT++))

    echo "[$JOB_COUNT] Submitted job ${JOB_ID}: ${EXP_NAME}"
    echo "    freq_A: [${FA1}, ${FA2}], freq_B: [${FB1}, ${FB2}]"
    echo "    Log: ${LOG_FILE}"
    echo ""
done

# ==========================================
# PART 2: Fourier Series M Tuning
# ==========================================

echo ""
echo "--- Part 2: Fourier Series M Tuning ---"
echo "Testing different number of Fourier terms with fixed α_m = β_m = 1.0"
echo ""

# Fourier M values to test
# Testing how many Fourier terms are needed for 100:1 anisotropic data
FOURIER_M_VALUES=(2 3 5 7 10 15 20 30)

for M in "${FOURIER_M_VALUES[@]}"; do
    # Create experiment name
    EXP_NAME="fourier_M${M}"

    # Log file
    LOG_FILE="${LOG_DIR}/${EXP_NAME}.out"

    # Submit job
    JOB_ID=$(sbatch \
        --job-name="coef_${EXP_NAME}" \
        --account=mathdept \
        --partition=smallgpu \
        --nodes=1 \
        --ntasks-per-node=1 \
        --gpus-per-node=1 \
        --cpus-per-task=64 \
        --mem=187500M \
        --time=06:00:00 \
        --output="${LOG_FILE}" \
        --wrap="module load conda && conda activate OT_IMM && \
                cd /home/wang6559/Desktop/stochastic-interpolants && \
                export PYTHONPATH=/home/wang6559/Desktop/stochastic-interpolants:\$PYTHONPATH && \
                python scripts/run_multivariate_tuning.py \
                    --exp-name ${EXP_NAME} \
                    --lr ${LEARNING_RATE} \
                    --coefficient-type fourier \
                    --fourier-m ${M} \
                    --fourier-alpha 1.0 \
                    --fourier-beta 1.0 \
                    --scheduler ${SCHEDULER} \
                    --num-epochs ${NUM_EPOCHS} \
                    --results-root ${RESULTS_ROOT} \
                    --seed 42" \
        | awk '{print $4}')

    ((JOB_COUNT++))

    echo "[$JOB_COUNT] Submitted job ${JOB_ID}: ${EXP_NAME}"
    echo "    Fourier M: ${M}, α_m = β_m = 1.0"
    echo "    Log: ${LOG_FILE}"
    echo ""
done

echo "================================================================"
echo "Summary"
echo "================================================================"
echo "Submitted ${JOB_COUNT} jobs total"
echo "- Part 1: ${#TRIG_CONFIGS[@]} trigonometric frequency configs"
echo "- Part 2: ${#FOURIER_M_VALUES[@]} Fourier M values"
echo "- Fixed learning rate: ${LEARNING_RATE}"
echo "- Scheduler: ${SCHEDULER}"
echo "- Epochs per job: ${NUM_EPOCHS}"
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
echo "View specific scheduler results:"
for SCHEDULER in "${SCHEDULERS[@]}"; do
    echo "  grep 'E\[|v|\^2\]' ${LOG_DIR}/*_${SCHEDULER}.out | tail -5"
done
echo ""
echo "Results will be saved to: ${RESULTS_ROOT}/"
echo ""
echo "Compare results with:"
echo "  python scripts/analyze_scheduler_results.py ${RESULTS_ROOT}"
echo "================================================================"