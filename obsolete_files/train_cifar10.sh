#!/bin/bash

# ==========================================
# CIFAR-10 Nonlinear Stochastic Interpolant Training Script
# ==========================================
# Usage:
#   ./train_cifar10.sh                    # Run with default settings
#   ./train_cifar10.sh --epochs 200       # Customize epochs
#   ./train_cifar10.sh --slurm            # Submit to SLURM
#   ./train_cifar10.sh --config my.json   # Use config file

echo "==========================================="
echo "CIFAR-10 Nonlinear SI Training"
echo "==========================================="

# Default parameters
EPOCHS=100
BATCH_SIZE=100
N_INNER=200
N_OUTER=5
LR_V=0.002
LR_FLOW=0.0002
CHECKPOINT_FREQ=10
USE_SLURM=false
CONFIG_FILE=""
SAVE_DIR=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --n_inner)
            N_INNER="$2"
            shift 2
            ;;
        --n_outer)
            N_OUTER="$2"
            shift 2
            ;;
        --lr_v)
            LR_V="$2"
            shift 2
            ;;
        --lr_flow)
            LR_FLOW="$2"
            shift 2
            ;;
        --checkpoint_freq)
            CHECKPOINT_FREQ="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --save_dir)
            SAVE_DIR="$2"
            shift 2
            ;;
        --slurm)
            USE_SLURM=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --epochs N           Number of training epochs (default: 100)"
            echo "  --batch_size N       Batch size (default: 100)"
            echo "  --n_inner N          Inner optimization steps (default: 200)"
            echo "  --n_outer N          Outer optimization steps (default: 5)"
            echo "  --lr_v RATE          Learning rate for velocity (default: 0.002)"
            echo "  --lr_flow RATE       Learning rate for flow (default: 0.0002)"
            echo "  --checkpoint_freq N  Save checkpoint every N epochs (default: 10)"
            echo "  --config FILE        Load settings from JSON config file"
            echo "  --save_dir DIR       Directory to save results"
            echo "  --slurm              Submit job to SLURM cluster"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create timestamp for unique save directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Set save directory if not provided
if [ -z "$SAVE_DIR" ]; then
    SAVE_DIR="results/cifar10_nonlinear_${TIMESTAMP}"
fi

# Create necessary directories
mkdir -p results
mkdir -p slurm_logs
mkdir -p "$SAVE_DIR"

# Build command
CMD="python train_cifar10_nonlinear.py"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --n_inner $N_INNER"
CMD="$CMD --n_outer $N_OUTER"
CMD="$CMD --lr_v $LR_V"
CMD="$CMD --lr_flow $LR_FLOW"
CMD="$CMD --checkpoint_freq $CHECKPOINT_FREQ"
CMD="$CMD --save_dir $SAVE_DIR"

if [ ! -z "$CONFIG_FILE" ]; then
    CMD="$CMD --config $CONFIG_FILE"
fi

# Display configuration
echo ""
echo "Configuration:"
echo "  Epochs:          $EPOCHS"
echo "  Batch size:      $BATCH_SIZE"
echo "  Inner steps:     $N_INNER"
echo "  Outer steps:     $N_OUTER"
echo "  LR velocity:     $LR_V"
echo "  LR flow:         $LR_FLOW"
echo "  Checkpoint freq: $CHECKPOINT_FREQ"
echo "  Save directory:  $SAVE_DIR"

if [ ! -z "$CONFIG_FILE" ]; then
    echo "  Config file:     $CONFIG_FILE"
fi

echo ""

# Run or submit to SLURM
if [ "$USE_SLURM" = true ]; then
    echo "Submitting to SLURM cluster..."

    # Create SLURM script
    SLURM_SCRIPT="slurm_cifar10_${TIMESTAMP}.sbatch"

    cat > "$SLURM_SCRIPT" <<EOF
#!/bin/bash
#SBATCH --job-name=cifar10_si
#SBATCH --output=slurm_logs/cifar10_%j.out
#SBATCH --error=slurm_logs/cifar10_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules (adjust as needed for your cluster)
module purge
module load cuda/11.8
module load python/3.10

# Activate conda environment if needed
# source activate stochastic_interpolants

# Print job info
echo "Job started at: \$(date)"
echo "Running on host: \$(hostname)"
echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Run training
$CMD

# Print completion time
echo "Job completed at: \$(date)"
EOF

    # Submit job
    JOB_ID=$(sbatch "$SLURM_SCRIPT" | awk '{print $4}')

    if [ $? -eq 0 ]; then
        echo "Job submitted successfully!"
        echo "Job ID: $JOB_ID"
        echo ""
        echo "Monitor with:"
        echo "  squeue -j $JOB_ID"
        echo ""
        echo "View output:"
        echo "  tail -f slurm_logs/cifar10_${JOB_ID}.out"
    else
        echo "Failed to submit job"
        exit 1
    fi

    # Clean up
    rm "$SLURM_SCRIPT"

else
    echo "Running locally..."
    echo "-------------------"
    echo ""

    # Run directly
    $CMD

    if [ $? -eq 0 ]; then
        echo ""
        echo "Training completed successfully!"
        echo "Results saved to: $SAVE_DIR"
    else
        echo ""
        echo "Training failed!"
        exit 1
    fi
fi

echo ""
echo "==========================================="