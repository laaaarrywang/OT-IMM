#!/bin/bash
# Test script to run a single multivariate interpolant job

#SBATCH --job-name=test_multivar
#SBATCH --account=mathdept
#SBATCH --partition=smallgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=187500M
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/test_multivar_%j.out

echo "=========================================="
echo "Testing multivariate interpolant training"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Load environment
module load conda
conda activate OT_IMM

# Change to repo directory
cd /home/wang6559/Desktop/stochastic-interpolants

# Set PYTHONPATH
export PYTHONPATH=/home/wang6559/Desktop/stochastic-interpolants:$PYTHONPATH

echo ""
echo "Python version:"
python --version

echo ""
echo "Testing import:"
python -c "import interflow; print('✓ interflow imported successfully')"
python -c "import matplotlib; print('✓ matplotlib imported successfully')"
python -c "import torch; print('✓ torch imported successfully')"

echo ""
echo "Testing device detection:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

echo ""
echo "Testing multivariate interpolant creation:"
python -c "
import sys
sys.path.insert(0, '/home/wang6559/Desktop/stochastic-interpolants')
import interflow.stochastic_interpolant as si
import torch

# Test creating multivariate interpolant
interp = si.Interpolant(
    path='multivariate',
    gamma_type=None,
    data_type='vector',
    data_dim=2,
    matrix_config={
        'matrix_type': 'diagonal',
        'exponent_p': [1.0, 0.5],
        'exponent_q': [1.0, 0.5],
    }
)
print('✓ Multivariate interpolant created successfully')

# Test calc_xt
x0 = torch.randn(2, 2)
x1 = torch.randn(2, 2)
t = torch.tensor([0.5])
xt = interp.calc_xt(t, x0, x1)
print(f'✓ calc_xt works, shape: {xt.shape}')
"

echo ""
echo "Running training script (2 epochs only for testing):"
python scripts/run_multivariate_tuning.py \
    --exp-name test_run \
    --lr 1e-3 \
    --exponent-p 1.0 0.5 \
    --exponent-q 1.0 0.5 \
    --num-epochs 2 \
    --results-root results_multivariate_test \
    --seed 42

EXIT_CODE=$?
echo ""
echo "=========================================="
echo "Script finished with exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
