#!/bin/bash
#SBATCH --job-name=checker_tuning
#SBATCH --account=mathdept
#SBATCH --partition=smallgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=187500M
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: sbatch ${BASH_SOURCE[0]} <config_path>"
  exit 1
fi

CONFIG_PATH=$1
CONFIG_PATH=$(realpath "$CONFIG_PATH")
EXP_NAME=$(basename "${CONFIG_PATH%.*}")
REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")

module load conda
if command -v conda >/dev/null 2>&1; then
  # Initialise conda for non-interactive shells
  eval "$(conda shell.bash hook)"
  conda activate OT_IMM
else
  echo "conda command not found after module load" >&2
  exit 1
fi

cd "$REPO_ROOT"
mkdir -p slurm_logs

python scripts/run_checker_tuning.py \
  --config "$CONFIG_PATH" \
  --exp-name "$EXP_NAME" \
  --results-root results0929/checker_tuning
