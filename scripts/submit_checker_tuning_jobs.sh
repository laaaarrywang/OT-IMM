#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")
CONFIG_DIR=${1:-"$REPO_ROOT/configs/checker_tuning"}
SBATCH_SCRIPT="$REPO_ROOT/scripts/checker_tuning_template.sbatch"

if [ ! -d "$CONFIG_DIR" ]; then
  echo "Config directory '$CONFIG_DIR' not found" >&2
  exit 1
fi

for cfg in "$CONFIG_DIR"/*.json; do
  [ -e "$cfg" ] || continue
  base=$(basename "$cfg")
  case "$base" in
    manifest* )
      echo "Skipping manifest file $base"
      continue
      ;;
  esac
  name=$(basename "${cfg%.*}")
  echo "Submitting $name"
  sbatch --job-name="$name" "$SBATCH_SCRIPT" "$cfg"
  sleep 0.5
done
