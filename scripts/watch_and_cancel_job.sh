#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "Usage: $0 <jobid> <results_dir> <expected_epochs> [poll_seconds]" >&2
  echo "   or: $0 <jobid> <results_root> --manifest <manifest.json> [poll_seconds]" >&2
  exit 1
fi

JOBID=$1
shift

RESULT_ROOT=$(realpath "$1")
shift

USE_MANIFEST=0
POLL_SECONDS=60
EXPECTED_EPOCHS=0
declare -A EXPECTED_MAP

if [ "$#" -ge 2 ] && [ "$1" = "--manifest" ]; then
  USE_MANIFEST=1
  MANIFEST_PATH=$(realpath "$2")
  shift 2
  if [ ! -f "$MANIFEST_PATH" ]; then
    echo "Manifest '$MANIFEST_PATH' not found" >&2
    exit 1
  fi
  readarray -t MANIFEST_LINES < <(python - "$MANIFEST_PATH" <<'PY'
import json, sys
path = sys.argv[1]
with open(path) as fh:
    data = json.load(fh)
if isinstance(data, dict):
    items = data.items()
elif isinstance(data, list):
    items = []
    for entry in data:
        if isinstance(entry, dict):
            items.append((entry.get("exp_name"), entry.get("epochs")))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            items.append((entry[0], entry[1]))
else:
    raise SystemExit("Manifest must be a dict or list")
for name, epochs in items:
    if name is None or epochs is None:
        continue
    print(f"{name} {int(epochs)}")
PY
  )
  if [ ${#MANIFEST_LINES[@]} -eq 0 ]; then
    echo "Manifest '$MANIFEST_PATH' is empty or invalid" >&2
    exit 1
  fi
  for line in "${MANIFEST_LINES[@]}"; do
    exp_name=$(echo "$line" | awk '{print $1}')
    epochs=$(echo "$line" | awk '{print $2}')
    EXPECTED_MAP["$exp_name"]=$epochs
  done
else
  EXPECTED_EPOCHS=$1
  shift
fi

if [ "$#" -ge 1 ]; then
  POLL_SECONDS=$1
  shift
fi

function epoch_count() {
  local metrics_file="$1"
  if [ ! -f "$metrics_file" ]; then
    echo 0
    return
  fi
  python - "$metrics_file" <<'PY'
import json, sys
try:
    path = sys.argv[1]
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, list):
        print(len(data))
    elif isinstance(data, dict) and "metrics" in data:
        print(len(data["metrics"]))
    else:
        print(0)
except Exception:
    print(0)
PY
}

while true; do
  if ! squeue -h -j "$JOBID" >/dev/null 2>&1; then
    echo "Job $JOBID no longer in queue; exiting monitor." >&2
    exit 0
  fi

  if [ "$USE_MANIFEST" -eq 1 ]; then
    all_done=1
    for exp in "${!EXPECTED_MAP[@]}"; do
      metrics_path="$RESULT_ROOT/$exp/metrics.json"
      done_epochs=$(epoch_count "$metrics_path")
      expected_epochs=${EXPECTED_MAP[$exp]}
      echo "[$(date '+%H:%M:%S')] $done_epochs / $expected_epochs epochs for $exp (job $JOBID)"
      if [ "$done_epochs" -lt "$expected_epochs" ]; then
        all_done=0
      fi
    done
    if [ "$all_done" -eq 1 ]; then
      echo "All configurations reached expected epochs; cancelling job $JOBID"
      scancel "$JOBID"
      exit 0
    fi
  else
    metrics_path="$RESULT_ROOT/metrics.json"
    done_epochs=$(epoch_count "$metrics_path")
    echo "[$(date '+%H:%M:%S')] $done_epochs / $EXPECTED_EPOCHS epochs detected for job $JOBID"
    if [ "$done_epochs" -ge "$EXPECTED_EPOCHS" ]; then
      echo "Detected completion artifacts; cancelling job $JOBID"
      scancel "$JOBID"
      exit 0
    fi
  fi

  sleep "$POLL_SECONDS"

done
