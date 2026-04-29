#!/bin/bash -l
#SBATCH --job-name=nba-submit
#SBATCH --partition=gpu
#SBATCH --qos=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=ee-452

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
CONFIG="${CONFIG:-e3_dynamic_graph_direct}"
SEARCH_ROOT="${SEARCH_ROOT:-/scratch/izar/$USER/nba_forecasting}"
SUMMARY="${SUMMARY:-}"
CHECKPOINT="${CHECKPOINT:-}"
SUBMISSION_ID="${SLURM_JOB_ID:-manual}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/runs/submissions/${CONFIG}_${SUBMISSION_ID}_submission.csv}"
source "$PROJECT_ROOT/cluster/izar_env.sh"

mkdir -p "$PROJECT_ROOT/.mplconfig" logs "$(dirname "$OUTPUT")"

load_izar_modules
activate_izar_env
configure_izar_threads
check_izar_python

cd "$PROJECT_ROOT"

if [[ -z "$CHECKPOINT" ]]; then
  if [[ -z "$SUMMARY" ]]; then
    if [[ ! -d "$SEARCH_ROOT" ]]; then
      echo "Search root does not exist: $SEARCH_ROOT" >&2
      echo "Set CHECKPOINT=/path/to/best.pt or SUMMARY=/path/to/ray_tune_summary.json." >&2
      exit 1
    fi
    SUMMARY="$(
      find "$SEARCH_ROOT" \
        -path "*/ray_tune/$CONFIG/*/ray_tune_summary.json" \
        -print | sort | tail -1
    )"
  fi

  if [[ -z "$SUMMARY" ]]; then
    echo "No Ray Tune summary found for CONFIG=$CONFIG under $SEARCH_ROOT." >&2
    echo "Set CHECKPOINT=/path/to/best.pt or SUMMARY=/path/to/ray_tune_summary.json." >&2
    exit 1
  fi

  CHECKPOINT="$(
    python - "$SUMMARY" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
run_dir = Path(summary["best_trial"]["metrics"]["run_dir"])
print(run_dir / "best.pt")
PY
  )"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

echo "Config:     $CONFIG"
echo "Summary:    ${SUMMARY:-not used}"
echo "Checkpoint: $CHECKPOINT"
echo "Output:     $OUTPUT"

python -m nba_forecasting submit \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --device "$DEVICE"

ls -lh "$OUTPUT"
