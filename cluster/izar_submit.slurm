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
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
IZAR_MODULES="${IZAR_MODULES:-gcc/11.3.0 cuda/12.1.1}"
CONFIG="${CONFIG:-e3_dynamic_graph_direct}"
DEVICE="${DEVICE:-cuda}"
SEARCH_ROOT="${SEARCH_ROOT:-/scratch/izar/$USER/nba_forecasting}"
SUMMARY="${SUMMARY:-}"
CHECKPOINT="${CHECKPOINT:-}"
SUBMISSION_ID="${SLURM_JOB_ID:-manual}"
OUTPUT="${OUTPUT:-$PROJECT_ROOT/runs/submissions/${CONFIG}_${SUBMISSION_ID}_submission.csv}"

mkdir -p "$PROJECT_ROOT/.mplconfig" logs "$(dirname "$OUTPUT")"

module purge
module load $IZAR_MODULES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
source .venv/bin/activate

python -m nba_forecasting doctor --expect-cuda

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
