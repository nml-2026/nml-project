#!/bin/bash -l
#SBATCH --job-name=nba-tune-1node
#SBATCH --partition=gpu
#SBATCH --qos=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=ee-452
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
IZAR_MODULES="${IZAR_MODULES:-gcc/11.3.0 cuda/12.1.1}"
RUN_ROOT="${RUN_ROOT:-/scratch/izar/$USER/nba_forecasting/$SLURM_JOB_ID}"
CONFIG="${CONFIG:-e3_dynamic_graph_direct}"
DEVICE="${DEVICE:-cuda}"
SEARCH_PROFILE="${SEARCH_PROFILE:-basic}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
MAX_CONCURRENT_TRIALS="${MAX_CONCURRENT_TRIALS:-2}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-19}"
GPUS_PER_TRIAL="${GPUS_PER_TRIAL:-1}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-40}}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-2}"
TRACE_TRAINABLE="${TRACE_TRAINABLE:-0}"
ESTIMATE_TARGET_SAMPLES="${ESTIMATE_TARGET_SAMPLES:-}"
ESTIMATE_TARGET_EPOCHS="${ESTIMATE_TARGET_EPOCHS:-}"
EPOCHS="${EPOCHS:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"

mkdir -p "$RUN_ROOT" "$PROJECT_ROOT/.mplconfig" logs
if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/nba-tune-$SLURM_JOB_ID}"
  mkdir -p "$TMPDIR"
fi
mkdir -p "$TMPDIR/ray" "$TMPDIR/ray_results"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export RAY_DEDUP_LOGS=0
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"
export PYTHONUNBUFFERED=1

module purge
module load $IZAR_MODULES

cd "$PROJECT_ROOT"
source .venv/bin/activate

python -m nba_forecasting doctor --expect-cuda

EXTRA_ARGS=()
if [[ -n "$EPOCHS" ]]; then
  EXTRA_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
  EXTRA_ARGS+=(--set "training.max_train_batches=$MAX_TRAIN_BATCHES")
fi
if [[ -n "$MAX_VAL_BATCHES" ]]; then
  EXTRA_ARGS+=(--set "training.max_val_batches=$MAX_VAL_BATCHES")
fi
if [[ "$TRACE_TRAINABLE" == "1" ]]; then
  EXTRA_ARGS+=(--trace-trainable)
fi

python -m nba_forecasting tune \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --output-root "$RUN_ROOT" \
  "${EXTRA_ARGS[@]}" \
  --search-profile "$SEARCH_PROFILE" \
  --num-samples "$NUM_SAMPLES" \
  --search-alg optuna \
  --scheduler asha \
  --cpus-per-trial "$CPUS_PER_TRIAL" \
  --gpus-per-trial "$GPUS_PER_TRIAL" \
  --ray-num-cpus "$RAY_NUM_CPUS" \
  --ray-num-gpus "$RAY_NUM_GPUS" \
  --ray-temp-dir "$TMPDIR/ray" \
  --tune-storage-path "$TMPDIR/ray_results" \
  --max-concurrent-trials "$MAX_CONCURRENT_TRIALS"

if [[ -n "$ESTIMATE_TARGET_SAMPLES" && -n "$ESTIMATE_TARGET_EPOCHS" ]]; then
  python -m nba_forecasting estimate-runtime \
    --root "$RUN_ROOT" \
    --target-samples "$ESTIMATE_TARGET_SAMPLES" \
    --target-epochs "$ESTIMATE_TARGET_EPOCHS"
fi
