#!/bin/bash -l
#SBATCH --job-name=nba-tune
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
RUN_ROOT="${RUN_ROOT:-/scratch/izar/$USER/nba_forecasting/$SLURM_JOB_ID}"
CONFIGS="${CONFIGS:-${CONFIG:-dynamic_graph_direct}}"
SEARCH_PROFILE="${SEARCH_PROFILE:-basic}"
NUM_SAMPLES="${NUM_SAMPLES:-16}"
MAX_CONCURRENT_TRIALS="${MAX_CONCURRENT_TRIALS:-2}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-19}"
GPUS_PER_TRIAL="${GPUS_PER_TRIAL:-1}"
GRACE_PERIOD="${GRACE_PERIOD:-1}"
REDUCTION_FACTOR="${REDUCTION_FACTOR:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-40}}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-2}"
TRACE_TRAINABLE="${TRACE_TRAINABLE:-0}"
ESTIMATE_TARGET_SAMPLES="${ESTIMATE_TARGET_SAMPLES:-}"
ESTIMATE_TARGET_EPOCHS="${ESTIMATE_TARGET_EPOCHS:-}"
EPOCHS="${EPOCHS:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
EXTRA_SEARCH="${EXTRA_SEARCH:-}"
EXTRA_SET="${EXTRA_SET:-}"

source "$PROJECT_ROOT/cluster/izar_env.sh"
prepare_izar_dirs
load_izar_modules
activate_izar_env
configure_izar_runtime
configure_izar_ray
check_izar_python

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
if [[ -n "$EXTRA_SET" ]]; then
  for kv in $EXTRA_SET; do
    EXTRA_ARGS+=(--set "$kv")
  done
fi
if [[ -n "$EXTRA_SEARCH" ]]; then
  for spec in $EXTRA_SEARCH; do
    EXTRA_ARGS+=(--search "$spec")
  done
fi

read -r -a CONFIG_ARGS <<< "$CONFIGS"

for config in "${CONFIG_ARGS[@]}"; do
  echo "=== Ray Tune start: ${config} ==="
  python -m nba_forecasting tune \
    --config "$config" \
    --device "$DEVICE" \
    --output-root "$RUN_ROOT" \
    "${EXTRA_ARGS[@]}" \
    --search-profile "$SEARCH_PROFILE" \
    --num-samples "$NUM_SAMPLES" \
    --search-alg optuna \
    --scheduler asha \
    --grace-period "$GRACE_PERIOD" \
    --reduction-factor "$REDUCTION_FACTOR" \
    --cpus-per-trial "$CPUS_PER_TRIAL" \
    --gpus-per-trial "$GPUS_PER_TRIAL" \
    --ray-num-cpus "$RAY_NUM_CPUS" \
    --ray-num-gpus "$RAY_NUM_GPUS" \
    --ray-temp-dir "$TMPDIR/ray" \
    --tune-storage-path "$TMPDIR/ray_results" \
    --max-concurrent-trials "$MAX_CONCURRENT_TRIALS"
  echo "=== Ray Tune complete: ${config} ==="
  if [[ -n "$ESTIMATE_TARGET_SAMPLES" && -n "$ESTIMATE_TARGET_EPOCHS" ]]; then
    python -m nba_forecasting estimate-runtime \
      --root "$RUN_ROOT" \
      --target-samples "$ESTIMATE_TARGET_SAMPLES" \
      --target-epochs "$ESTIMATE_TARGET_EPOCHS"
  fi
done

if [[ ${#CONFIG_ARGS[@]} -gt 1 ]]; then
  python -m nba_forecasting summarize-results \
    --root "$RUN_ROOT" \
    --format markdown \
    --output "$RUN_ROOT/result_summary.md"
  cat "$RUN_ROOT/result_summary.md"

  python -m nba_forecasting summarize-timing \
    --root "$RUN_ROOT" \
    --format markdown \
    --output "$RUN_ROOT/timing_summary.md"
  cat "$RUN_ROOT/timing_summary.md"
fi
