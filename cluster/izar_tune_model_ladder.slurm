#!/bin/bash -l
#SBATCH --job-name=nba-tune-ladder
#SBATCH --partition=gpu
#SBATCH --qos=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=180G
#SBATCH --time=11:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=ee-452

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
RUN_ROOT="${RUN_ROOT:-/scratch/izar/$USER/nba_forecasting/$SLURM_JOB_ID}"
CONFIGS="${CONFIGS:-e3_dynamic_graph_direct e1_social_ssm e3_graph_ssm_no_social}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
MAX_CONCURRENT_TRIALS="${MAX_CONCURRENT_TRIALS:-2}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-8}"
GPUS_PER_TRIAL="${GPUS_PER_TRIAL:-1}"
EPOCHS="${EPOCHS:-}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-}"
source "$PROJECT_ROOT/cluster/izar_env.sh"

mkdir -p "$RUN_ROOT" "$PROJECT_ROOT/.mplconfig" logs

load_izar_modules
activate_izar_env
configure_izar_threads
check_izar_python

cd "$PROJECT_ROOT"

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

read -r -a CONFIG_ARGS <<< "$CONFIGS"

for config in "${CONFIG_ARGS[@]}"; do
  echo "=== Ray Tune start: ${config} ==="
  python -m nba_forecasting tune \
    --config "$config" \
    --device "$DEVICE" \
    --output-root "$RUN_ROOT" \
    "${EXTRA_ARGS[@]}" \
    --search 'model.hidden_dim=choice(128,192,256)' \
    --search 'training.lr=loguniform(1e-4,5e-3)' \
    --search 'model.dropout=uniform(0.0,0.2)' \
    --num-samples "$NUM_SAMPLES" \
    --search-alg optuna \
    --scheduler asha \
    --cpus-per-trial "$CPUS_PER_TRIAL" \
    --gpus-per-trial "$GPUS_PER_TRIAL" \
    --max-concurrent-trials "$MAX_CONCURRENT_TRIALS"
  echo "=== Ray Tune complete: ${config} ==="
done
