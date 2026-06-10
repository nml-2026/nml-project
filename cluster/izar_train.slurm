#!/bin/bash -l
#SBATCH --job-name=nba-train
#SBATCH --partition=gpu
#SBATCH --qos=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=ee-452
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
RUN_ROOT="${RUN_ROOT:-/scratch/izar/$USER/nba_forecasting/$SLURM_JOB_ID}"
CONFIGS="${CONFIGS:-${CONFIG:-baseline dynamic_graph_direct graph_transformer}}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
EVALUATE_SPLITS="${EVALUATE_SPLITS:-val holdout}"
TTA_GROUP="${TTA_GROUP:-}"

source "$PROJECT_ROOT/cluster/izar_env.sh"
prepare_izar_dirs
load_izar_modules
activate_izar_env
configure_izar_runtime
check_izar_python

read -r -a CONFIG_ARGS <<< "$CONFIGS"
read -r -a SEED_ARGS <<< "$SEEDS"
read -r -a SPLIT_ARGS <<< "$EVALUATE_SPLITS"

TRAIN_ARGS=(
  --config "${CONFIG_ARGS[@]}"
  --seeds "${SEED_ARGS[@]}"
  --device "$DEVICE"
  --output-root "$RUN_ROOT"
  --evaluate-splits "${SPLIT_ARGS[@]}"
)
if [[ -n "$EPOCHS" ]]; then
  TRAIN_ARGS+=(--epochs "$EPOCHS")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  TRAIN_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$TTA_GROUP" ]]; then
  TRAIN_ARGS+=(--set "evaluation.tta_group=$TTA_GROUP")
fi

python -m nba_forecasting train "${TRAIN_ARGS[@]}"
