#!/bin/bash -l
#SBATCH --job-name=nba-smoke
#SBATCH --partition=gpu
#SBATCH --qos=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=ee-452
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
CONFIG="${CONFIG:-dynamic_graph_direct}"
EPOCHS="${EPOCHS:-1}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-2}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-1}"

source "$PROJECT_ROOT/cluster/izar_env.sh"
prepare_izar_dirs
load_izar_modules
activate_izar_env
configure_izar_runtime
check_izar_python

python -m nba_forecasting train \
  --config "$CONFIG" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --output-root runs/izar_smoke \
  --set "training.max_train_batches=$MAX_TRAIN_BATCHES" \
  --set "training.max_val_batches=$MAX_VAL_BATCHES"
