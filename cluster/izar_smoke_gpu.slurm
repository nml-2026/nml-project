#!/bin/bash -l
#SBATCH --job-name=nba-smoke-gpu
#SBATCH --partition=gpu
#SBATCH --qos=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=ee-452
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
source "$PROJECT_ROOT/cluster/izar_env.sh"

mkdir -p "$PROJECT_ROOT/.mplconfig" logs

load_izar_modules
activate_izar_env
configure_izar_threads
check_izar_python

cd "$PROJECT_ROOT"

python -m nba_forecasting train \
  --config e3_dynamic_graph_direct \
  --device "$DEVICE" \
  --epochs 1 \
  --output-root runs/izar_smoke \
  --set training.max_train_batches=2 \
  --set training.max_val_batches=1
