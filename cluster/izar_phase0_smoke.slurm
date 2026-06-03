#!/bin/bash -l
#SBATCH --job-name=nba-phase0-smoke
#SBATCH --partition=gpu
#SBATCH --qos=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=ee-452
#SBATCH --gres=gpu:1

# Phase 0 (evaluation foundation) smoke test on real data:
#   1. Short train of e3_dynamic_graph_direct -> history with val_ade_l2 / val_fde_l2.
#   2. Evaluate the resulting checkpoint on the full holdout split -> metrics.json
#      (ade_l2 / fde_l2 in court feet) + per_step_error.json / per_step_error.png.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
source "$PROJECT_ROOT/cluster/izar_env.sh"

mkdir -p "$PROJECT_ROOT/.mplconfig" logs

load_izar_modules
activate_izar_env
configure_izar_threads
check_izar_python

cd "$PROJECT_ROOT"

OUTPUT_ROOT="runs/izar_phase0"

python -m nba_forecasting train \
  --config e3_dynamic_graph_direct \
  --device "$DEVICE" \
  --epochs 1 \
  --output-root "$OUTPUT_ROOT" \
  --set training.max_train_batches=6 \
  --set training.max_val_batches=2

RUN_DIR="$(ls -dt "$OUTPUT_ROOT"/e3_dynamic_graph_direct/*/ | head -1)"
echo "PHASE0 run dir: $RUN_DIR"

python -m nba_forecasting evaluate \
  --checkpoint "${RUN_DIR%/}/best.pt" \
  --split holdout \
  --device "$DEVICE"

EVAL_DIR="${RUN_DIR%/}/eval_holdout"
echo "PHASE0 eval dir: $EVAL_DIR"
echo "===== PHASE0 history tail ====="
tail -n 3 "${RUN_DIR%/}/history.csv"
echo "===== PHASE0 holdout metrics.json ====="
cat "$EVAL_DIR/metrics.json"
echo
echo "===== PHASE0 holdout per_step_error.json ====="
cat "$EVAL_DIR/per_step_error.json"
echo
echo "===== PHASE0 artifacts ====="
ls -1 "$RUN_DIR" "$EVAL_DIR"
