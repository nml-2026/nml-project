#!/bin/bash -l
#SBATCH --job-name=nba-phase4-wavg
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

# Phase 4 (weight averaging) ablation on the frozen split, directly comparable to
# Phases 1-3 (same seed 0 / 50 epochs / batch 192 / e5_graph_transformer_tuned_reflect_xy
# base). For each configuration: train, then evaluate the resulting best.pt on val and
# holdout, printing metrics.json (competition `ade` squared; ade_l2 / fde_l2 in court feet).
#
#   OFF       reflect_xy baseline           (expect val 6.5122 / holdout 6.4066)
#   EMA 0.99  exponential moving average, decay 0.99
#   EMA 0.999 exponential moving average, decay 0.999  (headline config)
#   EMA 0.9999 exponential moving average, decay 0.9999
#   SWA 0.75  stochastic weight averaging, last 25% of epochs at the constant LR
#
# A train-many job can't vary ema_decay per config (it shares overrides), so each run is a
# separate train+evaluate driven by this loop.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-192}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/izar_phase4}"
source "$PROJECT_ROOT/cluster/izar_env.sh"

mkdir -p "$PROJECT_ROOT/.mplconfig" logs

load_izar_modules
activate_izar_env
configure_izar_threads
check_izar_python

cd "$PROJECT_ROOT"

# run_one NAME CONFIG [extra --set args...]
run_one() {
  local name="$1"; shift
  local config="$1"; shift

  echo "================================================================"
  echo "PHASE4 run: $name  (config=$config $*)"
  echo "================================================================"

  python -m nba_forecasting train \
    --config "$config" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-root "$OUTPUT_ROOT" \
    --set "experiment.name=$name" \
    "$@"

  local run_dir
  run_dir="$(ls -dt "$OUTPUT_ROOT/$name"/*/ | head -1)"
  run_dir="${run_dir%/}"
  echo "PHASE4 $name run dir: $run_dir"

  for split in val holdout; do
    python -m nba_forecasting evaluate \
      --checkpoint "$run_dir/best.pt" \
      --split "$split" \
      --device "$DEVICE"
    echo "===== PHASE4 $name $split metrics.json ====="
    cat "$run_dir/eval_$split/metrics.json"
    echo
  done

  echo "===== PHASE4 $name train_summary.json ====="
  cat "$run_dir/train_summary.json"
  echo
}

run_one "e5_graph_transformer_tuned_reflect_xy"           "e5_graph_transformer_tuned_reflect_xy"
run_one "e5_graph_transformer_tuned_reflect_xy_ema_d099"  "e5_graph_transformer_tuned_reflect_xy_ema" --set "training.ema_decay=0.99"
run_one "e5_graph_transformer_tuned_reflect_xy_ema"       "e5_graph_transformer_tuned_reflect_xy_ema"
run_one "e5_graph_transformer_tuned_reflect_xy_ema_d9999" "e5_graph_transformer_tuned_reflect_xy_ema" --set "training.ema_decay=0.9999"
run_one "e5_graph_transformer_tuned_reflect_xy_swa"       "e5_graph_transformer_tuned_reflect_xy_swa"

echo "PHASE4 ablation complete."
