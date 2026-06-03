#!/bin/bash -l
#SBATCH --job-name=nba-phase5-lrsched
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

# Phase 5 (LR schedule + longer training, and SWA revisit with SWALR) ablation on the frozen
# split, comparable to Phases 1-4 (seed 0 / batch 192 / e5_graph_transformer_tuned_reflect_xy
# base). Budget raised to 80 epochs so cosine can anneal fully. For each configuration: train,
# then evaluate best.pt on val and holdout, printing metrics.json (competition `ade` squared;
# ade_l2 / fde_l2 in court feet).
#
#   OFF           reflect_xy baseline, fixed LR (patience 10 early-stops; re-anchors ~6.4066)
#   cosine        cosine-annealed LR, 80 ep / patience 20
#   cosine_warmup 5-epoch linear warmup then cosine, 80 ep / patience 20
#   swalr         SWA with SWALR elevated-LR tail (swa_start 0.75, swa_lr 1.5e-4) -- the
#                 Phase-5 SWA revisit; compare to Phase-4 constant-LR SWA (holdout 6.2471)
#                 and EMA 0.999 (6.0117)
#   ema_cosine    EMA 0.999 + cosine schedule (does the schedule help the Phase-4 winner?)
#   ema_swalr     EMA 0.999 + SWALR-SWA together (optional combined recipe)
#
# Phase-4 anchors for the table: EMA 0.999 = 6.0117, constant-LR SWA = 6.2471 (holdout ade).
#
# Optional LR re-tune (separate job, not this script): run the existing tune path with
#   --search-profile lr_scheduler  (sweeps training.lr + training.swa_lr) against
#   e5_graph_transformer_tuned_reflect_xy_swalr, e.g. via cluster/izar_tune_single_node.slurm.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-192}"
SEED="${SEED:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/izar_phase5}"
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
  echo "PHASE5 run: $name  (config=$config $*)"
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
  echo "PHASE5 $name run dir: $run_dir"

  for split in val holdout; do
    python -m nba_forecasting evaluate \
      --checkpoint "$run_dir/best.pt" \
      --split "$split" \
      --device "$DEVICE"
    echo "===== PHASE5 $name $split metrics.json ====="
    cat "$run_dir/eval_$split/metrics.json"
    echo
  done

  echo "===== PHASE5 $name train_summary.json ====="
  cat "$run_dir/train_summary.json"
  echo
}

run_one "e5_graph_transformer_tuned_reflect_xy"               "e5_graph_transformer_tuned_reflect_xy"
run_one "e5_graph_transformer_tuned_reflect_xy_cosine"        "e5_graph_transformer_tuned_reflect_xy_cosine"
run_one "e5_graph_transformer_tuned_reflect_xy_cosine_warmup" "e5_graph_transformer_tuned_reflect_xy_cosine_warmup"
run_one "e5_graph_transformer_tuned_reflect_xy_swalr"         "e5_graph_transformer_tuned_reflect_xy_swalr"
run_one "e5_graph_transformer_tuned_reflect_xy_ema_cosine"    "e5_graph_transformer_tuned_reflect_xy_ema" \
  --set "training.lr_scheduler=cosine" --set "training.patience=20"
run_one "e5_graph_transformer_tuned_reflect_xy_ema_swalr"     "e5_graph_transformer_tuned_reflect_xy_swalr" \
  --set "training.ema_decay=0.999"

echo "PHASE5 ablation complete."
