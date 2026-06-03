#!/bin/bash -l
#SBATCH --job-name=nba-phase6-ensemble
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

# Phase 6 (seed / arch ensemble) on the frozen split. Trains seeds 1-4 of the canonical
# e5_graph_transformer_tuned_reflect_xy (50 ep / batch 192, same recipe as the Phase-1/3
# seed-0 run), then runs the pre-registered ensemble + TTA-xy holdout ablation at N in {1,3,5}
# and the chosen-N submissions. Seed 0 is the EXISTING canonical checkpoint (LB-winning recipe,
# public LB 3.01065) and is reused UNCHANGED -- retraining it would not reproduce the
# holdout ade=5.9137 anchor (per-seed jitter ~2%).
#
# Findings that govern this phase (see implementation_plan.md / worklog.md):
#   - Ensemble the PLAIN reflect_xy seeds (no EMA/SWA) -- that is the LB-winning recipe.
#   - Variance levers OVERLAP: TTA xy already averages 4 reflected views, so the seed-ensemble
#     gain is measured ON TOP OF TTA xy and is expected to be SUBLINEAR.
#   - N=1 (ensemble of just seed 0) MUST reproduce holdout ade=5.9137 -- the apples-to-apples
#     anchor. If it deviates beyond 4th-decimal jitter, STOP (core averaging bug).
#   - Supersede the Phase-3 submission only on a >~2% holdout gain CONFIRMED on the public LB
#     vs 3.01065; otherwise log the ensemble as a hedge.
#
# Architecture mixing (e3_dynamic_graph_direct) is GATED: this script trains + evaluates e3
# standalone and prints its holdout ade (TTA xy). It does NOT auto-mix e3. Only if e3's
# standalone holdout ade is competitive with e5's (~5.91 with TTA) should a human add e3's
# best.pt to a follow-up ensemble-evaluate/ensemble-submit call.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-192}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/izar_phase6}"
SEEDS="${SEEDS:-1 2 3 4}"
TTA="${TTA:-xy}"
# Canonical seed-0 checkpoint (LB 3.01065). Override SEED0_CKPT if it lives elsewhere on Izar
# (e.g. the job-2945705 run dir or /scratch). Must be the SAME frozen-split recipe.
SEED0_CKPT="${SEED0_CKPT:-$PROJECT_ROOT/runs/phase1_ablation/tuned_xy/best.pt}"
# All legs read the same local data; override avoids depending on per-checkpoint baked paths.
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
E3_EPOCHS="${E3_EPOCHS:-35}"
SUBMIT_DIR="${SUBMIT_DIR:-$PROJECT_ROOT/runs/submissions}"

source "$PROJECT_ROOT/cluster/izar_env.sh"

mkdir -p "$PROJECT_ROOT/.mplconfig" logs "$SUBMIT_DIR"

load_izar_modules
activate_izar_env
configure_izar_threads
check_izar_python

cd "$PROJECT_ROOT"

if [[ ! -f "$SEED0_CKPT" ]]; then
  echo "ERROR: seed-0 checkpoint not found at $SEED0_CKPT" >&2
  echo "       set SEED0_CKPT to the canonical reflect_xy best.pt (LB 3.01065) on this machine." >&2
  exit 1
fi

# -----------------------------------------------------------------------------
# 1. Train seeds 1-4 of the canonical config (only the seed varies -> train-many).
# -----------------------------------------------------------------------------
read -r -a SEED_ARGS <<< "$SEEDS"
echo "================================================================"
echo "PHASE6: training seeds [$SEEDS] of e5_graph_transformer_tuned_reflect_xy"
echo "================================================================"
python -m nba_forecasting train-many \
  --configs e5_graph_transformer_tuned_reflect_xy \
  --seeds "${SEED_ARGS[@]}" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --output-root "$OUTPUT_ROOT" \
  --name-suffix phase6 \
  --evaluate-splits val holdout

# -----------------------------------------------------------------------------
# 2. Resolve the per-seed best.pt paths (newest run dir per seed name).
# -----------------------------------------------------------------------------
resolve_ckpt() {  # resolve_ckpt SEED -> prints best.pt path
  local seed="$1"
  local name="e5_graph_transformer_tuned_reflect_xy_phase6_s${seed}"
  local run_dir
  run_dir="$(ls -dt "$OUTPUT_ROOT/$name"/*/ 2>/dev/null | head -1)"
  run_dir="${run_dir%/}"
  if [[ -z "$run_dir" || ! -f "$run_dir/best.pt" ]]; then
    echo "ERROR: no best.pt for seed $seed under $OUTPUT_ROOT/$name" >&2
    exit 1
  fi
  echo "$run_dir/best.pt"
}

CKPT_S1="$(resolve_ckpt 1)"
CKPT_S2="$(resolve_ckpt 2)"
CKPT_S3="$(resolve_ckpt 3)"
CKPT_S4="$(resolve_ckpt 4)"

# Ordered checkpoint lists; seed 0 (the anchor) is always FIRST so it drives the base
# config/split for evaluation.
CKPTS_N1=("$SEED0_CKPT")
CKPTS_N3=("$SEED0_CKPT" "$CKPT_S1" "$CKPT_S2")
CKPTS_N5=("$SEED0_CKPT" "$CKPT_S1" "$CKPT_S2" "$CKPT_S3" "$CKPT_S4")

echo "PHASE6 checkpoints:"
echo "  seed0: $SEED0_CKPT"
echo "  seed1: $CKPT_S1"
echo "  seed2: $CKPT_S2"
echo "  seed3: $CKPT_S3"
echo "  seed4: $CKPT_S4"

# -----------------------------------------------------------------------------
# 3. Ensemble + TTA-xy ablation: holdout + val for N in {1,3,5}.
#    N=1 holdout ade MUST reproduce 5.9137 (the anchor).
# -----------------------------------------------------------------------------
eval_ensemble() {  # eval_ensemble N CKPT...
  local n="$1"; shift
  for split in holdout val; do
    local out_dir="$OUTPUT_ROOT/eval_ensemble_${split}_n${n}_tta_${TTA}"
    python -m nba_forecasting ensemble-evaluate \
      --checkpoints "$@" \
      --split "$split" \
      --device "$DEVICE" \
      --tta "$TTA" \
      --data-dir "$DATA_DIR" \
      --output-dir "$out_dir"
    echo "===== PHASE6 ensemble N=$n $split metrics.json ====="
    cat "$out_dir/metrics.json"
    echo
  done
}

eval_ensemble 1 "${CKPTS_N1[@]}"
eval_ensemble 3 "${CKPTS_N3[@]}"
eval_ensemble 5 "${CKPTS_N5[@]}"

# -----------------------------------------------------------------------------
# 4. Submissions for N=3 and N=5 (cheap) -- upload whichever clears the LB gate.
# -----------------------------------------------------------------------------
python -m nba_forecasting ensemble-submit \
  --checkpoints "${CKPTS_N3[@]}" --device "$DEVICE" --tta "$TTA" --data-dir "$DATA_DIR" \
  --output "$SUBMIT_DIR/e5_ensemble_n3_tta_${TTA}_submission.csv"
python -m nba_forecasting ensemble-submit \
  --checkpoints "${CKPTS_N5[@]}" --device "$DEVICE" --tta "$TTA" --data-dir "$DATA_DIR" \
  --output "$SUBMIT_DIR/e5_ensemble_n5_tta_${TTA}_submission.csv"
echo "PHASE6 submissions written to $SUBMIT_DIR (n3, n5)."

# -----------------------------------------------------------------------------
# 5. GATED architecture-diversity check: train + evaluate e3 standalone (NO auto-mix).
#    Only mix e3 in (a follow-up ensemble call) if its holdout ade is competitive with e5.
# -----------------------------------------------------------------------------
echo "================================================================"
echo "PHASE6: gated e3_dynamic_graph_direct standalone check"
echo "================================================================"
python -m nba_forecasting train \
  --config e3_dynamic_graph_direct \
  --device "$DEVICE" \
  --seed 0 \
  --epochs "$E3_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --output-root "$OUTPUT_ROOT" \
  --set "experiment.name=e3_dynamic_graph_direct_phase6"

E3_RUN_DIR="$(ls -dt "$OUTPUT_ROOT/e3_dynamic_graph_direct_phase6"/*/ | head -1)"
E3_RUN_DIR="${E3_RUN_DIR%/}"
python -m nba_forecasting evaluate \
  --checkpoint "$E3_RUN_DIR/best.pt" \
  --split holdout \
  --device "$DEVICE" \
  --tta "$TTA" \
  --data-dir "$DATA_DIR"
echo "===== PHASE6 e3 standalone holdout (tta $TTA) metrics.json ====="
cat "$E3_RUN_DIR/eval_holdout_tta_${TTA}/metrics.json"
echo
echo "GATE: compare e3 holdout ade above to e5's ~5.91 (TTA xy). Mix e3 in only if competitive:"
echo "  python -m nba_forecasting ensemble-evaluate --checkpoints ${CKPTS_N5[*]} $E3_RUN_DIR/best.pt --split holdout --tta $TTA --data-dir $DATA_DIR"

echo "PHASE6 complete."
