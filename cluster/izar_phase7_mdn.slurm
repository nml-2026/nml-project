#!/bin/bash -l
#SBATCH --job-name=nba-phase7-mdn
#SBATCH --partition=gpu
#SBATCH --qos=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=ee-452

# Phase 7 (D1): Mixture Density Network head + minADE_K on the frozen split. Trains the
# canonical e5_graph_transformer_tuned_reflect_xy with an MDN head at K in {1,3,5} (seed 0,
# same 50 ep / batch 192 recipe), evaluates each on holdout+val with TTA xy, and prints the
# deterministic point-estimate metrics (ade / ade_l2 / fde_l2) AND the multimodal
# minADE_K / minFDE_K. Generates an MDN submission (pi-weighted mean + TTA xy) for the LB
# record.
#
# Findings / framing that govern this phase (see implementation_plan.md / worklog.md):
#   - The competition scores ONE deterministic future with squared error; an NLL-trained MDN
#     top-mode/mean is NOT expected to beat the Phase-6 N=3 ensemble (LB 2.95365). This phase
#     is report/originality value + the minADE_K prerequisite, not a likely LB win.
#   - K=1 is the apples-to-apples anchor: an MDN with one mode should land NEAR the
#     deterministic baseline's holdout ade (~6.41 standalone / ~5.91 with TTA xy). A large gap
#     signals an NLL-vs-MSE point-estimate issue, not a bug.
#   - Checkpoint selection stays on val `ade` (point-estimate squared error) so selection
#     tracks Kaggle even though the objective is NLL.
#
# Mixing an MDN leg into the Phase-6 ensemble is GATED (like the e3 arch check): only if its
# standalone point-estimate holdout ade (TTA xy) is competitive with e5's ~5.91. This script
# prints the number and the follow-up command; it does NOT auto-mix.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-192}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs/izar_phase7}"
KS="${KS:-1 3 5}"
TTA="${TTA:-xy}"
# K used for the submission CSV (the multimodal headline; override after seeing the ablation).
SUBMIT_K="${SUBMIT_K:-3}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
SUBMIT_DIR="${SUBMIT_DIR:-$PROJECT_ROOT/runs/submissions}"

source "$PROJECT_ROOT/cluster/izar_env.sh"

mkdir -p "$PROJECT_ROOT/.mplconfig" logs "$SUBMIT_DIR"

load_izar_modules
activate_izar_env
configure_izar_threads
check_izar_python

cd "$PROJECT_ROOT"

read -r -a K_ARGS <<< "$KS"

resolve_ckpt() {  # resolve_ckpt NAME -> prints newest best.pt for that experiment name
  local name="$1"
  local run_dir
  run_dir="$(ls -dt "$OUTPUT_ROOT/$name"/*/ 2>/dev/null | head -1)"
  run_dir="${run_dir%/}"
  if [[ -z "$run_dir" || ! -f "$run_dir/best.pt" ]]; then
    echo "ERROR: no best.pt for $name under $OUTPUT_ROOT/$name" >&2
    exit 1
  fi
  echo "$run_dir/best.pt"
}

# -----------------------------------------------------------------------------
# 1. Train + evaluate the MDN head at each K (seed 0, frozen split).
# -----------------------------------------------------------------------------
for k in "${K_ARGS[@]}"; do
  name="e5_graph_transformer_tuned_reflect_xy_mdn_k${k}"
  echo "================================================================"
  echo "PHASE7: training MDN K=$k -> $name"
  echo "================================================================"
  python -m nba_forecasting train \
    --config e5_graph_transformer_tuned_reflect_xy_mdn \
    --device "$DEVICE" \
    --seed 0 \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --output-root "$OUTPUT_ROOT" \
    --set "experiment.name=$name" \
    --set "model.mdn_components=$k"

  ckpt="$(resolve_ckpt "$name")"
  for split in holdout val; do
    python -m nba_forecasting evaluate \
      --checkpoint "$ckpt" \
      --split "$split" \
      --device "$DEVICE" \
      --tta "$TTA" \
      --data-dir "$DATA_DIR"
    echo "===== PHASE7 MDN K=$k $split (tta $TTA) metrics.json ====="
    cat "$(dirname "$ckpt")/eval_${split}_tta_${TTA}/metrics.json"
    echo
  done
done

# -----------------------------------------------------------------------------
# 2. Submission from the chosen K (pi-weighted mean + TTA xy) for the LB record.
# -----------------------------------------------------------------------------
SUBMIT_NAME="e5_graph_transformer_tuned_reflect_xy_mdn_k${SUBMIT_K}"
SUBMIT_CKPT="$(resolve_ckpt "$SUBMIT_NAME")"
python -m nba_forecasting submit \
  --checkpoint "$SUBMIT_CKPT" \
  --device "$DEVICE" \
  --tta "$TTA" \
  --data-dir "$DATA_DIR" \
  --output "$SUBMIT_DIR/e5_mdn_k${SUBMIT_K}_tta_${TTA}_submission.csv"
echo "PHASE7 submission written: $SUBMIT_DIR/e5_mdn_k${SUBMIT_K}_tta_${TTA}_submission.csv"

# -----------------------------------------------------------------------------
# 3. GATED ensemble-leg check (NO auto-mix). Compare the chosen-K MDN standalone holdout ade
#    (TTA xy, printed above) to e5's ~5.91. Mix the MDN leg in only if competitive:
# -----------------------------------------------------------------------------
echo "GATE: if MDN K=$SUBMIT_K holdout ade (tta $TTA) is competitive with ~5.91, try mixing it in:"
echo "  python -m nba_forecasting ensemble-evaluate \\"
echo "    --checkpoints <phase6 seed0..seed2 best.pt> $SUBMIT_CKPT \\"
echo "    --split holdout --tta $TTA --data-dir $DATA_DIR"

echo "PHASE7 complete."
