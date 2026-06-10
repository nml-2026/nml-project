#!/usr/bin/env bash
# Ray Tune sweeps for graph Transformer MDN and MART hyperparameters.
#
# Usage:
#   bash scripts/run_tune.sh                  # both models (sequential)
#   TARGET=mart bash scripts/run_tune.sh
#   TARGET=graph_transformer bash scripts/run_tune.sh
#   GPUS=0,1 NUM_SAMPLES=24 bash scripts/run_tune.sh

set -euo pipefail

TARGET="${TARGET:-both}"
GPUS="${GPUS:-0,1}"
NUM_GPUS="${NUM_GPUS:-2}"
NUM_CPUS="${NUM_CPUS:-}"
NUM_SAMPLES="${NUM_SAMPLES:-24}"
MAX_CONCURRENT_TRIALS="${MAX_CONCURRENT_TRIALS:-2}"
CPUS_PER_TRIAL="${CPUS_PER_TRIAL:-2}"
OUTPUT_ROOT="${OUTPUT_ROOT:-runs}"
PRUNE_VAL_ADE="${PRUNE_VAL_ADE:-20}"
PRUNE_MIN_EPOCH="${PRUNE_MIN_EPOCH:-3}"

export CUDA_VISIBLE_DEVICES="$GPUS"

RAY_CPU_ARGS=()
if [[ -n "$NUM_CPUS" ]]; then
  RAY_CPU_ARGS+=(--ray-num-cpus "$NUM_CPUS")
fi

COMMON_ARGS=(
  --device cuda
  --output-root "$OUTPUT_ROOT"
  --epochs 100
  --num-workers 4
  --set training.patience=100
  --search-alg optuna
  --scheduler asha
  --grace-period 10
  --reduction-factor 3
  --num-samples "$NUM_SAMPLES"
  --gpus-per-trial 1
  --cpus-per-trial "$CPUS_PER_TRIAL"
  "${RAY_CPU_ARGS[@]}"
  --ray-num-gpus "$NUM_GPUS"
  --max-concurrent-trials "$MAX_CONCURRENT_TRIALS"
  --prune-metric-threshold "$PRUNE_VAL_ADE"
  --prune-min-epoch "$PRUNE_MIN_EPOCH"
)

run_graph_transformer_tune() {
  echo "=== Graph Transformer MDN Optuna sweep ==="
  python -m nba_forecasting tune \
    --config configs/tune_graph_transformer.toml \
    "${COMMON_ARGS[@]}" \
    --search "model.family=choice(graph_transformer)" \
    --search "model.head.mdn_point_estimate=choice(mean,mode,medoid)" \
    --search "model.graph_transformer.num_virtual_nodes=choice(0,2)" \
    --search "training.lr=choice(0.000349,0.0005)" \
    --search "training.weight_decay=choice(0.0,0.0001)" \
    "$@"
}

run_mart_tune() {
  echo "=== MART Optuna sweep ==="
  python -m nba_forecasting tune \
    --config configs/tune_mart.toml \
    "${COMMON_ARGS[@]}" \
    --search "model.mart.mart_decoder_variant=choice(type_conditioned)" \
    --search "model.mart.mart_point_aggregation=choice(learned_softmax)" \
    --search "loss.sample_loss_type=choice(hard_min_ade,softmin_ade)" \
    --search "loss.sample_loss_space=choice(normalized,raw)" \
    --search "training.lr_scheduler=choice(multistep,cosine_warmup)" \
    --search "training.lr_milestones=choice([60,85],[100,200])" \
    --search "training.lr_gamma=choice(0.5)" \
    --search "training.lr=loguniform(2e-4,8e-4)" \
    --search "training.lr_warmup_epochs=choice(3,5,8)" \
    --search "training.lr_min=loguniform(1e-6,5e-5)" \
    --search "training.weight_decay=choice(0.0,0.0001)" \
    --search "loss.softmin_temperature=uniform(1.0,4.0)" \
    --search "loss.raw_loss_scale_ft=uniform(3.0,8.0)" \
    "$@"
}

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES TARGET=$TARGET NUM_SAMPLES=$NUM_SAMPLES"

case "$TARGET" in
  graph_transformer)
    run_graph_transformer_tune "$@"
    ;;
  mart)
    run_mart_tune "$@"
    ;;
  both)
    run_graph_transformer_tune "$@"
    run_mart_tune "$@"
    ;;
  *)
    echo "TARGET must be graph_transformer, mart, or both" >&2
    exit 1
    ;;
esac
