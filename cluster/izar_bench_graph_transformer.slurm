#!/bin/bash -l
#SBATCH --job-name=nba-gt-bench
#SBATCH --partition=gpu
#SBATCH --qos=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:1
#SBATCH --mem=90G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --account=ee-452
#SBATCH --mail-type=NONE

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
IZAR_MODULES="${IZAR_MODULES:-gcc/11.3.0 cuda/12.1.1}"
RUN_ROOT="${RUN_ROOT:-/scratch/izar/$USER/nba_forecasting/graph_transformer_bench_$SLURM_JOB_ID}"
CONFIG="${CONFIG:-e5_graph_transformer}"
DEVICE="${DEVICE:-cuda}"
VARIANTS="${VARIANTS:-baseline,compile,compile_fused_graph_prep,compile_fused_graph_prep_graph_cache_fused_adamw}"
BATCH_SIZE="${BATCH_SIZE:-192}"
NUM_WORKERS="${NUM_WORKERS:-4}"
COMPONENT_WARMUP="${COMPONENT_WARMUP:-100}"
COMPONENT_ITERATIONS="${COMPONENT_ITERATIONS:-500}"
REAL_WARMUP_STEPS="${REAL_WARMUP_STEPS:-20}"
REAL_STEPS="${REAL_STEPS:-300}"
INCLUDE_REAL_STEP="${INCLUDE_REAL_STEP:-1}"
FAIL_FAST="${FAIL_FAST:-0}"
MIXED_PRECISION_MODES="${MIXED_PRECISION_MODES:-false,true}"

mkdir -p "$RUN_ROOT" "$PROJECT_ROOT/.mplconfig" logs
if [[ -z "${TMPDIR:-}" ]]; then
  export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/nba-gt-bench-$SLURM_JOB_ID}"
fi
mkdir -p "$TMPDIR"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$TMPDIR/torch_extensions}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"
export PYTHONUNBUFFERED=1

module purge
module load $IZAR_MODULES

cd "$PROJECT_ROOT"
source .venv/bin/activate

python -m nba_forecasting doctor --expect-cuda

EXTRA_ARGS=()
if [[ "$INCLUDE_REAL_STEP" != "1" ]]; then
  EXTRA_ARGS+=(--skip-real-step)
fi
if [[ "$FAIL_FAST" == "1" ]]; then
  EXTRA_ARGS+=(--fail-fast)
fi

python -m nba_forecasting bench-model-optimizations \
  --config "$CONFIG" \
  --variants "$VARIANTS" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --component-warmup "$COMPONENT_WARMUP" \
  --component-iterations "$COMPONENT_ITERATIONS" \
  --real-warmup-steps "$REAL_WARMUP_STEPS" \
  --real-steps "$REAL_STEPS" \
  --mixed-precision-modes "$MIXED_PRECISION_MODES" \
  --output-dir "$RUN_ROOT" \
  "${EXTRA_ARGS[@]}"
