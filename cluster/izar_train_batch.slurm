#!/bin/bash -l
#SBATCH --job-name=nba-train-batch
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
IZAR_MODULES="${IZAR_MODULES:-gcc/11.3.0 cuda/12.1.1}"
RUN_ROOT="${RUN_ROOT:-/scratch/izar/$USER/nba_forecasting/$SLURM_JOB_ID}"
CONFIGS="${CONFIGS:-e0_reference e1_joint_gru e1_social_ssm e3_dynamic_graph_direct e3_graph_ssm_no_social}"
DEVICE="${DEVICE:-cuda}"
SEEDS="${SEEDS:-0}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-192}"
EVALUATE_SPLITS="${EVALUATE_SPLITS:-val holdout}"

mkdir -p "$RUN_ROOT" "$PROJECT_ROOT/.mplconfig" logs

module purge
module load $IZAR_MODULES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
source .venv/bin/activate

python -m nba_forecasting doctor --expect-cuda

read -r -a CONFIG_ARGS <<< "$CONFIGS"
read -r -a SEED_ARGS <<< "$SEEDS"
read -r -a SPLIT_ARGS <<< "$EVALUATE_SPLITS"

python -m nba_forecasting train-many \
  --configs "${CONFIG_ARGS[@]}" \
  --seeds "${SEED_ARGS[@]}" \
  --device "$DEVICE" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --output-root "$RUN_ROOT" \
  --evaluate-splits "${SPLIT_ARGS[@]}"
