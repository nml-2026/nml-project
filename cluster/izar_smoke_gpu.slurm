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
#SBATCH --mail-type=NONE
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
IZAR_MODULES="${IZAR_MODULES:-gcc/11.3.0 cuda/12.1.1}"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$PROJECT_ROOT/.mplconfig" logs

module purge
module load $IZAR_MODULES
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"
export PYTHONUNBUFFERED=1

cd "$PROJECT_ROOT"
source .venv/bin/activate

python -m nba_forecasting doctor --expect-cuda

python -m nba_forecasting train \
  --config e3_dynamic_graph_direct \
  --device "$DEVICE" \
  --epochs 1 \
  --output-root runs/izar_smoke \
  --set training.max_train_batches=2 \
  --set training.max_val_batches=1
