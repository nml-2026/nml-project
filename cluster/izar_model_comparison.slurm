#!/bin/bash -l
#SBATCH --job-name=nba-mc
#SBATCH --partition=gpu
#SBATCH --qos=ee-452
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --account=ee-452
#SBATCH --mail-type=NONE
#SBATCH --array=0-9

# One model-comparison job per array task (1 GPU each). All tasks are independent.
#
# Usage:
#   python -m nba_forecasting model-comparison --smoke
#   sbatch cluster/izar_model_comparison.slurm

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
JOBS=(
  baseline
  joint_gru
  static_graph
  dynamic_graph_autoregressive
  dynamic_graph_direct
  graph_transformer
  graph_transformer_mdn
  mart
  connectivity_no_graph
  connectivity_shuffled_graph
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB="${JOBS[$TASK_ID]}"

source "$PROJECT_ROOT/cluster/izar_env.sh"
prepare_izar_dirs
load_izar_modules
activate_izar_env
configure_izar_runtime
check_izar_python

export LOG_FILE="$PROJECT_ROOT/logs/model_comparison_${SLURM_ARRAY_JOB_ID}_${TASK_ID}.log"

echo "array_job=$SLURM_ARRAY_JOB_ID task=$TASK_ID job=$JOB"
python -m nba_forecasting model-comparison \
  --job "$JOB" \
  --device cuda \
  --comparison-root "$PROJECT_ROOT/runs/model_comparison" \
  --connectivity-root "$PROJECT_ROOT/runs/connectivity_ablations" \
  --log-file "$LOG_FILE"
