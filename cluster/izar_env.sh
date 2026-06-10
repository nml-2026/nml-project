#!/bin/bash
# Shared Izar runtime setup for Slurm jobs.
#
#   source "$PROJECT_ROOT/cluster/izar_env.sh"
#   load_izar_modules
#   activate_izar_env
#   configure_izar_runtime
#   check_izar_python

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
IZAR_MODULES="${IZAR_MODULES:-gcc/11.3.0 cuda/12.1.1}"
DEVICE="${DEVICE:-cuda}"

load_izar_modules() {
  module purge
  # shellcheck disable=SC2086
  module load $IZAR_MODULES
}

activate_izar_env() {
  cd "$PROJECT_ROOT"
  # shellcheck disable=SC1091
  source .venv/bin/activate
}

configure_izar_runtime() {
  export NBA_TRAINING_SLOTS="${NBA_TRAINING_SLOTS:-1}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$PROJECT_ROOT/.mplconfig}"
  export PYTHONUNBUFFERED=1
}

configure_izar_ray() {
  if [[ -z "${TMPDIR:-}" ]]; then
    export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/nba-ray-${SLURM_JOB_ID:-manual}}"
    mkdir -p "$TMPDIR"
  fi
  mkdir -p "$TMPDIR/ray" "$TMPDIR/ray_results"
  export RAY_DEDUP_LOGS=0
  export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0
}

check_izar_python() {
  python -m nba_forecasting doctor --expect-cuda
}

prepare_izar_dirs() {
  mkdir -p "$PROJECT_ROOT/.mplconfig" logs
  if [[ -n "${RUN_ROOT:-}" ]]; then
    mkdir -p "$RUN_ROOT"
  fi
}
