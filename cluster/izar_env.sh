#!/bin/bash

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
VENV_PATH="${VENV_PATH:-$HOME/venvs/nba-forecasting-izar}"
IZAR_MODULES="${IZAR_MODULES:-gcc/11.3.0 openmpi/4.1.3-cuda python cuda py-torch/2.0.1-mpi-openmp-cuda}"
DEVICE="${DEVICE:-cuda}"

load_izar_modules() {
  module purge
  module load $IZAR_MODULES
}

activate_izar_env() {
  source "$VENV_PATH/bin/activate"
  local venv_site
  venv_site="$(python -c 'import site; print(site.getsitepackages()[0])')"
  export PYTHONPATH="$venv_site:$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
  export MPLCONFIGDIR="$PROJECT_ROOT/.mplconfig"
  mkdir -p "$MPLCONFIGDIR"
}

configure_izar_threads() {
  local thread_count="${SLURM_CPUS_PER_TASK:-1}"
  export OMP_NUM_THREADS="$thread_count"
  export MKL_NUM_THREADS="$thread_count"
  export PYTHONUNBUFFERED=1
}

check_izar_python() {
  python -c "import numpy, torch; print('python ok:', 'torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count(), 'numpy', numpy.__version__)"
  if [[ "$DEVICE" == "cuda" ]]; then
    python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 'DEVICE=cuda requested, but this allocation does not expose CUDA')"
  fi
}
