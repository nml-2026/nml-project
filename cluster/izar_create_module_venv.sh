#!/bin/bash -l
set -euo pipefail

# Usage:
#   PROJECT_ROOT=/path/to/nml-project \
#   VENV_PATH=$HOME/venvs/nba-forecasting-izar \
#   bash cluster/izar_create_module_venv.sh

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
source "$PROJECT_ROOT/cluster/izar_env.sh"

load_izar_modules

python --version
python -c "import torch; print('module torch', torch.__version__)"

python -m venv --system-site-packages "$VENV_PATH"
activate_izar_env

python -m pip install --upgrade pip
python -m pip install --upgrade --force-reinstall --no-cache-dir "numpy>=1.26,<2"
python -m pip install --no-cache-dir \
  "ray[tune]>=2.52.1" \
  "optuna>=3.6" \
  "matplotlib>=3.10" \
  "numpy>=1.26,<2" \
  "pandas>=2.2" \
  "tqdm>=4.66" \
  "tomli>=2.0"

python -m pip install --no-cache-dir --no-deps -e "$PROJECT_ROOT"

python -c "import numpy, torch; print('torch', torch.__version__); print('numpy', numpy.__version__)"
python -c "import matplotlib; print('matplotlib', matplotlib.__version__)"

echo "Virtual environment ready at $VENV_PATH"
echo "Activate it with: source $VENV_PATH/bin/activate"
