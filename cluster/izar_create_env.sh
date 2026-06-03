#!/bin/bash -l
set -euo pipefail

# Izar: load CUDA modules, then install the uv CUDA 12.1 extra.
#
# Usage:
#   PROJECT_ROOT=$HOME/nml-project bash cluster/izar_create_env.sh

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/nml-project}"
UV_EXTRA="${UV_EXTRA:-cu121}"
IZAR_MODULES="${IZAR_MODULES:-gcc/11.3.0 cuda/12.1.1}"

module purge
module load $IZAR_MODULES

export PROJECT_ROOT UV_EXTRA
exec bash "$PROJECT_ROOT/cluster/create_env.sh"
