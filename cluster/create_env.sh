#!/bin/bash
set -euo pipefail

# Bootstrap the selected uv extra.
#
# Usage:
#   PROJECT_ROOT=/workspace/nml-project UV_EXTRA=cu124 bash cluster/create_env.sh
#
# Torch variants are pyproject extras: cpu, cu121, cu124, cu128.

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/nml-project}"
UV_EXTRA="${UV_EXTRA:-cu124}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

if [[ ! -d "$PROJECT_ROOT" ]]; then
  echo "PROJECT_ROOT does not exist: $PROJECT_ROOT" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install it with:" >&2
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "Other systems: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
uv sync --python "$PYTHON_VERSION" --extra "$UV_EXTRA"
source .venv/bin/activate
python -m nba_forecasting doctor

echo "uv environment ready with extra: $UV_EXTRA"
echo "Activate it with: source .venv/bin/activate"
echo "Then run commands with: python -m nba_forecasting ..."
