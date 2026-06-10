#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d data || ! -f data/train.csv || ! -f data/test.csv || ! -f data/sample_submission.csv ]]; then
  echo "data/ required: expected data/train.csv, data/test.csv, data/sample_submission.csv, data/train/*.pt, and data/test/*.pt" >&2
  exit 1
fi

python -m nba_forecasting doctor
python -m nba_forecasting prepare-data
python -m nba_forecasting train \
  --config dynamic_graph_direct \
  --device auto \
  --epochs 5 \
  --num-workers 0 \
  --name-suffix demo \
  --set training.max_train_batches=6 \
  --set training.max_val_batches=2
