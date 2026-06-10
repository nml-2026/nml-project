# Local Setup

Use [uv](https://docs.astral.sh/uv/) for local development. uv manages Python and all Python dependencies.

## Install uv

On Linux and macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows and other systems, see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

## Create an environment

From the repository root:

```bash
cd /path/to/nml-project
uv sync --extra cpu
source .venv/bin/activate
python -m nba_forecasting doctor
```

For development checks:

```bash
uv sync --extra cpu --extra dev
source .venv/bin/activate
python -m pytest -q
```

Torch variants are selected by uv extras. The CUDA extras use explicit PyTorch wheel indexes in `pyproject.toml`:

```bash
uv sync --extra cu121
uv sync --extra cu124
uv sync --extra cu128
source .venv/bin/activate
python -m nba_forecasting doctor --expect-cuda
```

Available Torch extras: `cpu`, `cu121`, `cu124`, and `cu128`.

## Data layout

Keep the Kaggle/NBA data under `data/`:

```text
data/
  train.csv
  test.csv
  sample_submission.csv
  train/*.pt
  test/*.pt
```

The Kaggle-provided CSV files and tensor directories are intentionally ignored by git. Each team member should download or copy the dataset locally into this layout. Do not commit datasets, checkpoints, run outputs, credentials, or Kaggle tokens to this public repository.

Each `.pt` sequence has shape `L x 11 x 4`, with features:

```text
[x, y, isplayer, team]
```

The first 8 time steps are the context window, and the target horizon is the next 12 time steps.

## Prepare portable artifacts

Before syncing to Izar or another remote GPU host, prepare consolidated tensors and the frozen split locally:

```bash
python -m nba_forecasting prepare-data
```

This creates:

```text
data/train_consolidated.pt
data/test_consolidated.pt
artifacts/splits/default_split.json
```

`default_split.json` is the frozen sequence-level train/validation/holdout split. Copy it to every remote machine so all jobs reuse the same split.

## Common commands

Train one experiment:

```bash
python -m nba_forecasting train --config baseline
```

Evaluate a checkpoint:

```bash
python -m nba_forecasting evaluate \
  --checkpoint runs/baseline/<timestamp>/best.pt \
  --split val
```

Generate a Kaggle submission:

```bash
python -m nba_forecasting submit \
  --checkpoint runs/dynamic_graph_direct/<timestamp>/best.pt \
  --output submission.csv
```

Short smoke training job on real data:

```bash
python -m nba_forecasting train \
  --config dynamic_graph_direct \
  --device auto \
  --epochs 1 \
  --set training.max_train_batches=6 \
  --set training.max_val_batches=2
```

Repeated trials or cross-validation:

```bash
python -m nba_forecasting repeat-trials \
  --config dynamic_graph_direct \
  --repeats 3 \
  --device auto

python -m nba_forecasting cross-validate \
  --config dynamic_graph_direct \
  --folds 5 \
  --device cpu
```

Ray Tune with Optuna search:

```bash
python -m nba_forecasting tune \
  --config dynamic_graph_direct \
  --device auto \
  --search-profile graph \
  --num-samples 12 \
  --search-alg optuna \
  --scheduler asha \
  --max-concurrent-trials 1
```

## DataLoader workers and CPU threads

`training.num_workers` in `base.toml` defaults to **`-1` (auto)**:

- **`0`**: disable workers (used in tests and smoke runs)
- **`-1`**: pick workers from `(cpu_count / training_slots) - 2`, capped at 8
- **`N > 0`**: explicit worker count (still capped per slot)

When running multiple trainings on one host (`model-comparison --gpus 0,1,…`), the launcher sets `NBA_TRAINING_SLOTS` to the number of concurrent GPU jobs so each process gets a fair CPU share. Training also sets `OMP_NUM_THREADS` / `MKL_NUM_THREADS` from the same budget. Progress bars are disabled automatically when `NBA_TRAINING_SLOTS > 1` (or set `NBA_FORECASTING_DISABLE_TQDM=1` manually).

Override for one run:

```bash
python -m nba_forecasting train --config baseline --num-workers 0
NBA_TRAINING_SLOTS=4 python -m nba_forecasting train --config graph_transformer
```

## Demo smoke test

For a quick end-to-end check (requires `data/`):

```bash
bash scripts/demo.sh
```

## Development checks

GitHub Actions runs linting, formatting checks, and tests on pull requests.

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check src
uv run ty check tests
uv run pytest
```

Optional pre-commit hooks after installing dev dependencies:

```bash
pre-commit install
```

## Workflow notes

- The pipeline freezes a sequence-level train/validation/holdout split and reuses it.
- Metrics are reported in raw coordinate space, even when training uses normalized coordinates.
- Logged ``ade`` is mean squared coordinate error in feet; ``ade_l2`` is mean Euclidean displacement per entity-step.
- `--device auto` prefers CUDA, then MPS, then CPU.
- Training outputs go under `runs/`.
- Batch experiment summaries go under `runs/batches/`.
- Cross-validation split metadata goes under `artifacts/cv/`.

## Remote environments

- SCITAS Izar (Slurm): [izar.md](izar.md)
- Other GPU hosts: [remote-gpu.md](remote-gpu.md)
