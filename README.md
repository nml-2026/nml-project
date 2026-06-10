# NBA Trajectory Forecasting

Project repository for the EPFL EE-452 Network Machine Learning NBA trajectory forecasting project.

Given 8 observed time steps for 10 NBA players plus the ball, the code predicts the next 12 positions for all 11 entities. The pipeline covers data loading, train/validation/holdout splits, model training, local evaluation, Ray Tune searches, plotting, and Kaggle submission generation.

## Repository layout

- `src/nba_forecasting/`: package code for data loading, models, training, evaluation, plotting, and submission generation
- `configs/`: TOML experiment configs for baselines, graph models, and ablations
- `docs/`: setup guides, model notes, and experiment planning
- `cluster/`: Slurm scripts for SCITAS Izar
- `scripts/`: model-comparison batch runs, Ray Tune sweeps, and result collection
- `tests/`: smoke and regression tests
- `data/`: local dataset location
- `runs/`: local training outputs, ignored by git
- `artifacts/`: generated splits and cross-validation metadata, ignored by git

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
cd /path/to/nml-project
uv sync --extra cpu
source .venv/bin/activate
python -m nba_forecasting doctor
python -m nba_forecasting train --config baseline
```

Full local setup, data layout, and CLI examples: [docs/setup/local.md](docs/setup/local.md).

## Documentation

| Topic               | Guide                                                |
| ------------------- | ---------------------------------------------------- |
| Doc index           | [docs/README.md](docs/README.md)                     |
| Codebase map        | [docs/architecture.md](docs/architecture.md)         |
| Local setup         | [docs/setup/local.md](docs/setup/local.md)           |
| Izar / Slurm        | [docs/setup/izar.md](docs/setup/izar.md)             |
| Other GPU hosts     | [docs/setup/remote-gpu.md](docs/setup/remote-gpu.md) |
| Curated experiments | [docs/experiments.md](docs/experiments.md)           |
| Model families      | [docs/models.md](docs/models.md)                     |

## Supported workflow

The repository is curated for a single reproducible path:

1. `prepare-data` to freeze the split and consolidated tensors
2. `train` (single or batch) / `repeat-trials` / `cross-validate` for model training
3. `evaluate` / `submit` / `ensemble-evaluate` / `ensemble-submit` for scoring and Kaggle export
4. `tune` / `scripts/run_tune.sh` for hyperparameter sweeps; `model-comparison` for the report batch
5. `summarize-results` and `export-best-config` for result review

Curated configs, Tune scripts, and final candidates: [docs/experiments.md](docs/experiments.md).

## Demo smoke test

Quick check that the environment, tests, and CLI pipeline work (requires `data/`):

```bash
bash scripts/demo.sh
```

## Development checks

```bash
uv sync --extra cpu --extra dev
uv run ruff check src tests
uv run ruff format --check src tests
uv run ty check src
uv run pytest
```
