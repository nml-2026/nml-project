# NBA Trajectory Forecasting

Project repository for the EPFL EE-452 Network Machine Learning NBA trajectory forecasting project.

Given 8 observed time steps for 10 NBA players plus the ball, the code predicts the next 12 positions for all 11 entities. The pipeline covers data loading, train/validation/holdout splits, model training, local evaluation, Ray Tune searches, plotting, and Kaggle submission generation.

## Repository Layout

- `src/nba_forecasting/`: package code for data loading, models, training, evaluation, plotting, and submission generation
- `configs/`: TOML experiment configs for baselines, graph models, and ablations
- `tests/`: smoke and regression tests
- `cluster/`: starter scripts for running experiments on SCITAS Izar
- `references/`: assignment brief and project reference material
- `data/`: local dataset location
- `runs/`: local training outputs, ignored by git
- `artifacts/`: generated splits and cross-validation metadata, ignored by git

## Setup

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development checks, install the optional dev dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run the test suite:

```bash
PYTHONPATH=src python -m pytest -q
```

For SCITAS Izar setup and Slurm jobs, see [cluster/README.md](cluster/README.md).

## Local Data

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

When copying the repository to Izar with `rsync`, use `.rsyncignore` from the repository root:

```bash
rsync -av --delete --exclude-from=.rsyncignore ./ izar:~/nml-project/
```

This copies the local `data/*.csv`, `data/train/*.pt`, and `data/test/*.pt` files, while excluding git metadata, references, logs, runs, caches, and build artifacts.

## Common Commands

Train one experiment:

```bash
PYTHONPATH=src python -m nba_forecasting train --config e0_reference
```

Evaluate a checkpoint:

```bash
PYTHONPATH=src python -m nba_forecasting evaluate \
  --checkpoint runs/e0_reference/<timestamp>/best.pt \
  --split val
```

Generate a Kaggle submission:

```bash
PYTHONPATH=src python -m nba_forecasting submit \
  --checkpoint runs/e3_dynamic_graph_direct/<timestamp>/best.pt \
  --output submission.csv
```

Run a short smoke training job on real data:

```bash
PYTHONPATH=src python -m nba_forecasting train \
  --config e3_dynamic_graph_direct \
  --device auto \
  --epochs 1 \
  --set training.max_train_batches=6 \
  --set training.max_val_batches=2
```

Run repeated trials or cross-validation:

```bash
PYTHONPATH=src python -m nba_forecasting repeat-trials \
  --config e3_dynamic_graph_direct \
  --repeats 3 \
  --device auto

PYTHONPATH=src python -m nba_forecasting cross-validate \
  --config e3_dynamic_graph_direct \
  --folds 5 \
  --device cpu
```

Run Ray Tune with Optuna search:

```bash
PYTHONPATH=src python -m nba_forecasting tune \
  --config e3_dynamic_graph_direct \
  --device auto \
  --search model.hidden_dim=choice(128,192,256) \
  --search training.lr=loguniform(1e-4,5e-3) \
  --search model.dropout=uniform(0.0,0.2) \
  --num-samples 12 \
  --search-alg optuna \
  --scheduler asha \
  --max-concurrent-trials 1
```

## Experiment Starting Points

- `e0_reference`: reference non-graph baseline
- `e1_joint_gru`: stronger non-graph GRU baseline
- `e1_joint_autoreg_gru`: autoregressive GRU baseline
- `e1_social_ssm`: state-space baseline with social mixing
- `e2_static_graph`: static graph model
- `e2_static_graph_gat`: static graph model with attention message passing
- `e3_dynamic_graph`: dynamic graph model
- `e3_dynamic_graph_direct`: direct 12-step graph decoder
- `e3_graph_ssm`: graph plus state-space-model blocks
- `e3_graph_ssm_no_social`: graph plus state-space-model blocks without social SSM layers
- `e4_no_dynamic_edges`, `e4_no_semantic_flags`, `e4_no_velocity`: ablations

Use `configs/base.toml` as the common configuration reference.
Configs are full-run defaults. For quick tests, keep the same config name and pass runtime overrides such as `--epochs`, `training.max_train_batches`, and `training.max_val_batches`.

## Development Checks

GitHub Actions runs linting, formatting checks, shell syntax checks, and tests on pull requests. Local pre-commit hooks are optional after installing the dev dependencies:

```bash
pre-commit install
```

## Workflow Notes

- The pipeline freezes a sequence-level train/validation/holdout split and reuses it.
- Metrics are reported in raw coordinate space, even when training uses normalized coordinates.
- `--device auto` prefers CUDA, then MPS, then CPU.
- Training outputs go under `runs/`.
- Batch experiment summaries go under `runs/batches/`.
- Cross-validation split metadata goes under `artifacts/cv/`.
