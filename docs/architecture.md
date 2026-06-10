# Codebase Architecture

High-level map of the repository. Use this with [models.md](models.md) and [experiments.md](experiments.md) when navigating the project.

## Design goals

- **One CLI** (`python -m nba_forecasting`) for prepare, train, evaluate, submit, tune, and diagnostics.
- **Config-driven experiments** — TOML under `configs/` inherits from `base.toml`; runtime overrides via `--set path=value`.
- **Registered model families** — each family has a builder, supported loss modes, and optional Ray Tune search profiles.
- **Reproducible splits** — frozen `artifacts/splits/default_split.json` shared across local and cluster runs.
- **Tested pipeline** — pytest covers configs, models, trainer, evaluator, submission, and CLI smoke paths.

## Repository layout

```text
configs/              Experiment defaults (TOML)
src/nba_forecasting/  Installable package
tests/                Unit and integration tests
docs/                 Setup guides, models, experiments, this file
scripts/              Batch runs (model comparison, tune, collect results)
cluster/              Izar Slurm helpers
data/                 Local dataset (gitignored)
runs/                 Training outputs (gitignored)
artifacts/            Frozen splits and CV metadata (gitignored)
```

## Package modules (`src/nba_forecasting/`)

| Module | Role |
| --- | --- |
| `cli.py` | Argument parser and dispatch for all commands |
| `config.py` | TOML loading, typed config dataclasses, overrides |
| `data.py` | Dataset I/O, [train-time augmentations](#train-time-augmentations), dataloaders, `prepare-data` |
| `metrics.py` | ADE/FDE, softmin and mixture losses |
| `doctor.py` | Environment and optional CUDA/Inductor checks |
| `ray_tune.py` | Hyperparameter search integration |
| `results.py` | Summarize runs, export best configs |
| `model_comparison.py` | Report-table job registry and batch runner |
| `plots.py` | Qualitative trajectory plots |
| `models/` | Model implementations and `MODEL_FAMILIES` registry |
| `runner/` | Training loop, evaluation, submission I/O |

### `models/`

| File | Families |
| --- | --- |
| `builders.py` | `FamilySpec` registry, `build_model`, loss policy |
| `baselines.py` | `baseline`, `joint_gru` |
| `dynamic_graph.py` | `static_graph`, `dynamic_graph_autoregressive`, `dynamic_graph_direct` |
| `graph_transformer.py` | `graph_transformer` (delta and MDN heads) |
| `mart.py` | `mart_hypergraph` (ported baseline; see [models.md](models.md#mart)) |
| `common.py` | Shared forecast types, graph prep, `forecast_model` helper |

### `runner/`

| File | Role |
| --- | --- |
| `trainer.py` | Epoch loop, early stopping, EMA/SWA hooks |
| `recipe.py` | Loss assembly (point, sample-mode, mixture NLL) |
| `evaluator.py` | Val/holdout metrics, TTA, ensembles |
| `submission.py` | Kaggle CSV export |
| `io.py` | Checkpoint load/save, device/autocast helpers |

## Typical data flow

```mermaid
flowchart LR
  TOML[configs/*.toml] --> Config[RunConfig]
  Data[data/*.pt] --> Loader[DataLoader]
  Config --> Build[build_model]
  Build --> Train[trainer.train]
  Loader --> Train
  Train --> CKPT[runs/.../best.pt]
  CKPT --> Eval[evaluate]
  CKPT --> Submit[submit]
```

1. **Prepare** (`prepare-data`) consolidates tensors and writes the frozen split.
2. **Train** loads config + data, builds the model family, runs the trainer, writes `history.csv` and `best.pt`.
3. **Evaluate** reloads checkpoint + config, runs metrics on val/holdout (optional xy TTA).
4. **Submit** writes `sample_submission.csv`-shaped predictions.

## Train-time augmentations {#train-time-augmentations}

Augmentations are configured under `features` in `base.toml` and applied only on the **training** loader (`data.py`). Validation and holdout keep the original court orientation and entity ordering.

### Court reflection

`features.reflection` mirrors normalized positions about the court axes:

- `y_prob` — left/right mirror ($y \to -y$ about the centerline)
- `x_prob` — attack-direction flip ($x \to -x$)
- `mode = "random"` — each axis is flipped independently per window with its own probability (default in `base.toml`: $p{=}0.5$ for both)

Reflections are label-preserving symmetries: team and entity-type static features are unchanged. Velocities flip sign along the mirrored axis. Implementation details and TTA inverse mapping live in `reflect_window` / `reflect_prediction` in `data.py`.

### Agent permutation

`features.augmentation.agent_permutation_prob` (default `0.5` in `base.toml`) randomly reorders **player** slots within a training window:

1. With probability $p$, `_permute_window_entities` runs after any reflection transform.
2. Player indices (all entities except the ball) are shuffled with `torch.randperm`.
3. The ball slot stays fixed (first entity with `isplayer = 0`).
4. The same permutation is applied to `context`, `future_xy`, `future_xy_raw`, `context_raw_xy`, and `static` so trajectories and labels stay aligned.

Player identities are not meaningful labels in this task—only relative geometry and team flags matter—so permuting slots encourages models (especially graph and set-based encoders) not to rely on a fixed player index. Evaluation and submission always use the dataset’s canonical ordering.

Optional transforms in the same pipeline (off by default): `context_noise_std_ft` (Gaussian noise on context positions) and `context_frame_dropout_prob` (drop or hold context frames).

## Config inheritance

```text
base.toml          Shared task shape, augmentations, default training knobs
  ├── baseline.toml / joint_gru.toml     GRU references (no velocity)
  ├── static_graph.toml … dynamic_graph_direct.toml
  ├── graph_transformer.toml / graph_transformer_mdn.toml
  ├── mart.toml
  └── tune_*.toml                      Ray Tune starting points
```

Override precedence: base → `extends` chain → CLI `--set` / `--epochs` / etc.

## Scripts vs CLI

| Need | Entry point |
| --- | --- |
| Single model, interactive | `python -m nba_forecasting train --config …` |
| Full report table batch | `python -m nba_forecasting model-comparison` |
| Hyperparameter search | `bash scripts/run_tune.sh` or `tune` CLI |
| Aggregate metrics | `python3 scripts/collect_results.py` |
| Quick health check | `python -m nba_forecasting doctor` |
| Demo / CI smoke | `bash scripts/demo.sh` |

## Tests

`tests/` mirrors package structure:

- `test_config.py`, `test_data.py`, `test_metrics.py` — core utilities
- `models/` — forward-pass and builder smoke per family
- `runner/` — trainer, evaluator, submission
- `test_cli.py` — end-to-end CLI on synthetic data

Run: `uv run pytest` (see [setup/local.md](setup/local.md#development-checks)).
