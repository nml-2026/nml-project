# Model Overview

The implemented models live under `src/nba_forecasting/models/`. The public entry point is:

```python
from nba_forecasting.models import MODEL_FAMILIES, build_model
```

Model families are registered in `models/builders.py` with `FamilySpec` metadata (`builder`, `loss_modes`, `search_profiles`). Loss compatibility is validated at config load time.

Experiment starting points are TOML files under `configs/`. These configs are normal full-run defaults; quick tests should use runtime overrides such as `--epochs`, `training.max_train_batches`, and `training.max_val_batches`.

## Inputs

Each example uses 8 observed time steps for 11 entities: 10 players plus the ball. The base feature set is:

```text
[x, y, isplayer, team]
```

Velocity features (`vx`, `vy` as finite differences of normalized positions) are controlled by `features.use_velocity`. The default in `base.toml` is `true` (graph models, graph Transformer, MART). GRU reference configs (`baseline.toml`, `joint_gru.toml`) keep `use_velocity = false` because those families use the position-only feature layout. When enabled, the per-step vector is:

```text
[x, y, vx, vy, isplayer, team]
```

The target is the next 12 `(x, y)` positions for all 11 entities.

## Supported Model Families

### `baseline` (`BaselineModel`)

Reference non-graph baseline close to the original notebook-style model. It runs a small custom GRU cell independently per entity, uses no velocity features, and predicts positions autoregressively.

### `joint_gru` (`JointGRUForecastModel`)

Flattens all 11 entities into one joint GRU encoder over the observed context, then decodes all 12 horizon steps in a single direct head. No graph structure; tests whether explicit interaction modeling is necessary.

### `static_graph` (`StaticGraphDirectModel`)

Static full-graph MPNN over entities at each context step, followed by a joint temporal GRU and direct multi-step decoder. Graph topology is fixed (not position-dependent); edge features may still use semantic flags.

### `dynamic_graph_autoregressive` (`DynamicGraphAutoregressiveModel`)

Dynamic kNN graph MPNN with per-entity node GRU state and autoregressive delta decoding over the 12-step horizon. Supports teacher forcing and scheduled sampling during training.

### `dynamic_graph_direct`

Direct-decoder dynamic graph model. At each observed time step, the graph is rebuilt from current positions using k-nearest neighbors, optionally unioned with semantic edges. A node GRU maintains per-entity temporal state, then a horizon-query decoder predicts all 12 future steps in one pass.

The default config enables relation types, entity embeddings, a separate ball head, wider hidden dimensions, and larger k-neighborhoods.

## Graph Transformer

### `graph_transformer`

Spatiotemporal Transformer candidate with an explicit graph component. Each encoder stage alternates graph-masked spatial attention over entities with temporal attention over the observed context for each entity. Spatial attention is restricted by the dynamic graph and receives learned edge bias from relative position, distance, and semantic edge features.

The decoder builds horizon/entity query tokens and applies graph attention again before predicting cumulative position deltas.

Graph ablations (`no_graph`, `shuffled_graph`) are available via `model.graph_transformer.graph_ablation_mode` overrides rather than separate configs:

```bash
python -m nba_forecasting train --config graph_transformer \
  --set model.graph_transformer.graph_ablation_mode=no_graph
```

## MART

### `mart` (`mart_hypergraph`)

MART is included as a **published external baseline**, not as original architecture work in this repo. We port the official PyTorch implementation (MIT license; see the header in `src/nba_forecasting/models/mart.py`) and wire it into the same training, evaluation, and submission stack as our graph models.

#### Citation

- **Paper:** *MART: Multi-Agent Relational Transformer for Multi-Agent Trajectory Prediction*, arXiv:2407.21635 (2024) — [https://arxiv.org/abs/2407.21635](https://arxiv.org/abs/2407.21635)
- **Code:** [`gist-ailab/MART`](https://github.com/gist-ailab/MART) (ported in `src/nba_forecasting/models/mart.py`)

#### Prior work (architecture we reuse)

The ported encoder follows the official implementation:

- Per-entity temporal embedding with positional encoding over the observed context
- Parallel **pair-wise relational Transformer** (`MARTRT`) and **hyper-relational Transformer** (`HRT`) branches
- An adaptive group estimator in the hypergraph branch
- A multi-mode decoder head that predicts $K$ trajectory samples per entity

We do **not** claim the hypergraph design, the pair/hyper branches, or the group estimator as our contribution.

#### Our work (integration and recipe)

What we implemented or chose for the NBA benchmark:

| Area | What we did |
| --- | --- |
| **Pipeline** | `FamilySpec` registration, `mart.toml` / `tune_mart.toml`, shared data loaders, augmentations, trainer, checkpointing, xy TTA evaluation, and Kaggle submission |
| **Input adapter** | Map NBA tensors `[x, y, (vx, vy), isplayer, team]` into MART's expected layout (inherits `use_velocity=true` from `base.toml`) |
| **Decoder variant** | `mart_decoder_variant = "type_conditioned"` — separate ball/player bias on top of the ported multi-mode head |
| **Inference aggregation** | `mart_point_aggregation = "learned_softmax"` — learned weights over the $K{=}20$ modes at eval/submit time |
| **Training loss** | `loss.sample_loss_type = "softmin_ade"` with `loss.sample_loss_space = "raw"` (**softmin-raw**): temperature-weighted minADE in denormalized court coordinates, with tuned `softmin_temperature` and `raw_loss_scale_ft` |
| **Hyperparameters** | Ray Tune sweep (`tune_mart.toml`, `scripts/run_tune.sh`) merged into `mart.toml` (lr, schedule, width, `mart_function_type`, etc.) |
| **Protocol** | Same frozen split as other families; report-table `mart` uses reflect xy and agent permutation ($p{=}0.5$ each, `base.toml`); 100 epochs because validation was still improving at 80 |
| **Aug ablations** | `mart_aug_none`, `mart_aug_reflect`, `mart_aug_perm` isolate train-time reflection vs.\ agent permutation (see [experiments.md](experiments.md#mart-augmentation-ablations)) |

Optional knobs we added on top of the port (not used in the report table): Gaussian per-mode distributions (`mart_mode_distribution = "gaussian"`), alternate aggregations (`mean`, `argmax`, `distribution_*`), and additional loss modes wired through `resolve_training_loss_policy`.

For experiment commands and matched-budget comparison details, see [experiments.md](experiments.md).

## Additional Tunables

The code supports several knobs that are not all exposed as separate configs:

- `use_relation_types`: typed same-team / opponent / ball relations.
- `use_entity_embeddings`: learned embeddings for player/ball and team identity.
- `separate_ball_head`: separate output head for ball dynamics.
- `k_neighbors`: graph sparsity for dynamic edges.
- `transformer_heads`, `spatial_layers`, `temporal_layers`, `decoder_layers`, `transformer_ffn_multiplier`: graph Transformer capacity.
- `graph_ablation_mode`: normal, no-graph, or shuffled-graph behavior for graph Transformer ablations.
- `head_type`, `mdn_components`, `mdn_point_estimate`: Graph Transformer delta vs MDN forecast heads.
- `mart_function_type`, `mart_hyper_scale`, `mart_decoder_hidden_dim`, `mart_sample_k`, `mart_point_aggregation`, `mart_decoder_variant`, `mart_mode_distribution`: MART hypergraph capacity, multi-mode decoding, and aggregation behavior.
- `optimization_backend`, `optimizer_backend`, `runtime.mixed_precision`: global runtime optimization flags. CUDA tuned runs default to `torch.compile`, fused AdamW, and mixed precision; CPU and MPS runs fall back to eager execution, standard AdamW, and fp32.
