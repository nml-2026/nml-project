# Model Overview

The implemented models live under `src/nba_forecasting/models/`. The public entry point is:

```python
from nba_forecasting.models import build_model
```

Experiment starting points are TOML files under `configs/`. These configs are normal full-run defaults; quick tests should use runtime overrides such as `--epochs`, `training.max_train_batches`, and `training.max_val_batches`.

## Inputs

Each example uses 8 observed time steps for 11 entities: 10 players plus the ball. The base feature set is:

```text
[x, y, isplayer, team]
```

Most non-reference configs also enable velocity:

```text
[x, y, vx, vy, isplayer, team]
```

The target is the next 12 `(x, y)` positions for all 11 entities.

## Baselines

### `e0_reference`

Reference non-graph baseline. It is intentionally simple and close to the original notebook-style model. It runs a small custom GRU cell independently per entity, uses no velocity features, and predicts positions autoregressively.

### `e1_joint_gru`

Stronger non-graph baseline. It flattens all entities and features at each time step, encodes the 8-step context with a GRU, then directly predicts the full 12-step future for all entities. It can model interactions only implicitly through the joint flattened representation.

### `e1_joint_autoreg_gru`

Autoregressive version of the joint GRU. It encodes the context jointly, then rolls out one step at a time with a GRUCell, predicting position deltas. During training it supports teacher forcing and scheduled sampling through the shared training loop.

## SSM Baseline

### `e1_social_ssm`

Non-GNN state-space baseline with explicit social mixing. It first applies SSM blocks over each entity's temporal context. Then it builds a social sequence for each entity by sorting other entities by distance, projects edge features, and applies social SSM layers. A learned gate fuses temporal and social states, and an SSM decoder predicts the horizon.

This tests whether SSM-style temporal/social modeling is competitive without graph message passing.

## Graph Models

### `e2_static_graph`

Static graph baseline. For each observed time step, it applies graph layers over a fixed full entity graph, using edge features based on relative position, distance, and optional semantic flags. The per-step graph embeddings are flattened and passed through a GRU over time, then decoded directly into the full future horizon.

### `e2_static_graph_gat`

Same as `e2_static_graph`, but swaps the graph layer from MPNN-style message aggregation to edge-aware attention.

### `e3_dynamic_graph`

Main dynamic graph autoregressive model. At each time step, the graph is rebuilt from current positions using k-nearest neighbors, optionally unioned with semantic edges. A node GRU maintains per-entity temporal state. During prediction, it rolls forward autoregressively, repeatedly rebuilding graph features from predicted positions and predicting deltas.

### `e3_dynamic_graph_direct`

Direct-decoder dynamic graph model. It uses the same dynamic graph encoder idea, but instead of rolling out step by step, it decodes all 12 future steps with learned horizon queries and cross-attention over the encoded entity states.

The config also enables relation types, entity embeddings, a separate ball head, wider hidden dimensions, and larger k-neighborhoods.

## Graph + SSM Models

### `e3_graph_ssm`

Combines graph encoding with SSM-style temporal and social processing. First, each observed step is graph-encoded. Then temporal SSM layers process each entity's sequence of graph states. A social SSM block mixes entities based on distance-sorted social inputs. A gated fusion combines temporal and social states, and an SSM decoder predicts all horizon steps directly.

### `e3_graph_ssm_no_social`

Removes the learned social SSM layers from `e3_graph_ssm`. This tests whether the social SSM component helps beyond graph encoding plus temporal SSM dynamics.

## Current Ablation Configs

### `e4_no_dynamic_edges`

Extends `e3_dynamic_graph`, but disables dynamic kNN edges. This tests whether rebuilding the graph from positions is useful.

### `e4_no_semantic_flags`

Extends `e3_dynamic_graph`, but removes semantic edge flags such as same-team and involves-ball indicators. This tests whether basketball-specific metadata improves graph reasoning.

### `e4_no_velocity`

Extends `e3_dynamic_graph`, but removes velocity features. This tests whether explicit velocity helps beyond raw position history.

## Additional Tunables

The code supports several knobs that are not all exposed as separate configs:

- `message_passing_type`: MPNN vs GAT-style graph layers.
- `decoder_type`: autoregressive vs direct horizon decoding for dynamic graph models.
- `use_relation_types`: typed same-team / opponent / ball relations.
- `use_entity_embeddings`: learned embeddings for player/ball and team identity.
- `separate_ball_head`: separate output head for ball dynamics.
- `k_neighbors`: graph sparsity for dynamic edges.
- `edge_radius`: optional distance cutoff.
- `temporal_ssm_layers`, `social_ssm_layers`, `decoder_ssm_layers`: where SSM capacity matters.
- `teacher_forcing_prob` / `scheduled_sampling_final_prob`: rollout training strategy for autoregressive models.

## Main Comparisons

The first experiments should make three comparisons clear:

- Non-graph baselines vs graph models.
- Autoregressive rollout vs direct horizon decoding.
- GRU-style temporal modeling vs SSM-style temporal/social modeling.
