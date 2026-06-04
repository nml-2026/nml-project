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

### `e3_dynamic_graph_attention`

Same direct-decoder setup as `e3_dynamic_graph_direct`, but uses edge-aware attention for graph message passing. This is the lightweight graph-attention baseline over the 11 entities.

### `e3_dynamic_graph_mp_ssm`

Same direct-decoder setup as `e3_dynamic_graph_direct`, but uses an SSM-inspired message-passing layer. It keeps the standard graph interface while replacing plain mean aggregation with a gated node/message update.

## Graph + SSM Models

### `e3_graph_ssm`

Combines graph encoding with SSM-style temporal and social processing. First, each observed step is graph-encoded. Then temporal SSM layers process each entity's sequence of graph states. A social SSM block mixes entities based on egocentric distance-sorted social inputs that exclude the target entity itself. A gated fusion combines temporal and social states, and an SSM decoder predicts all horizon steps directly.

### `e3_graph_ssm_no_social`

Removes the learned social SSM layers from `e3_graph_ssm`. This is a true no-social path: the graph-temporal state bypasses the social branch before decoding.

## Graph Transformer

### `e5_graph_transformer`

Spatiotemporal Transformer candidate with an explicit graph component. Each encoder stage alternates graph-masked spatial attention over entities with temporal attention over the observed context for each entity. Spatial attention is restricted by the dynamic graph and receives learned edge bias from relative position, distance, and semantic edge features.

The decoder builds horizon/entity query tokens and applies graph attention again before predicting cumulative position deltas. This means graph structure affects both context encoding and horizon decoding.

### `e5_graph_transformer_no_graph`

Matched ablation for `e5_graph_transformer`. It keeps the same model capacity but replaces graph masks with dense entity attention and removes edge bias. This is not final-model eligible.

### `e5_graph_transformer_shuffled_graph`

Matched ablation for `e5_graph_transformer`. It keeps sparse graph-style masking but corrupts neighbor identity. This is not final-model eligible.

Treat `e5_graph_transformer` as a serious final candidate only if matched-budget validation runs beat both ablations.

## Current Ablation Configs

### `e4_no_dynamic_edges`

Extends `e3_dynamic_graph`, but disables dynamic kNN edges. This tests whether rebuilding the graph from positions is useful.

### `e4_no_semantic_flags`

Extends `e3_dynamic_graph`, but removes semantic edge flags such as same-team and involves-ball indicators. This tests whether basketball-specific metadata improves graph reasoning.

### `e4_no_velocity`

Extends `e3_dynamic_graph`, but removes velocity features. This tests whether explicit velocity helps beyond raw position history.

## Additional Tunables

The code supports several knobs that are not all exposed as separate configs:

- `message_passing_type`: MPNN, GAT-style attention, or SSM-inspired graph message passing.
- `decoder_type`: autoregressive vs direct horizon decoding for dynamic graph models.
- `use_relation_types`: typed same-team / opponent / ball relations.
- `use_entity_embeddings`: learned embeddings for player/ball and team identity.
- `separate_ball_head`: separate output head for ball dynamics.
- `k_neighbors`: graph sparsity for dynamic edges.
- `edge_radius`: optional distance cutoff.
- `temporal_ssm_layers`, `social_ssm_layers`, `decoder_ssm_layers`: where SSM capacity matters.
- `transformer_heads`, `spatial_layers`, `temporal_layers`, `decoder_layers`, `transformer_ffn_multiplier`: graph Transformer capacity.
- `graph_ablation_mode`: normal, no-graph, or shuffled-graph behavior for graph Transformer ablations.
- `optimization_backend`, `optimizer_backend`, `training.mixed_precision`: global runtime optimization flags. CUDA runs use `torch.compile`, fused AdamW, and autocast when requested; CPU and MPS runs fall back to eager execution, standard AdamW, and fp32.
- `teacher_forcing_prob` / `scheduled_sampling_final_prob`: rollout training strategy for autoregressive models.

## Main Comparisons

The first experiments should make three comparisons clear:

- Non-graph baselines vs graph models.
- Autoregressive rollout vs direct horizon decoding.
- GRU-style temporal modeling vs SSM-style temporal/social modeling.
