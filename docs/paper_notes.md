# Paper-Guided Model Notes

These notes summarize implementation ideas we plan to test. The papers remain references; external repositories are not vendored into this project.

## Social-Mamba

Reference: see [references](references.md).

Implementation-relevant ideas:

- Use egocentric social ordering rather than a global entity order.
- Exclude the target agent from its own social sequence.
- Keep temporal and social processing semantically separate, then fuse them explicitly.
- Treat a no-social ablation as a real bypass of the social branch.

Project mapping:

- `e1_social_ssm` and `e3_graph_ssm` now build self-excluding distance-sorted social sequences.
- `e3_graph_ssm_no_social` sets `social_ssm_layers = 0`, which bypasses social fusion and decodes from the temporal graph state.

## GraphSSM

Reference: [GraphSSM](references.md#papers).

Useful code reference: [EdisonLeeeee/GraphSSM](https://github.com/EdisonLeeeee/GraphSSM)

Implementation-relevant ideas:

- Treat graph data as a sequence of graph snapshots.
- Separate temporal evolution from graph propagation.
- Use graph-aware state updates rather than flattening all nodes into one sequence.

Project mapping:

- `e3_graph_ssm` graph-encodes each observed time step, applies temporal SSM layers per entity, then applies social SSM fusion.
- Search profile `graph_ssm` tunes SSM layer counts and convolution kernel size.

## Message-Passing SSM

Reference: [Message-Passing State-Space Models](references.md#papers).

Implementation-relevant ideas:

- Preserve message-passing compatibility and permutation equivariance.
- Use SSM-style gated state updates inside graph propagation instead of replacing the graph with a sequence-only model.
- Keep the layer dependency-free unless results justify a larger implementation.

Project mapping:

- `message_passing_type = "mp_ssm"` adds a lightweight gated message-passing layer.
- `e3_dynamic_graph_mp_ssm` is the first graph-eligible paper-inspired candidate.
- The `graph` and `graph_ssm` Tune profiles include `mp_ssm` as a search option.

## Graph-Mamba

Reference: [Graph-Mamba](references.md#papers).

Useful code reference: [bowang-lab/Graph-Mamba](https://github.com/bowang-lab/Graph-Mamba)

Implementation-relevant ideas:

- Node ordering and selection can matter when adapting sequence models to graphs.
- Ordering ideas are useful for future social/entity scans, but the full GraphGPS-style stack is outside the current first implementation pass.

Project mapping:

- Current social SSM ordering is egocentric and distance-based.
- We keep full Graph-Mamba-style architecture work as a later option if graph SSM candidates outperform simpler graph baselines.

## Graph Transformer Baseline

Implementation-relevant ideas:

- Use graph structure as the actual spatial attention mask, not as a post-hoc feature.
- Add edge-conditioned attention bias so relative geometry and semantic relations can affect attention weights.
- Keep matched no-graph and shuffled-graph ablations so the graph contribution must show up empirically.

Project mapping:

- `e5_graph_transformer` uses graph-masked and edge-biased spatial attention in both encoder and decoder.
- `e5_graph_transformer_no_graph` and `e5_graph_transformer_shuffled_graph` are required ablations before treating the model as reportable.
