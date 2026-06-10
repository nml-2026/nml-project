from __future__ import annotations

import math

import torch

from nba_forecasting.models import build_model
from nba_forecasting.models.graph_transformer import GraphMultiheadAttention
from tests.conftest import make_config, make_context
from tests.helpers import HasGraphInputs


def test_graph_models_return_expected_shapes() -> None:
    for (
        model_name,
        use_velocity,
        dynamic_edges,
        use_semantic_flags,
        decoder_type,
        use_relation_types,
        use_entity_embeddings,
    ) in [
        ("static_graph", True, False, True, "direct", False, False),
        ("dynamic_graph_autoregressive", True, True, True, "autoregressive", False, False),
        ("dynamic_graph_direct", True, True, True, "direct", True, True),
        ("graph_transformer", True, True, True, "direct", False, False),
    ]:
        config = make_config(
            model_name=model_name,
            use_velocity=use_velocity,
            dynamic_edges=dynamic_edges,
            use_semantic_flags=use_semantic_flags,
            decoder_type=decoder_type,
            use_relation_types=use_relation_types,
            use_entity_embeddings=use_entity_embeddings,
        )
        if model_name == "graph_transformer":
            config.model.common.hidden_dim = 64
            config.model.graph_transformer.transformer_heads = 4
            config.model.graph_transformer.spatial_layers = 1
            config.model.graph_transformer.temporal_layers = 1
            config.model.graph_transformer.decoder_layers = 1
        model = build_model(config)
        context = make_context(batch_size=2, use_velocity=use_velocity)
        future = torch.randn(2, 12, 11, 2)
        output = model(
            context,
            future_xy=future,
            teacher_forcing_prob=0.5,
            scheduled_sampling_prob=0.25,
        )
        assert tuple(output.shape) == (2, 12, 11, 2)
        assert torch.isfinite(output).all()


def test_graph_attention_masks_and_edge_biases() -> None:
    torch.manual_seed(5)
    attention = GraphMultiheadAttention(hidden_dim=8, edge_dim=3, num_heads=2, dropout=0.0).eval()
    node_state = torch.randn(1, 3, 8)
    edge_feat = torch.randn(1, 3, 3, 3)
    mask = torch.zeros(1, 3, 3, dtype=torch.bool)
    mask[:, :, 1] = True

    with torch.no_grad():
        baseline = attention(node_state, edge_feat, mask)
        changed = node_state.clone()
        changed[:, 2] = changed[:, 2] + 100.0
        changed_edges = edge_feat.clone()
        changed_edges[:, :, 2] = changed_edges[:, :, 2] + 100.0
        masked_changed = attention(changed, changed_edges, mask)

    assert tuple(baseline.shape) == (1, 3, 8)
    assert torch.allclose(baseline[:, 0], masked_changed[:, 0], atol=1e-6)

    edge_a = torch.zeros_like(edge_feat)
    edge_b = torch.zeros_like(edge_feat)
    edge_b[:, :, 1] = 1.0
    dense_mask = torch.ones(1, 3, 3, dtype=torch.bool)
    dense_mask[:, torch.arange(3), torch.arange(3)] = False
    with torch.no_grad():
        biased_a = attention(node_state, edge_a, dense_mask)
        biased_b = attention(node_state, edge_b, dense_mask)
    assert not torch.allclose(biased_a, biased_b)


def test_graph_attention_matches_torch_multihead_attention_without_edge_bias() -> None:
    torch.manual_seed(11)
    attention = GraphMultiheadAttention(hidden_dim=8, edge_dim=3, num_heads=2, dropout=0.0, use_edge_bias=False).eval()
    reference = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.0, batch_first=True).eval()
    with torch.no_grad():
        reference.in_proj_weight.copy_(torch.cat([attention.query.weight, attention.key.weight, attention.value.weight], dim=0))
        reference.in_proj_bias.copy_(torch.cat([attention.query.bias, attention.key.bias, attention.value.bias], dim=0))
        reference.out_proj.weight.copy_(attention.output.weight)
        reference.out_proj.bias.copy_(attention.output.bias)

    node_state = torch.randn(2, 4, 8)
    edge_feat = torch.randn(2, 4, 4, 3)
    mask = torch.tensor(
        [
            [
                [False, True, False, True],
                [True, False, True, False],
                [False, True, False, True],
                [True, False, True, False],
            ],
            [
                [False, True, True, False],
                [True, False, False, True],
                [True, False, False, True],
                [False, True, True, False],
            ],
        ]
    )

    with torch.no_grad():
        output = attention(node_state, edge_feat, mask)
        reference_mask = (~mask).repeat_interleave(attention.num_heads, dim=0)
        expected, _ = reference(node_state, node_state, node_state, attn_mask=reference_mask, need_weights=False)

    assert torch.allclose(output, expected, atol=1e-6)


def test_graph_attention_matches_explicit_edge_biased_formula() -> None:
    torch.manual_seed(13)
    attention = GraphMultiheadAttention(hidden_dim=8, edge_dim=3, num_heads=2, dropout=0.0).eval()
    node_state = torch.randn(2, 4, 8)
    edge_feat = torch.randn(2, 4, 4, 3)
    mask = torch.tensor(
        [
            [
                [False, True, False, True],
                [True, False, True, False],
                [False, True, False, True],
                [True, False, True, False],
            ],
            [
                [False, True, True, False],
                [True, False, False, True],
                [True, False, False, True],
                [False, True, True, False],
            ],
        ]
    )

    with torch.no_grad():
        output = attention(node_state, edge_feat, mask)
        batch_size, num_entities, _ = node_state.shape
        query = attention.query(node_state).reshape(batch_size, num_entities, attention.num_heads, attention.head_dim).transpose(1, 2)
        key = attention.key(node_state).reshape(batch_size, num_entities, attention.num_heads, attention.head_dim).transpose(1, 2)
        value = attention.value(node_state).reshape(batch_size, num_entities, attention.num_heads, attention.head_dim).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(attention.head_dim)
        logits = logits + attention.edge_bias(edge_feat).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(logits, dim=-1)
        expected = torch.matmul(weights, value).transpose(1, 2).reshape(batch_size, num_entities, attention.hidden_dim)
        expected = attention.output(expected)

    assert torch.allclose(output, expected, atol=1e-6)


def test_graph_transformer_ablation_masks_are_distinct() -> None:
    positions = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
                [10.0, 0.0],
            ]
        ]
    )
    static = torch.ones(1, 4, 2)
    masks = {}
    for mode in ["normal", "no_graph", "shuffled_graph"]:
        config = make_config("graph_transformer", use_velocity=True, use_semantic_flags=False)
        config.experiment.num_entities = 4
        config.model.common.hidden_dim = 16
        config.model.graph_transformer.transformer_heads = 4
        config.model.graph_transformer.graph_ablation_mode = mode
        config.model.graph.k_neighbors = 1
        model = build_model(config)
        assert isinstance(model, HasGraphInputs)
        _, masks[mode] = model.graph_inputs(positions, static)

    assert int(masks["normal"].sum()) < int(masks["no_graph"].sum())
    assert not torch.equal(masks["normal"], masks["shuffled_graph"])
    assert tuple(masks["normal"].shape) == tuple(masks["shuffled_graph"].shape)


def test_graph_transformer_is_permutation_equivariant_without_entity_embeddings() -> None:
    config = make_config(
        "graph_transformer",
        use_velocity=True,
        dynamic_edges=False,
        use_semantic_flags=True,
        use_entity_embeddings=False,
    )
    config.model.common.hidden_dim = 32
    config.model.graph_transformer.transformer_heads = 4
    config.model.graph_transformer.spatial_layers = 1
    config.model.graph_transformer.temporal_layers = 1
    config.model.graph_transformer.decoder_layers = 1
    config.model.common.dropout = 0.0
    model = build_model(config).eval()
    context = make_context(batch_size=2, use_velocity=True)
    permutation = torch.tensor([2, 0, 1, 3, 5, 4, 6, 8, 7, 10, 9])

    with torch.no_grad():
        output = model(context)
        permuted_output = model(context[:, :, permutation])

    assert torch.allclose(permuted_output, output[:, :, permutation], atol=1e-5)


def test_graph_transformer_virtual_nodes_preserve_shape_and_equivariance() -> None:
    config = make_config(
        "graph_transformer",
        use_velocity=True,
        dynamic_edges=False,
        use_semantic_flags=True,
        use_entity_embeddings=False,
    )
    config.model.common.hidden_dim = 32
    config.model.common.dropout = 0.0
    config.model.graph_transformer.transformer_heads = 4
    config.model.graph_transformer.spatial_layers = 1
    config.model.graph_transformer.temporal_layers = 1
    config.model.graph_transformer.decoder_layers = 1
    config.model.graph_transformer.num_virtual_nodes = 2
    model = build_model(config).eval()
    context = make_context(batch_size=2, use_velocity=True)
    permutation = torch.tensor([2, 0, 1, 3, 5, 4, 6, 8, 7, 10, 9])

    with torch.no_grad():
        output = model(context)
        permuted_output = model(context[:, :, permutation])

    assert tuple(output.shape) == (2, 12, 11, 2)
    assert torch.allclose(permuted_output, output[:, :, permutation], atol=1e-5)


def test_graph_transformer_forward_backward() -> None:
    config = make_config("graph_transformer", use_velocity=True, use_semantic_flags=True, use_entity_embeddings=True)
    config.model.common.hidden_dim = 32
    config.model.graph_transformer.transformer_heads = 4
    config.model.graph_transformer.spatial_layers = 1
    config.model.graph_transformer.temporal_layers = 1
    config.model.graph_transformer.decoder_layers = 1
    model = build_model(config)
    context = make_context(batch_size=2, use_velocity=True)
    future = torch.randn(2, 12, 11, 2)
    prediction = model(context)
    from nba_forecasting.metrics import normalized_mse_loss

    loss = normalized_mse_loss(prediction, future)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
