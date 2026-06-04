from __future__ import annotations

import unittest
from types import MethodType

import torch

from nba_forecasting.bench import GRAPH_TRANSFORMER_COMBINATORIAL_VARIANTS, GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS
from nba_forecasting.config import apply_overrides, load_config
from nba_forecasting.models import build_model


class GraphTransformerOptimizationTests(unittest.TestCase):
    def test_combinatorial_registry_includes_model_independent_optimizations(self) -> None:
        self.assertEqual(len(GRAPH_TRANSFORMER_COMBINATORIAL_VARIANTS), 16)
        self.assertEqual(
            GRAPH_TRANSFORMER_COMBINATORIAL_VARIANTS["compile__fused_adamw"],
            {
                "model.optimization_backend": "compile",
                "model.optimizer_backend": "adamw_fused",
            },
        )
        self.assertEqual(
            GRAPH_TRANSFORMER_COMBINATORIAL_VARIANTS["fused_graph_prep__graph_cache__compile__fused_adamw"],
            {
                "model.graph_prep_backend": "fused",
                "model.graph_cache_backend": "reuse",
                "model.optimization_backend": "compile",
                "model.optimizer_backend": "adamw_fused",
            },
        )
        self.assertEqual(
            GRAPH_TRANSFORMER_COMBINATORIAL_VARIANTS["fused_graph_prep"],
            {
                "model.graph_prep_backend": "fused",
                "model.optimization_backend": "none",
                "model.optimizer_backend": "adamw",
            },
        )
        self.assertEqual(
            GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS["compile_fused_graph_prep"],
            {
                "model.graph_prep_backend": "fused",
                "model.optimization_backend": "compile",
                "model.optimizer_backend": "adamw",
            },
        )
        self.assertEqual(
            GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS["compile_fused_graph_prep_graph_cache"],
            {
                "model.graph_prep_backend": "fused",
                "model.graph_cache_backend": "reuse",
                "model.optimization_backend": "compile",
                "model.optimizer_backend": "adamw",
            },
        )

    def test_fused_graph_prep_matches_torch_backend(self) -> None:
        config, _ = load_config("e5_graph_transformer")
        baseline = build_model(apply_overrides(config, {"model.graph_prep_backend": "torch"}))
        fused = build_model(apply_overrides(config, {"model.graph_prep_backend": "fused"}))
        positions = torch.randn(6, config.experiment.num_entities, 2)
        static = torch.zeros(6, config.experiment.num_entities, 2)
        static[:, 0, 0] = 0
        static[:, 1:, 0] = 1
        static[:, 1:, 1] = torch.where(torch.arange(config.experiment.num_entities - 1) % 2 == 0, 1, -1)

        edge_ref, mask_ref = baseline.graph_inputs(positions, static)
        edge_fused, mask_fused = fused.graph_inputs(positions, static)

        self.assertTrue(torch.equal(mask_fused, mask_ref))
        torch.testing.assert_close(edge_fused, edge_ref)

    def test_graph_cache_matches_baseline_forward_and_gradients(self) -> None:
        torch.manual_seed(0)
        config, _ = load_config("e5_graph_transformer")
        overrides = {
            "model.dropout": 0.0,
            "model.graph_cache_backend": "baseline",
            "model.graph_prep_backend": "torch",
        }
        baseline = build_model(apply_overrides(config, overrides))
        cached = build_model(apply_overrides(config, {**overrides, "model.graph_cache_backend": "reuse"}))
        cached.load_state_dict(baseline.state_dict())
        context = torch.randn(2, config.experiment.context_size, config.experiment.num_entities, config.input_dim)
        context[..., -2:-1] = 1.0
        context[:, :, 0, -2] = 0.0
        context[..., -1:] = torch.where(
            torch.arange(config.experiment.num_entities).view(1, 1, -1, 1) % 2 == 0,
            torch.tensor(1.0),
            torch.tensor(-1.0),
        )

        baseline.train()
        cached.train()
        baseline_out = baseline(context)
        cached_out = cached(context)
        torch.testing.assert_close(cached_out, baseline_out, rtol=1e-5, atol=1e-6)

        baseline_out.square().mean().backward()
        cached_out.square().mean().backward()
        for name in [
            "node_proj.weight",
            "spatial_layers.0.attention.query.weight",
            "decoder_layers.0.attention.edge_bias.weight",
            "decoder_input.weight",
        ]:
            torch.testing.assert_close(cached.get_parameter(name).grad, baseline.get_parameter(name).grad, rtol=1e-5, atol=1e-6)

    def test_graph_cache_reduces_forward_graph_input_calls(self) -> None:
        config, _ = load_config("e5_graph_transformer")
        context = torch.randn(1, config.experiment.context_size, config.experiment.num_entities, config.input_dim)
        context[..., -2:-1] = 1.0
        context[:, :, 0, -2] = 0.0
        context[..., -1:] = torch.where(
            torch.arange(config.experiment.num_entities).view(1, 1, -1, 1) % 2 == 0,
            torch.tensor(1.0),
            torch.tensor(-1.0),
        )

        counts: dict[str, int] = {}

        def count_graph_inputs(model, key: str) -> None:
            original = model.graph_inputs
            counts[key] = 0

            def wrapped(self, positions, static):
                del self
                counts[key] += 1
                return original(positions, static)

            model.graph_inputs = MethodType(wrapped, model)

        baseline = build_model(apply_overrides(config, {"model.graph_cache_backend": "baseline"}))
        cached = build_model(apply_overrides(config, {"model.graph_cache_backend": "reuse"}))
        count_graph_inputs(baseline, "baseline")
        count_graph_inputs(cached, "cached")

        with torch.no_grad():
            baseline(context)
            cached(context)

        self.assertEqual(counts["baseline"], config.model.spatial_layers + 1)
        self.assertEqual(counts["cached"], 2)


if __name__ == "__main__":
    unittest.main()
