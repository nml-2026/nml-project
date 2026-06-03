from __future__ import annotations

import unittest

import torch

from nba_forecasting.config import apply_overrides, load_config
from nba_forecasting.models import build_model


class GraphTransformerOptimizationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
