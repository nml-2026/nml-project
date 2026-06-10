from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from nba_forecasting.data import Normalizer
from nba_forecasting.models import build_model
from nba_forecasting.runner import evaluate_model
from tests.conftest import make_config, make_context
from tests.helpers import SimpleDataset


def test_evaluate_model_runs_and_aggregates_metrics() -> None:
    context = make_context(batch_size=2, use_velocity=False)
    target = torch.randn(2, 12, 11, 2)
    batch = {
        "context": context,
        "future_xy": target,
        "future_xy_raw": target,
        "context_raw_xy": context[..., :2],
        "sequence_id": torch.tensor([0, 1]),
        "start_idx": torch.tensor([0, 0]),
        "static": torch.zeros(2, 11, 2),
    }
    normalizer = Normalizer(mu=torch.zeros(2), sigma=torch.ones(2))
    dataloader = DataLoader(SimpleDataset([batch]), batch_size=None)

    config = make_config("baseline", use_velocity=False)
    model = build_model(config)

    metrics, examples = evaluate_model(
        model=model,
        dataloader=dataloader,
        normalizer=normalizer,
        device=torch.device("cpu"),
        num_examples=1,
        mixed_precision=False,
        use_velocity=config.features.use_velocity,
    )
    for key in ("loss", "mse", "ade", "rmse", "ade_l2", "fde_l2"):
        assert key in metrics
        assert math.isfinite(metrics[key])

    assert len(examples["best"]) == 1
    assert len(examples["worst"]) == 1
