from __future__ import annotations

import torch

from nba_forecasting.models import build_model
from tests.conftest import make_config, make_context


def test_reference_model_returns_expected_shape() -> None:
    config = make_config(model_name="baseline", use_velocity=False)
    model = build_model(config)
    context = make_context(batch_size=2, use_velocity=False)
    output = model(
        context,
        future_xy=torch.randn(2, 12, 11, 2),
        teacher_forcing_prob=0.5,
        scheduled_sampling_prob=0.25,
    )
    assert tuple(output.shape) == (2, 12, 11, 2)
    assert torch.isfinite(output).all()
