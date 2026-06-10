from __future__ import annotations

import pytest
import torch

from nba_forecasting.data import Normalizer
from nba_forecasting.models import resolve_training_loss_policy, validate_loss_config
from nba_forecasting.models.builders import TrainingLossPolicy
from nba_forecasting.models.common import ForecastOutput
from nba_forecasting.runner.recipe import ForecastTrainingRecipe
from tests.conftest import make_config


def test_baseline_uses_point_mse_policy() -> None:
    config = make_config("baseline", use_velocity=False)
    policy = resolve_training_loss_policy(config)

    assert policy.primary == "point_mse"
    assert policy.mixture_point_estimate is None
    validate_loss_config(config)


def test_graph_transformer_delta_uses_point_mse_policy() -> None:
    config = make_config("graph_transformer", use_velocity=True)
    policy = resolve_training_loss_policy(config)

    assert policy.primary == "point_mse"
    assert policy.mixture_point_estimate is None


def test_graph_transformer_mdn_uses_mixture_nll_policy() -> None:
    config = make_config("graph_transformer", use_velocity=True)
    config.model.head.head_type = "mdn"
    config.model.head.mdn_components = 4
    config.model.head.mdn_point_estimate = "medoid"
    policy = resolve_training_loss_policy(config)

    assert policy.primary == "mixture_nll"
    assert policy.mixture_point_estimate == "medoid"


def test_mart_uses_sample_mode_policy_with_optional_distribution() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    policy = resolve_training_loss_policy(config)

    assert policy.primary == "sample_modes"
    assert policy.mixture_point_estimate is None

    config.model.mart.mart_mode_distribution = "gaussian"
    gaussian_policy = resolve_training_loss_policy(config)

    assert gaussian_policy.primary == "sample_modes"


def test_mart_can_opt_into_primary_mixture_nll_via_loss_config() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.model.mart.mart_mode_distribution = "gaussian"
    config.loss.sample_loss_type = "mixture_nll"
    policy = resolve_training_loss_policy(config)
    assert policy.primary == "mixture_nll"


def test_validate_loss_config_rejects_incompatible_primary(monkeypatch) -> None:
    config = make_config("baseline", use_velocity=False)
    monkeypatch.setattr(
        "nba_forecasting.models.builders.resolve_training_loss_policy",
        lambda _: TrainingLossPolicy(primary="sample_modes"),
    )

    with pytest.raises(ValueError, match="does not support primary loss"):
        validate_loss_config(config)


def test_physics_penalties_apply_to_sample_loss_path() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.loss.lambda_smooth = 0.1
    recipe = ForecastTrainingRecipe(
        loss_config=config.loss,
        normalizer=Normalizer(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([1.0, 1.0])),
        device=torch.device("cpu"),
        grad_clip=0.0,
        mixed_precision=False,
    )
    target = torch.zeros(1, 3, 1, 2)
    modes = torch.tensor(
        [
            [
                [[0.0, 0.0]],
                [[2.0, 1.0]],
                [[1.0, 3.0]],
            ]
        ],
        requires_grad=True,
    )
    output = ForecastOutput(point=modes.squeeze(3), modes=modes)
    batch = {"static": torch.tensor([[[1.0, 1.0]]])}

    loss, components = recipe.compute_sample_loss(output, target, batch)

    assert components["smooth"] > 0.0
    assert torch.isfinite(loss)
