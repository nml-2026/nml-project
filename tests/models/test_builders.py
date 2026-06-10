from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nba_forecasting.config import load_config
from nba_forecasting.models import build_model, collect_search_profiles, get_family_spec, list_model_family_names
from nba_forecasting.models.common import ForecastOutput, forecast_model
from tests.conftest import make_config, make_context


def _runnable_config_paths() -> list[Path]:
    config_dir = Path(__file__).parents[2] / "configs"
    return sorted(path for path in config_dir.glob("*.toml") if path.name != "base.toml")


def test_configs_load_and_build_models() -> None:
    for config_path in _runnable_config_paths():
        config, _ = load_config(config_path)
        model = build_model(config)
        assert isinstance(model, torch.nn.Module)
        use_velocity = config.features.use_velocity and config.model.family not in {"baseline", "joint_gru"}
        output = forecast_model(model, make_context(batch_size=1, use_velocity=use_velocity))
        assert isinstance(output, ForecastOutput)
        assert tuple(output.point.shape) == (
            1,
            config.experiment.horizon_size,
            config.experiment.num_entities,
            2,
        )


def test_build_model_runs_forward_pass() -> None:
    config = make_config("baseline", use_velocity=False)
    model = build_model(config)
    output = forecast_model(model, make_context(batch_size=2, use_velocity=False))
    assert isinstance(output, ForecastOutput)
    assert tuple(output.point.shape) == (2, config.experiment.horizon_size, config.experiment.num_entities, 2)


def test_registered_model_families_build() -> None:
    for family in list_model_family_names():
        config = make_config(family)
        model = get_family_spec(family).builder(config)
        assert isinstance(model, torch.nn.Module)


def test_build_model_unknown_family_raises_key_error() -> None:
    with pytest.raises(KeyError, match="Unknown model.family"):
        build_model(make_config("not_a_real_family"))


def test_collect_search_profiles_exposes_family_bundles() -> None:
    profiles = collect_search_profiles()
    assert set(profiles) >= {"graph", "graph_fixed_message", "graph_transformer", "mart"}
    assert any(spec.startswith("model.common.hidden_dim=") for spec in profiles["graph_transformer"])
    assert any(spec.startswith("model.mart.mart_function_type=") for spec in profiles["mart"])
