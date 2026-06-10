from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from nba_forecasting.models import build_model
from nba_forecasting.runner import submit_checkpoint, submit_ensemble
from nba_forecasting.runner.io import _load_checkpoint
from tests.conftest import make_config
from tests.helpers import write_synthetic_dataset


def test_submit_checkpoint_matches_template_shape(tmp_path: Path) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24], test_count=2)
    config = make_config("baseline", data_dir=str(data_dir), use_velocity=False)
    model = build_model(config)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "split": {"train": [0, 1], "val": [2], "holdout": []},
            "normalizer": {"mu": [0.0, 0.0], "sigma": [1.0, 1.0]},
            "best_metrics": {"ade": 0.0, "mse": 0.0, "rmse": 0.0},
            "epoch": 0,
        },
        checkpoint_path,
    )
    output_path = submit_checkpoint(checkpoint_path, tmp_path / "submission.csv")
    frame = pd.read_csv(output_path)
    assert frame.shape == (2, 265)
    assert frame.columns[0] == "id"
    assert frame["id"].tolist() == [0, 1]


def test_submit_checkpoint_uses_config_default_tta(tmp_path: Path, monkeypatch) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24], test_count=1)
    config = make_config("baseline", data_dir=str(data_dir), use_velocity=False)
    config.evaluation.tta_group = "xy"
    model = build_model(config)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "split": {"train": [0, 1], "val": [2], "holdout": []},
            "normalizer": {"mu": [0.0, 0.0], "sigma": [1.0, 1.0]},
            "best_metrics": {"ade": 0.0, "mse": 0.0, "rmse": 0.0},
            "epoch": 0,
        },
        checkpoint_path,
    )
    seen: list[str] = []

    def fake_tta_predict(model, context, normalizer, tta_group, use_velocity, distribution_point_estimate=None):
        del model, normalizer, use_velocity, distribution_point_estimate
        seen.append(tta_group)
        return torch.zeros(context.shape[0], 12, 11, 2)

    monkeypatch.setattr("nba_forecasting.runner.submission._tta_predict", fake_tta_predict)

    submit_checkpoint(checkpoint_path, tmp_path / "submission.csv")

    assert seen == ["xy"]


def test_load_checkpoint_accepts_compiled_state_dict_prefix(tmp_path: Path) -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.model.common.hidden_dim = 32
    config.model.common.message_dim = 64
    config.model.common.num_layers = 1
    config.model.graph_transformer.transformer_heads = 4
    config.model.mart.mart_decoder_hidden_dim = 64
    config.model.mart.mart_sample_k = 2
    model = build_model(config)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": {f"_orig_mod.{key}": value for key, value in model.state_dict().items()},
            "config": config.to_dict(),
            "split": {"train": [0, 1], "val": [2], "holdout": []},
            "normalizer": {"mu": [0.0, 0.0], "sigma": [1.0, 1.0]},
            "best_metrics": {"ade": 0.0, "mse": 0.0, "rmse": 0.0},
            "epoch": 0,
        },
        checkpoint_path,
    )

    _, _, _, loaded_model = _load_checkpoint(checkpoint_path, torch.device("cpu"))

    assert isinstance(loaded_model, type(model))


def test_submit_ensemble_matches_template_shape(tmp_path: Path) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24], test_count=2)
    config = make_config("baseline", data_dir=str(data_dir), use_velocity=False)
    model = build_model(config)
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "split": {"train": [0, 1], "val": [2], "holdout": []},
            "normalizer": {"mu": [0.0, 0.0], "sigma": [1.0, 1.0]},
            "best_metrics": {"ade": 0.0, "mse": 0.0, "rmse": 0.0},
            "epoch": 0,
        },
        checkpoint_path,
    )
    output_path = submit_ensemble([checkpoint_path], tmp_path / "ensemble_submission.csv")
    frame = pd.read_csv(output_path)
    assert frame.shape == (2, 265)
    assert frame.columns[0] == "id"
    assert frame["id"].tolist() == [0, 1]
