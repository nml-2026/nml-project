from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from nba_forecasting.data import Normalizer
from nba_forecasting.models import build_model
from nba_forecasting.runner import evaluate_checkpoint, evaluate_ensemble, evaluate_model
from tests.conftest import make_config, make_context
from tests.helpers import SimpleDataset, write_synthetic_dataset


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


def _save_tiny_mdn_checkpoint(
    path: Path, data_dir: Path, split: dict[str, list[int]], num_components: int = 3, seed: int = 0
) -> None:
    """Build a minimal MDN-head graph_transformer `best.pt` (untrained) for ensemble minADE_K tests."""
    config = make_config("graph_transformer", data_dir=str(data_dir), decoder_type="direct")
    config.model.common.hidden_dim = 64
    config.model.head.head_type = "mdn"
    config.model.head.mdn_components = num_components
    config.model.head.mdn_point_estimate = "mean"
    torch.manual_seed(seed)
    model = build_model(config)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config.to_dict(),
            "split": split,
            "normalizer": {"mu": [5.0, -3.0], "sigma": [2.0, 4.0]},
            "best_metrics": {"ade": 0.0, "mse": 0.0, "rmse": 0.0},
            "epoch": 0,
        },
        path,
    )


def test_mdn_ensemble_reports_pooled_minade_k(tmp_path: Path) -> None:
    # An all-MDN ensemble reports minADE_K over the pooled K * Nseeds modes.
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24], test_count=2)
    split = {"train": [0, 1], "val": [2], "holdout": []}
    ckpt_a = tmp_path / "mdn_a.pt"
    ckpt_b = tmp_path / "mdn_b.pt"
    _save_tiny_mdn_checkpoint(ckpt_a, data_dir, split, num_components=3, seed=0)
    _save_tiny_mdn_checkpoint(ckpt_b, data_dir, split, num_components=3, seed=1)
    ensemble_dir = evaluate_ensemble(
        [ckpt_a, ckpt_b], "val", tmp_path / "mdn_ens_eval", device_arg="cpu", tta_group="xy"
    )
    metrics = json.loads((ensemble_dir / "metrics.json").read_text())
    for key in ("min_ade_k", "min_fde_k", "mdn_components"):
        assert key in metrics
    # Pooled mode count is K * Nseeds = 3 * 2.
    assert metrics["mdn_components"] == 6.0
    assert metrics["min_ade_k"] > 0.0


def test_pooled_minade_k_identical_legs_equals_single(tmp_path: Path) -> None:
    # Two copies of the same checkpoint pool to duplicate modes, so the oracle min
    # over 2K equals the single-leg min over K -- pooling never raises minADE_K.
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24], test_count=2)
    split = {"train": [0, 1], "val": [2], "holdout": []}
    ckpt = tmp_path / "mdn.pt"
    _save_tiny_mdn_checkpoint(ckpt, data_dir, split, num_components=3, seed=0)
    single = json.loads(
        (evaluate_checkpoint(ckpt, "val", tmp_path / "single", device_arg="cpu") / "metrics.json").read_text()
    )
    duplicated = json.loads(
        (
            evaluate_ensemble([ckpt, ckpt], "val", tmp_path / "dup", device_arg="cpu", tta_group="xy")
            / "metrics.json"
        ).read_text()
    )
    assert single["min_ade_k"] == pytest.approx(duplicated["min_ade_k"], abs=1e-5)
    assert single["min_fde_k"] == pytest.approx(duplicated["min_fde_k"], abs=1e-5)


def test_n1_mdn_ensemble_minade_k_matches_single_checkpoint(tmp_path: Path) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24], test_count=2)
    split = {"train": [0, 1], "val": [2], "holdout": []}
    ckpt = tmp_path / "mdn.pt"
    _save_tiny_mdn_checkpoint(ckpt, data_dir, split, num_components=3, seed=0)
    single = json.loads(
        (evaluate_checkpoint(ckpt, "val", tmp_path / "single", device_arg="cpu") / "metrics.json").read_text()
    )
    ensemble = json.loads(
        (evaluate_ensemble([ckpt], "val", tmp_path / "n1", device_arg="cpu", tta_group="xy") / "metrics.json").read_text()
    )
    assert single["min_ade_k"] == pytest.approx(ensemble["min_ade_k"], abs=1e-5)
    assert single["min_fde_k"] == pytest.approx(ensemble["min_fde_k"], abs=1e-5)
    assert ensemble["mdn_components"] == 3.0
