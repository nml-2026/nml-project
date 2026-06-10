from __future__ import annotations

import json
from pathlib import Path

import torch

from nba_forecasting.data import (
    ContextNoiseTransform,
    CourtReflectionTransform,
    Normalizer,
    SequenceWindowDataset,
    build_training_pipeline,
    collect_sequence_ids,
    collect_sequence_paths,
    compute_normalizer,
    freeze_kfold_sequence_splits,
    freeze_sequence_split,
    prepare_remote_dataset,
    reflect_context,
    reflect_prediction_xy,
    reflect_shifts,
)
from tests.conftest import make_config
from tests.helpers import write_synthetic_dataset


def test_freeze_split_is_deterministic_and_disjoint(tmp_path: Path) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24, 21, 23, 25, 26, 27, 28, 29])
    split_path = tmp_path / "artifacts" / "splits" / "default.json"
    first = freeze_sequence_split(data_dir / "train", split_path, seed=7)
    second = freeze_sequence_split(data_dir / "train", split_path, seed=999)
    assert first == second
    total = len(first["train"]) + len(first["val"]) + len(first["holdout"])
    assert total == 10
    assert not set(first["train"]).intersection(first["val"])
    assert not set(first["train"]).intersection(first["holdout"])
    assert not set(first["val"]).intersection(first["holdout"])


def test_prepare_remote_dataset_consolidates_train_test_and_split(tmp_path: Path) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24, 21, 23, 25, 26, 27, 28, 29])
    split_path = tmp_path / "artifacts" / "splits" / "default.json"
    summary = prepare_remote_dataset(data_dir, split_path, seed=7)
    train_consolidated = data_dir / "train_consolidated.pt"
    test_consolidated = data_dir / "test_consolidated.pt"
    assert summary["train_consolidated"] == str(train_consolidated.resolve())
    assert summary["test_consolidated"] == str(test_consolidated.resolve())
    assert train_consolidated.exists()
    assert test_consolidated.exists()
    assert split_path.exists()

    train_dir = data_dir / "train"
    for path in train_dir.glob("*.pt"):
        path.unlink()
    train_dir.rmdir()
    assert len(collect_sequence_ids(train_dir)) == 10
    split = freeze_sequence_split(train_dir, split_path, seed=999)
    assert len(split["train"]) + len(split["val"]) + len(split["holdout"]) == 10


def test_freeze_kfold_splits_cover_pool_and_preserve_holdout(tmp_path: Path) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24, 21, 23, 25, 26, 27, 28, 29])
    base_split_path = tmp_path / "artifacts" / "splits" / "default.json"
    base_split = freeze_sequence_split(data_dir / "train", base_split_path, seed=7)
    fold_paths = freeze_kfold_sequence_splits(
        train_dir=data_dir / "train",
        output_dir=tmp_path / "artifacts" / "cv" / "k3",
        num_folds=3,
        seed=7,
        base_split_path=base_split_path,
        include_holdout=False,
    )
    assert len(fold_paths) == 3
    seen_val: set[int] = set()
    for fold_path in fold_paths:
        split = json.loads(fold_path.read_text())
        assert split["holdout"] == base_split["holdout"]
        seen_val.update(split["val"])
        assert not set(split["train"]).intersection(split["val"])
    assert seen_val == set(base_split["train"] + base_split["val"])


def test_window_dataset_uses_all_valid_windows(synthetic_dataset_path: Path) -> None:
    paths = collect_sequence_paths(synthetic_dataset_path / "train")
    normalizer = compute_normalizer(paths.values())
    dataset = SequenceWindowDataset(
        sequence_paths=[paths[0], paths[1], paths[2]],
        normalizer=normalizer,
        context_size=8,
        horizon_size=12,
        use_velocity=True,
    )
    expected_windows = (20 - 20 + 1) + (22 - 20 + 1) + (24 - 20 + 1)
    assert len(dataset) == expected_windows
    sample = dataset[1]
    assert tuple(sample["context"].shape) == (8, 11, 6)
    assert tuple(sample["future_xy"].shape) == (12, 11, 2)
    assert tuple(sample["future_xy_raw"].shape) == (12, 11, 2)
    assert tuple(sample["static"].shape) == (11, 2)


def _build_reflect_pair(data_dir: Path) -> tuple[SequenceWindowDataset, SequenceWindowDataset, Normalizer]:
    paths = collect_sequence_paths(data_dir / "train")
    normalizer = compute_normalizer(paths.values())
    sequence_paths = [paths[0], paths[1], paths[2]]
    baseline = SequenceWindowDataset(
        sequence_paths=sequence_paths,
        normalizer=normalizer,
        context_size=8,
        horizon_size=12,
        use_velocity=True,
        reflect_prob=0.0,
    )
    mirrored = SequenceWindowDataset(
        sequence_paths=sequence_paths,
        normalizer=normalizer,
        context_size=8,
        horizon_size=12,
        use_velocity=True,
        reflect_prob=1.0,
    )
    return baseline, mirrored, normalizer


def test_reflect_prob_zero_is_identity(synthetic_dataset_path: Path) -> None:
    baseline, _, _ = _build_reflect_pair(synthetic_dataset_path)
    paths = collect_sequence_paths(synthetic_dataset_path / "train")
    normalizer = compute_normalizer(paths.values())
    plain = SequenceWindowDataset(
        sequence_paths=[paths[0], paths[1], paths[2]],
        normalizer=normalizer,
        context_size=8,
        horizon_size=12,
        use_velocity=True,
    )
    for idx in range(len(baseline)):
        a, b = baseline[idx], plain[idx]
        for key in ("context", "future_xy", "future_xy_raw", "context_raw_xy", "static"):
            assert torch.equal(a[key], b[key]), f"reflect_prob=0 changed {key}"


def test_agent_permutation_shuffles_players_and_keeps_ball_fixed() -> None:
    normalizer = Normalizer(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([1.0, 1.0]))
    time = torch.arange(3, dtype=torch.float32).view(3, 1, 1)
    entity = torch.arange(11, dtype=torch.float32).view(1, 11, 1)
    x = entity + 0.1 * time
    y = -entity + 0.2 * time
    is_player = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float32).view(1, 11, 1).expand(3, -1, -1)
    team = torch.tensor([0, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1], dtype=torch.float32).view(1, 11, 1).expand(3, -1, -1)
    seq = torch.cat([x, y, is_player, team], dim=-1)
    base = SequenceWindowDataset(
        sequence_paths={0: seq},
        normalizer=normalizer,
        context_size=2,
        horizon_size=1,
        use_velocity=True,
        seed=4,
    )[0]
    shuffled = SequenceWindowDataset(
        sequence_paths={0: seq},
        normalizer=normalizer,
        context_size=2,
        horizon_size=1,
        use_velocity=True,
        entity_permutation_prob=1.0,
        seed=4,
    )[0]

    torch.testing.assert_close(shuffled["context"][:, 0], base["context"][:, 0])
    torch.testing.assert_close(shuffled["future_xy"][:, 0], base["future_xy"][:, 0])
    torch.testing.assert_close(shuffled["static"][0], base["static"][0])
    assert not torch.equal(shuffled["context"][:, 1:, :2], base["context"][:, 1:, :2])
    assert sorted(shuffled["context"][0, 1:, 0].tolist()) == sorted(base["context"][0, 1:, 0].tolist())
    assert sorted(shuffled["static"][1:, 1].tolist()) == sorted(base["static"][1:, 1].tolist())


def test_reflect_prob_one_mirrors_y_and_preserves_x_and_team(synthetic_dataset_path: Path) -> None:
    baseline, mirrored, _ = _build_reflect_pair(synthetic_dataset_path)
    for idx in range(len(baseline)):
        base, flip = baseline[idx], mirrored[idx]
        assert torch.allclose(flip["context_raw_xy"][..., 1], -base["context_raw_xy"][..., 1])
        assert torch.allclose(flip["future_xy_raw"][..., 1], -base["future_xy_raw"][..., 1])
        assert torch.equal(flip["context_raw_xy"][..., 0], base["context_raw_xy"][..., 0])
        assert torch.equal(flip["future_xy_raw"][..., 0], base["future_xy_raw"][..., 0])
        assert torch.allclose(flip["context"][..., 3], -base["context"][..., 3])
        assert torch.equal(flip["context"][..., 2], base["context"][..., 2])
        assert torch.equal(flip["static"], base["static"])


def test_reflect_normalized_and_raw_stay_consistent(synthetic_dataset_path: Path) -> None:
    _, mirrored, normalizer = _build_reflect_pair(synthetic_dataset_path)
    for idx in range(len(mirrored)):
        flip = mirrored[idx]
        recon = normalizer.denormalize_xy(flip["context"][..., :2])
        assert torch.allclose(recon, flip["context_raw_xy"], atol=1e-4)


def _build_dataset(data_dir: Path, reflect_prob: float, reflect_x_prob: float) -> SequenceWindowDataset:
    paths = collect_sequence_paths(data_dir / "train")
    normalizer = compute_normalizer(paths.values())
    return SequenceWindowDataset(
        sequence_paths=[paths[0], paths[1], paths[2]],
        normalizer=normalizer,
        context_size=8,
        horizon_size=12,
        use_velocity=True,
        reflect_prob=reflect_prob,
        reflect_x_prob=reflect_x_prob,
    )


def test_reflect_x_prob_one_mirrors_x_and_preserves_y_and_team(synthetic_dataset_path: Path) -> None:
    base = _build_dataset(synthetic_dataset_path, 0.0, 0.0)
    flip_x = _build_dataset(synthetic_dataset_path, 0.0, 1.0)
    for idx in range(len(base)):
        b, f = base[idx], flip_x[idx]
        assert torch.allclose(f["context_raw_xy"][..., 0], -b["context_raw_xy"][..., 0])
        assert torch.allclose(f["future_xy_raw"][..., 0], -b["future_xy_raw"][..., 0])
        assert torch.equal(f["context_raw_xy"][..., 1], b["context_raw_xy"][..., 1])
        assert torch.allclose(f["context"][..., 2], -b["context"][..., 2])
        assert torch.equal(f["context"][..., 3], b["context"][..., 3])
        assert torch.equal(f["static"], b["static"])


def test_reflect_xy_mirrors_both_axes(synthetic_dataset_path: Path) -> None:
    base = _build_dataset(synthetic_dataset_path, 0.0, 0.0)
    normalizer = compute_normalizer(collect_sequence_paths(synthetic_dataset_path / "train").values())
    flip_xy = _build_dataset(synthetic_dataset_path, 1.0, 1.0)
    for idx in range(len(base)):
        b, f = base[idx], flip_xy[idx]
        assert torch.allclose(f["future_xy_raw"], -b["future_xy_raw"])
        assert torch.allclose(f["context_raw_xy"], -b["context_raw_xy"])
        assert torch.equal(f["static"], b["static"])
        recon = normalizer.denormalize_xy(f["context"][..., :2])
        assert torch.allclose(recon, f["context_raw_xy"], atol=1e-4)


def test_reflect_context_matches_training_path(synthetic_dataset_path: Path) -> None:
    normalizer = compute_normalizer(collect_sequence_paths(synthetic_dataset_path / "train").values())
    shift_x, shift_y = reflect_shifts(normalizer)
    base = _build_dataset(synthetic_dataset_path, 0.0, 0.0)
    cases = {
        "x": (_build_dataset(synthetic_dataset_path, 0.0, 1.0), True, False),
        "y": (_build_dataset(synthetic_dataset_path, 1.0, 0.0), False, True),
        "xy": (_build_dataset(synthetic_dataset_path, 1.0, 1.0), True, True),
    }
    for _, (mirrored, flip_x, flip_y) in cases.items():
        for idx in range(len(base)):
            expected = mirrored[idx]["context"]
            got = reflect_context(base[idx]["context"], True, shift_x, shift_y, flip_x, flip_y)
            assert torch.allclose(got, expected, atol=1e-5)


def test_build_training_pipeline_composes_augmentation_and_reflection() -> None:
    config = make_config("baseline", use_velocity=True)
    config.features.augmentation.context_noise_std_ft = 0.1
    config.features.reflection.mode = "random"
    config.features.reflection.y_prob = 0.5
    config.features.reflection.x_prob = 0.25
    normalizer = Normalizer(mu=torch.zeros(2), sigma=torch.ones(2))

    pipeline = build_training_pipeline(config, normalizer)

    assert len(pipeline.transforms) == 2
    assert isinstance(pipeline.transforms[0], ContextNoiseTransform)
    assert isinstance(pipeline.transforms[1], CourtReflectionTransform)


def test_reflect_prediction_xy_is_involution(synthetic_dataset_path: Path) -> None:
    normalizer = compute_normalizer(collect_sequence_paths(synthetic_dataset_path / "train").values())
    shift_x, shift_y = reflect_shifts(normalizer)
    prediction = torch.randn(2, 12, 11, 2)
    identity = reflect_prediction_xy(prediction, shift_x, shift_y, False, False)
    assert torch.equal(identity, prediction)
    once = reflect_prediction_xy(prediction, shift_x, shift_y, True, True)
    twice = reflect_prediction_xy(once, shift_x, shift_y, True, True)
    assert torch.allclose(twice, prediction, atol=1e-5)
