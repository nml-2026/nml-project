from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from nba_forecasting.data import (
    SequenceWindowDataset,
    collect_sequence_ids,
    collect_sequence_paths,
    compute_normalizer,
    freeze_kfold_sequence_splits,
    freeze_sequence_split,
    prepare_remote_dataset,
)
from nba_forecasting.metrics import compute_raw_metrics, normalized_mse_loss
from tests.helpers import write_synthetic_dataset


class DataAndMetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data_dir = write_synthetic_dataset(self.root, train_lengths=[20, 22, 24, 21, 23, 25, 26, 27, 28, 29])

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_freeze_split_is_deterministic_and_disjoint(self) -> None:
        split_path = self.root / "artifacts" / "splits" / "default.json"
        first = freeze_sequence_split(self.data_dir / "train", split_path, seed=7)
        second = freeze_sequence_split(self.data_dir / "train", split_path, seed=999)
        self.assertEqual(first, second)
        total = len(first["train"]) + len(first["val"]) + len(first["holdout"])
        self.assertEqual(total, 10)
        self.assertFalse(set(first["train"]).intersection(first["val"]))
        self.assertFalse(set(first["train"]).intersection(first["holdout"]))
        self.assertFalse(set(first["val"]).intersection(first["holdout"]))

    def test_prepare_remote_dataset_consolidates_train_test_and_split(self) -> None:
        split_path = self.root / "artifacts" / "splits" / "default.json"
        summary = prepare_remote_dataset(self.data_dir, split_path, seed=7)
        train_consolidated = self.data_dir / "train_consolidated.pt"
        test_consolidated = self.data_dir / "test_consolidated.pt"
        self.assertEqual(summary["train_consolidated"], str(train_consolidated.resolve()))
        self.assertEqual(summary["test_consolidated"], str(test_consolidated.resolve()))
        self.assertTrue(train_consolidated.exists())
        self.assertTrue(test_consolidated.exists())
        self.assertTrue(split_path.exists())

        train_dir = self.data_dir / "train"
        for path in train_dir.glob("*.pt"):
            path.unlink()
        train_dir.rmdir()
        self.assertEqual(len(collect_sequence_ids(train_dir)), 10)
        split = freeze_sequence_split(train_dir, split_path, seed=999)
        self.assertEqual(len(split["train"]) + len(split["val"]) + len(split["holdout"]), 10)

    def test_freeze_kfold_splits_cover_pool_and_preserve_holdout(self) -> None:
        base_split_path = self.root / "artifacts" / "splits" / "default.json"
        base_split = freeze_sequence_split(self.data_dir / "train", base_split_path, seed=7)
        fold_paths = freeze_kfold_sequence_splits(
            train_dir=self.data_dir / "train",
            output_dir=self.root / "artifacts" / "cv" / "k3",
            num_folds=3,
            seed=7,
            base_split_path=base_split_path,
            include_holdout=False,
        )
        self.assertEqual(len(fold_paths), 3)
        seen_val: set[int] = set()
        for fold_path in fold_paths:
            split = json.loads(fold_path.read_text())
            self.assertEqual(split["holdout"], base_split["holdout"])
            seen_val.update(split["val"])
            self.assertFalse(set(split["train"]).intersection(split["val"]))
        self.assertEqual(seen_val, set(base_split["train"] + base_split["val"]))

    def test_window_dataset_uses_all_valid_windows(self) -> None:
        paths = collect_sequence_paths(self.data_dir / "train")
        normalizer = compute_normalizer(paths.values())
        dataset = SequenceWindowDataset(
            sequence_paths=[paths[0], paths[1], paths[2]],
            normalizer=normalizer,
            context_size=8,
            horizon_size=12,
            use_velocity=True,
        )
        expected_windows = (20 - 20 + 1) + (22 - 20 + 1) + (24 - 20 + 1)
        self.assertEqual(len(dataset), expected_windows)
        sample = dataset[1]
        self.assertEqual(tuple(sample["context"].shape), (8, 11, 6))
        self.assertEqual(tuple(sample["future_xy"].shape), (12, 11, 2))
        self.assertEqual(tuple(sample["future_xy_raw"].shape), (12, 11, 2))
        self.assertEqual(tuple(sample["static"].shape), (11, 2))

    def test_metrics_match_expected_values(self) -> None:
        prediction = torch.tensor([[[[1.0, 2.0]]]])
        target = torch.zeros_like(prediction)
        metrics = compute_raw_metrics(prediction, target)
        self.assertAlmostEqual(metrics["mse"], 2.5, places=6)
        self.assertAlmostEqual(metrics["ade"], 5.0, places=6)
        self.assertAlmostEqual(metrics["rmse"], 2.5**0.5, places=6)
        self.assertAlmostEqual(float(normalized_mse_loss(prediction, target).item()), 2.5, places=6)


if __name__ == "__main__":
    unittest.main()
