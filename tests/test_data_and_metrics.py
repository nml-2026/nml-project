from __future__ import annotations

import json
import math
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
    reflect_context,
    reflect_prediction_xy,
    reflect_shifts,
)
from nba_forecasting.metrics import (
    ball_plausibility_loss,
    boundary_loss,
    compute_raw_metrics,
    min_ade_k,
    min_fde_k,
    mixture_nll,
    normalized_mse_loss,
    per_step_l2_error,
    smoothness_loss,
)
from tests.helpers import IS_PLAYER, TEAM_VALUES, write_synthetic_dataset


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

    def _build_reflect_pair(self) -> tuple[SequenceWindowDataset, SequenceWindowDataset, object]:
        paths = collect_sequence_paths(self.data_dir / "train")
        normalizer = compute_normalizer(paths.values())
        kwargs = dict(
            sequence_paths=[paths[0], paths[1], paths[2]],
            normalizer=normalizer,
            context_size=8,
            horizon_size=12,
            use_velocity=True,
        )
        baseline = SequenceWindowDataset(reflect_prob=0.0, **kwargs)
        mirrored = SequenceWindowDataset(reflect_prob=1.0, **kwargs)
        return baseline, mirrored, normalizer

    def test_reflect_prob_zero_is_identity(self) -> None:
        baseline, _, _ = self._build_reflect_pair()
        paths = collect_sequence_paths(self.data_dir / "train")
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
                self.assertTrue(torch.equal(a[key], b[key]), f"reflect_prob=0 changed {key}")

    def test_reflect_prob_one_mirrors_y_and_preserves_x_and_team(self) -> None:
        baseline, mirrored, _ = self._build_reflect_pair()
        for idx in range(len(baseline)):
            base, flip = baseline[idx], mirrored[idx]
            # Raw y negated, raw x untouched.
            self.assertTrue(torch.allclose(flip["context_raw_xy"][..., 1], -base["context_raw_xy"][..., 1]))
            self.assertTrue(torch.allclose(flip["future_xy_raw"][..., 1], -base["future_xy_raw"][..., 1]))
            self.assertTrue(torch.equal(flip["context_raw_xy"][..., 0], base["context_raw_xy"][..., 0]))
            self.assertTrue(torch.equal(flip["future_xy_raw"][..., 0], base["future_xy_raw"][..., 0]))
            # Velocity y negated (channel 3), velocity x untouched (channel 2).
            self.assertTrue(torch.allclose(flip["context"][..., 3], -base["context"][..., 3]))
            self.assertTrue(torch.equal(flip["context"][..., 2], base["context"][..., 2]))
            # Static (is_player, team) unchanged -- y-mirror keeps the basket and roles.
            self.assertTrue(torch.equal(flip["static"], base["static"]))

    def test_reflect_normalized_and_raw_stay_consistent(self) -> None:
        _, mirrored, normalizer = self._build_reflect_pair()
        for idx in range(len(mirrored)):
            flip = mirrored[idx]
            # Denormalizing the mirrored normalized positions must reproduce the
            # mirrored raw coordinates (validates the -2*mu_y/sigma_y shift term).
            recon = normalizer.denormalize_xy(flip["context"][..., :2])
            self.assertTrue(torch.allclose(recon, flip["context_raw_xy"], atol=1e-4))

    def _build_dataset(self, reflect_prob: float, reflect_x_prob: float) -> SequenceWindowDataset:
        paths = collect_sequence_paths(self.data_dir / "train")
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

    def test_reflect_x_prob_one_mirrors_x_and_preserves_y_and_team(self) -> None:
        base = self._build_dataset(0.0, 0.0)
        flip_x = self._build_dataset(0.0, 1.0)
        for idx in range(len(base)):
            b, f = base[idx], flip_x[idx]
            # Raw x negated, raw y untouched.
            self.assertTrue(torch.allclose(f["context_raw_xy"][..., 0], -b["context_raw_xy"][..., 0]))
            self.assertTrue(torch.allclose(f["future_xy_raw"][..., 0], -b["future_xy_raw"][..., 0]))
            self.assertTrue(torch.equal(f["context_raw_xy"][..., 1], b["context_raw_xy"][..., 1]))
            # Velocity x negated (channel 2), velocity y untouched (channel 3).
            self.assertTrue(torch.allclose(f["context"][..., 2], -b["context"][..., 2]))
            self.assertTrue(torch.equal(f["context"][..., 3], b["context"][..., 3]))
            self.assertTrue(torch.equal(f["static"], b["static"]))

    def test_reflect_xy_mirrors_both_axes(self) -> None:
        base = self._build_dataset(0.0, 0.0)
        normalizer = compute_normalizer(collect_sequence_paths(self.data_dir / "train").values())
        flip_xy = self._build_dataset(1.0, 1.0)
        for idx in range(len(base)):
            b, f = base[idx], flip_xy[idx]
            self.assertTrue(torch.allclose(f["future_xy_raw"], -b["future_xy_raw"]))
            self.assertTrue(torch.allclose(f["context_raw_xy"], -b["context_raw_xy"]))
            self.assertTrue(torch.equal(f["static"], b["static"]))
            # Both normalized axes stay consistent with their (negated) raw coordinates.
            recon = normalizer.denormalize_xy(f["context"][..., :2])
            self.assertTrue(torch.allclose(recon, f["context_raw_xy"], atol=1e-4))

    def test_reflect_context_matches_training_path(self) -> None:
        # The inference-time reflect_context helper must reproduce the context arm of the
        # training-time _reflect_window (exercised here via the reflect_prob=1 dataset).
        normalizer = compute_normalizer(collect_sequence_paths(self.data_dir / "train").values())
        shift_x, shift_y = reflect_shifts(normalizer)
        base = self._build_dataset(0.0, 0.0)
        cases = {
            "x": (self._build_dataset(0.0, 1.0), True, False),
            "y": (self._build_dataset(1.0, 0.0), False, True),
            "xy": (self._build_dataset(1.0, 1.0), True, True),
        }
        for _, (mirrored, flip_x, flip_y) in cases.items():
            for idx in range(len(base)):
                expected = mirrored[idx]["context"]
                got = reflect_context(base[idx]["context"], True, shift_x, shift_y, flip_x, flip_y)
                self.assertTrue(torch.allclose(got, expected, atol=1e-5))

    def test_reflect_prediction_xy_is_involution(self) -> None:
        normalizer = compute_normalizer(collect_sequence_paths(self.data_dir / "train").values())
        shift_x, shift_y = reflect_shifts(normalizer)
        prediction = torch.randn(2, 12, 11, 2)
        # No flip is the identity.
        identity = reflect_prediction_xy(prediction, shift_x, shift_y, False, False)
        self.assertTrue(torch.equal(identity, prediction))
        # Applying the same reflection twice returns the original (it is its own inverse).
        once = reflect_prediction_xy(prediction, shift_x, shift_y, True, True)
        twice = reflect_prediction_xy(once, shift_x, shift_y, True, True)
        self.assertTrue(torch.allclose(twice, prediction, atol=1e-5))

    def test_metrics_match_expected_values(self) -> None:
        prediction = torch.tensor([[[[1.0, 2.0]]]])
        target = torch.zeros_like(prediction)
        metrics = compute_raw_metrics(prediction, target)
        self.assertAlmostEqual(metrics["mse"], 2.5, places=6)
        self.assertAlmostEqual(metrics["ade"], 5.0, places=6)
        self.assertAlmostEqual(metrics["rmse"], 2.5**0.5, places=6)
        self.assertAlmostEqual(metrics["ade_l2"], 5.0**0.5, places=6)
        self.assertAlmostEqual(metrics["fde_l2"], 5.0**0.5, places=6)
        self.assertAlmostEqual(float(normalized_mse_loss(prediction, target).item()), 2.5, places=6)

    def test_per_step_l2_error_returns_per_horizon_distances(self) -> None:
        prediction = torch.zeros(2, 3, 4, 2)
        target = torch.zeros_like(prediction)
        target[:, 0] = 3.0  # 2D offset of (3, 3) -> L2 = sqrt(18) at step 0
        target[:, 2, :, 0] = 4.0  # offset (4, 0) -> L2 = 4 at step 2
        per_step = per_step_l2_error(prediction, target)
        self.assertEqual(tuple(per_step.shape), (3,))
        self.assertAlmostEqual(float(per_step[0].item()), 18.0**0.5, places=6)
        self.assertAlmostEqual(float(per_step[1].item()), 0.0, places=6)
        self.assertAlmostEqual(float(per_step[2].item()), 4.0, places=6)


class AuxiliaryLossTests(unittest.TestCase):
    def test_smoothness_zero_for_constant_velocity(self) -> None:
        # A path moving at constant velocity has zero acceleration -> zero penalty.
        time = torch.arange(12, dtype=torch.float32).view(1, 12, 1, 1)
        path = time * torch.tensor([1.5, -0.7]).view(1, 1, 1, 2)
        loss = smoothness_loss(path.expand(2, 12, 11, 2).contiguous())
        self.assertEqual(tuple(loss.shape), ())
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_smoothness_known_kink(self) -> None:
        # Positions 0 -> 1 -> 0: accel = 0 - 2*1 + 0 = -2 in x; sum of squares = 4.
        path = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)
        path = torch.cat([path, torch.zeros_like(path)], dim=-1)  # [1, 3, 1, 2]
        loss = smoothness_loss(path)
        self.assertAlmostEqual(float(loss.item()), 4.0, places=6)

    def test_boundary_zero_when_in_bounds(self) -> None:
        pred = torch.tensor([40.0, 20.0]).view(1, 1, 1, 2)
        loss = boundary_loss(pred, court_x_half=47.0, court_y_half=25.0, margin=2.0)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_boundary_hinge_value_when_out_of_bounds(self) -> None:
        # x=50 with x_half+margin=49 -> overshoot 1; y in bounds -> loss = 1**2 = 1.
        pred = torch.tensor([50.0, 0.0]).view(1, 1, 1, 2)
        loss = boundary_loss(pred, court_x_half=47.0, court_y_half=25.0, margin=2.0)
        self.assertAlmostEqual(float(loss.item()), 1.0, places=6)

    def _static_batch(self) -> torch.Tensor:
        return torch.stack([IS_PLAYER, TEAM_VALUES], dim=-1).unsqueeze(0)  # [1, 11, 2]

    def test_ball_plausibility_zero_when_near_player(self) -> None:
        static = self._static_batch()
        pred = torch.zeros(1, 1, 11, 2)
        pred[:, :, 1:, 0] = 3.0  # players 3 ft away in x; ball at origin (idx 0, team==0)
        loss = ball_plausibility_loss(pred, static=static, max_dist=8.0)
        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_ball_plausibility_hinge_when_isolated(self) -> None:
        static = self._static_batch()
        pred = torch.zeros(1, 1, 11, 2)
        pred[:, :, 1:, 0] = 20.0  # every player 20 ft away -> overshoot 12 -> 144.
        loss = ball_plausibility_loss(pred, static=static, max_dist=8.0)
        self.assertAlmostEqual(float(loss.item()), 144.0, places=6)


class MixtureMetricTests(unittest.TestCase):
    """Phase 7 (D1): Gaussian-mixture NLL and marginal minADE_K / minFDE_K."""

    def test_mixture_nll_k1_matches_bivariate_gaussian(self) -> None:
        # Single component, unit isotropic covariance -> NLL = sum_t [log(2pi) + 0.5||y-mu||^2].
        target = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])  # [B=1, T=2, N=1, 2]
        means = torch.zeros(1, 2, 1, 1, 2)
        log_sigma = torch.zeros(1, 2, 1, 1, 2)  # sigma = 1
        rho = torch.zeros(1, 2, 1, 1)
        log_pi = torch.zeros(1, 1, 1)  # log(1)
        params = (means, log_sigma, rho, log_pi)
        nll = mixture_nll(params, target)
        expected = 2.0 * math.log(2.0 * math.pi) + 0.5 * (1.0**2 + 2.0**2)
        self.assertAlmostEqual(float(nll.item()), expected, places=5)

    def test_mixture_nll_k2_is_logsumexp_over_modes(self) -> None:
        torch.manual_seed(0)
        target = torch.randn(2, 3, 4, 2)
        means = torch.randn(2, 3, 4, 2, 2)
        log_sigma = torch.zeros(2, 3, 4, 2, 2)
        rho = torch.zeros(2, 3, 4, 2)
        logits = torch.randn(2, 4, 2)
        log_pi = torch.log_softmax(logits, dim=-1)
        params = (means, log_sigma, rho, log_pi)
        nll = mixture_nll(params, target)
        # Manual: per-mode trajectory log-lik with sigma=1, rho=0 is the Gaussian sum over steps.
        diff = target.unsqueeze(3) - means  # [B,T,N,K,2]
        per_step = -math.log(2.0 * math.pi) - 0.5 * (diff**2).sum(dim=-1)  # [B,T,N,K]
        traj = per_step.sum(dim=1)  # [B,N,K]
        manual = -torch.logsumexp(log_pi + traj, dim=-1).mean()
        self.assertAlmostEqual(float(nll.item()), float(manual.item()), places=5)

    def test_mixture_nll_is_finite_under_extreme_rho_and_sigma(self) -> None:
        target = torch.randn(2, 4, 3, 2)
        means = torch.randn(2, 4, 3, 2, 2)
        log_sigma = torch.full((2, 4, 3, 2, 2), 20.0)  # clamp guards huge sigma
        rho = torch.full((2, 4, 3, 2), 5.0)  # |rho|>1 before the head's tanh; guarded here
        log_pi = torch.log_softmax(torch.randn(2, 3, 2), dim=-1)
        nll = mixture_nll((means, log_sigma, rho, log_pi), target)
        self.assertTrue(torch.isfinite(nll))

    def test_min_ade_k_selects_best_mode(self) -> None:
        target = torch.zeros(1, 2, 1, 2)
        modes = torch.zeros(1, 2, 1, 2, 2)  # K=2
        modes[..., 0, :] = 5.0  # mode 0 is 5 ft off everywhere
        modes[..., 1, :] = 0.0  # mode 1 matches the target exactly
        self.assertAlmostEqual(min_ade_k(modes, target), 0.0, places=6)
        self.assertAlmostEqual(min_fde_k(modes, target), 0.0, places=6)

    def test_min_ade_k_k1_equals_ade_l2(self) -> None:
        torch.manual_seed(1)
        target = torch.randn(3, 12, 11, 2)
        pred = torch.randn(3, 12, 11, 2)
        raw = compute_raw_metrics(pred, target)
        modes = pred.unsqueeze(3)  # K=1
        self.assertAlmostEqual(min_ade_k(modes, target), raw["ade_l2"], places=5)
        self.assertAlmostEqual(min_fde_k(modes, target), raw["fde_l2"], places=5)


if __name__ == "__main__":
    unittest.main()
