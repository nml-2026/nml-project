from __future__ import annotations

import math

import torch

from nba_forecasting.metrics import (
    ball_plausibility_loss,
    best_mode_indices,
    boundary_loss,
    compute_raw_metrics,
    min_ade_k,
    min_fde_k,
    mixture_nll,
    normalized_mse_loss,
    per_step_l2_error,
    smoothness_loss,
    softmin_ade_loss,
)
from nba_forecasting.models.common import reduce_mixture
from tests.helpers import IS_PLAYER, TEAM_VALUES


def test_metrics_match_expected_values() -> None:
    prediction = torch.tensor([[[[1.0, 2.0]]]])
    target = torch.zeros_like(prediction)
    metrics = compute_raw_metrics(prediction, target)
    assert abs(metrics["mse"] - 2.5) < 1e-5
    assert abs(metrics["ade"] - metrics["mse"]) < 1e-5
    assert abs(metrics["rmse"] - 2.5**0.5) < 1e-5
    assert abs(metrics["ade_l2"] - 5.0**0.5) < 1e-5
    assert abs(metrics["fde_l2"] - 5.0**0.5) < 1e-5
    assert abs(float(normalized_mse_loss(prediction, target).item()) - 2.5) < 1e-5


def test_per_step_l2_error_returns_per_horizon_distances() -> None:
    prediction = torch.zeros(2, 3, 4, 2)
    target = torch.zeros_like(prediction)
    target[:, 0] = 3.0
    target[:, 2, :, 0] = 4.0
    per_step = per_step_l2_error(prediction, target)
    assert tuple(per_step.shape) == (3,)
    assert abs(float(per_step[0].item()) - 18.0**0.5) < 1e-5
    assert abs(float(per_step[1].item()) - 0.0) < 1e-5
    assert abs(float(per_step[2].item()) - 4.0) < 1e-5


def test_smoothness_zero_for_constant_velocity() -> None:
    time = torch.arange(12, dtype=torch.float32).view(1, 12, 1, 1)
    path = time * torch.tensor([1.5, -0.7]).view(1, 1, 1, 2)
    loss = smoothness_loss(path.expand(2, 12, 11, 2).contiguous())
    assert tuple(loss.shape) == ()
    assert abs(float(loss.item()) - 0.0) < 1e-5


def test_smoothness_known_kink() -> None:
    path = torch.tensor([0.0, 1.0, 0.0]).view(1, 3, 1, 1)
    path = torch.cat([path, torch.zeros_like(path)], dim=-1)
    loss = smoothness_loss(path)
    assert abs(float(loss.item()) - 4.0) < 1e-5


def test_softmin_ade_and_best_mode_indices() -> None:
    target = torch.zeros(1, 2, 1, 2)
    modes = torch.tensor([[[[[0.0, 0.0], [2.0, 0.0]]], [[[0.0, 0.0], [2.0, 0.0]]]]])

    loss = softmin_ade_loss(modes, target, temperature=0.5)
    best = best_mode_indices(modes, target)

    assert tuple(best.shape) == (1, 1)
    assert int(best.item()) == 0
    assert 0.0 <= float(loss.item()) < 1.0


def test_boundary_zero_when_in_bounds() -> None:
    pred = torch.tensor([40.0, 20.0]).view(1, 1, 1, 2)
    loss = boundary_loss(pred, court_x_half=47.0, court_y_half=25.0, margin=2.0)
    assert abs(float(loss.item()) - 0.0) < 1e-5


def test_boundary_hinge_value_when_out_of_bounds() -> None:
    pred = torch.tensor([50.0, 0.0]).view(1, 1, 1, 2)
    loss = boundary_loss(pred, court_x_half=47.0, court_y_half=25.0, margin=2.0)
    assert abs(float(loss.item()) - 1.0) < 1e-5


def _static_batch() -> torch.Tensor:
    return torch.stack([IS_PLAYER, TEAM_VALUES], dim=-1).unsqueeze(0)


def test_ball_plausibility_zero_when_near_player() -> None:
    static = _static_batch()
    pred = torch.zeros(1, 1, 11, 2)
    pred[:, :, 1:, 0] = 3.0
    loss = ball_plausibility_loss(pred, static=static, max_dist=8.0)
    assert abs(float(loss.item()) - 0.0) < 1e-5


def test_ball_plausibility_hinge_when_isolated() -> None:
    static = _static_batch()
    pred = torch.zeros(1, 1, 11, 2)
    pred[:, :, 1:, 0] = 20.0
    loss = ball_plausibility_loss(pred, static=static, max_dist=8.0)
    assert abs(float(loss.item()) - 144.0) < 1e-5


def test_mixture_nll_k1_matches_bivariate_gaussian() -> None:
    target = torch.tensor([[[[1.0, 0.0], [0.0, 2.0]]]])
    means = torch.zeros(1, 2, 1, 1, 2)
    log_sigma = torch.zeros(1, 2, 1, 1, 2)
    rho = torch.zeros(1, 2, 1, 1)
    log_pi = torch.zeros(1, 1, 1)
    params = (means, log_sigma, rho, log_pi)
    nll = mixture_nll(params, target)
    expected = 2.0 * math.log(2.0 * math.pi) + 0.5 * (1.0**2 + 2.0**2)
    assert abs(float(nll.item()) - expected) < 1e-4


def test_mixture_nll_k2_is_logsumexp_over_modes() -> None:
    torch.manual_seed(0)
    target = torch.randn(2, 3, 4, 2)
    means = torch.randn(2, 3, 4, 2, 2)
    log_sigma = torch.zeros(2, 3, 4, 2, 2)
    rho = torch.zeros(2, 3, 4, 2)
    logits = torch.randn(2, 4, 2)
    log_pi = torch.log_softmax(logits, dim=-1)
    params = (means, log_sigma, rho, log_pi)
    nll = mixture_nll(params, target)
    diff = target.unsqueeze(3) - means
    per_step = -math.log(2.0 * math.pi) - 0.5 * (diff**2).sum(dim=-1)
    traj = per_step.sum(dim=1)
    manual = -torch.logsumexp(log_pi + traj, dim=-1).mean()
    assert abs(float(nll.item()) - float(manual.item())) < 1e-4


def test_mixture_nll_is_finite_under_extreme_rho_and_sigma() -> None:
    target = torch.randn(2, 4, 3, 2)
    means = torch.randn(2, 4, 3, 2, 2)
    log_sigma = torch.full((2, 4, 3, 2, 2), 20.0)
    rho = torch.full((2, 4, 3, 2), 5.0)
    log_pi = torch.log_softmax(torch.randn(2, 3, 2), dim=-1)
    nll = mixture_nll((means, log_sigma, rho, log_pi), target)
    assert torch.isfinite(nll)


def test_min_ade_k_selects_best_mode() -> None:
    target = torch.zeros(1, 2, 1, 2)
    modes = torch.zeros(1, 2, 1, 2, 2)
    modes[..., 0, :] = 5.0
    modes[..., 1, :] = 0.0
    assert abs(min_ade_k(modes, target) - 0.0) < 1e-5
    assert abs(min_fde_k(modes, target) - 0.0) < 1e-5


def test_reduce_mixture_medoid_selects_central_existing_mode() -> None:
    means = torch.tensor([[[[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]], [[[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]]]])
    log_pi = torch.log_softmax(torch.tensor([[[10.0, 0.0, 0.0]]]), dim=-1)

    prediction = reduce_mixture(means, log_pi, "medoid")

    expected = means[:, :, :, 1, :]
    assert torch.equal(prediction, expected)


def test_min_ade_k_k1_equals_ade_l2() -> None:
    torch.manual_seed(1)
    target = torch.randn(3, 12, 11, 2)
    pred = torch.randn(3, 12, 11, 2)
    raw = compute_raw_metrics(pred, target)
    modes = pred.unsqueeze(3)
    assert abs(min_ade_k(modes, target) - raw["ade_l2"]) < 1e-4
    assert abs(min_fde_k(modes, target) - raw["fde_l2"]) < 1e-4
