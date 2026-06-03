from __future__ import annotations

import math

import torch


def normalized_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def per_example_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2, dim=(1, 2, 3))


def compute_raw_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = prediction - target
    mse = torch.mean(diff**2).item()
    ade = torch.mean(torch.sum(diff**2, dim=-1)).item()
    rmse = mse**0.5
    distance = torch.linalg.vector_norm(diff, dim=-1)
    ade_l2 = torch.mean(distance).item()
    fde_l2 = torch.mean(distance[:, -1]).item()
    return {
        "mse": mse,
        "ade": ade,
        "rmse": rmse,
        "ade_l2": ade_l2,
        "fde_l2": fde_l2,
    }


def per_step_l2_error(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean L2 displacement at each horizon step, shape ``[T]``."""
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    return distance.mean(dim=(0, 2))


def smoothness_loss(prediction: torch.Tensor) -> torch.Tensor:
    """Penalize jagged paths via the squared second time-difference (acceleration).

    Args:
        prediction: positions, shape ``[B, T, N, 2]`` (any consistent space; raw feet
            in training so the weight has interpretable scale). Requires ``T >= 3``.
    Returns:
        scalar mean over batch/entities/time of the squared acceleration norm.
    """
    accel = prediction[:, 2:] - 2.0 * prediction[:, 1:-1] + prediction[:, :-2]
    return torch.mean(torch.sum(accel**2, dim=-1))


def boundary_loss(
    prediction_raw: torch.Tensor,
    court_x_half: float,
    court_y_half: float,
    margin: float = 0.0,
) -> torch.Tensor:
    """Hinge penalty for predictions outside the (origin-centered) court extents.

    Args:
        prediction_raw: positions in court feet, shape ``[B, T, N, 2]``.
        court_x_half, court_y_half: court half-extents in feet.
        margin: feet of slack added to each half-extent before the hinge engages.
    Returns:
        scalar mean of the squared out-of-bounds overshoot (x and y summed).
    """
    over_x = torch.relu(prediction_raw[..., 0].abs() - (court_x_half + margin))
    over_y = torch.relu(prediction_raw[..., 1].abs() - (court_y_half + margin))
    return torch.mean(over_x**2 + over_y**2)


def _bivariate_log_prob(
    target: torch.Tensor,
    means: torch.Tensor,
    log_sigma: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    """Per-step bivariate-Gaussian log-density of ``target`` under each mixture component.

    Args:
        target: positions ``[B, T, N, 2]``.
        means: per-mode position means ``[B, T, N, K, 2]``.
        log_sigma: per-mode log std-dev ``[B, T, N, K, 2]``.
        rho: per-mode xy correlation in (-1, 1), ``[B, T, N, K]``.
    Returns:
        log-density ``[B, T, N, K]`` (the standard Graves/Social-LSTM form).
    """
    diff = target.unsqueeze(3) - means  # [B, T, N, K, 2]
    sigma = log_sigma.exp()
    dx = diff[..., 0] / sigma[..., 0]
    dy = diff[..., 1] / sigma[..., 1]
    one_minus_rho2 = (1.0 - rho**2).clamp_min(1e-6)
    z = dx**2 + dy**2 - 2.0 * rho * dx * dy
    return -math.log(2.0 * math.pi) - log_sigma[..., 0] - log_sigma[..., 1] - 0.5 * torch.log(one_minus_rho2) - z / (2.0 * one_minus_rho2)


def mixture_nll(params, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of a per-entity Gaussian mixture over trajectories.

    Each mode's trajectory log-likelihood is the sum over horizon steps of its per-step
    bivariate-Gaussian log-density; the K coherent modes are combined with a per-entity
    weight via ``logsumexp``. ``params`` is a ``(means, log_sigma, rho, log_pi)`` tuple
    (e.g. a ``MixtureParams``); ``target`` is normalized positions ``[B, T, N, 2]``.
    Returns the scalar mean NLL over (batch, entity).
    """
    means, log_sigma, rho, log_pi = params
    step_log_prob = _bivariate_log_prob(target, means, log_sigma, rho)  # [B, T, N, K]
    traj_log_prob = step_log_prob.sum(dim=1)  # [B, N, K]
    log_mixture = torch.logsumexp(log_pi + traj_log_prob, dim=-1)  # [B, N]
    return -log_mixture.mean()


def min_ade_k(modes_raw: torch.Tensor, target_raw: torch.Tensor) -> float:
    """Marginal minADE_K (feet): per entity, the min over its K modes of the mode's ADE.

    Args:
        modes_raw: per-mode position trajectories in court feet, ``[B, T, N, K, 2]``.
        target_raw: ground-truth positions in court feet, ``[B, T, N, 2]``.
    Returns:
        mean over (batch, entity) of the best-mode average displacement error.
    """
    distance = torch.linalg.vector_norm(modes_raw - target_raw.unsqueeze(3), dim=-1)  # [B, T, N, K]
    ade_per_mode = distance.mean(dim=1)  # [B, N, K]
    return ade_per_mode.min(dim=-1).values.mean().item()


def min_fde_k(modes_raw: torch.Tensor, target_raw: torch.Tensor) -> float:
    """Marginal minFDE_K (feet): per entity, the min over its K modes of the final-step L2."""
    final_distance = torch.linalg.vector_norm(modes_raw[:, -1] - target_raw[:, -1].unsqueeze(2), dim=-1)  # [B, N, K]
    return final_distance.min(dim=-1).values.mean().item()


def ball_plausibility_loss(
    prediction_raw: torch.Tensor,
    static: torch.Tensor,
    max_dist: float,
) -> torch.Tensor:
    """Hinge penalty when the ball is implausibly far from every player.

    Args:
        prediction_raw: positions in court feet, shape ``[B, T, N, 2]``.
        static: per-entity (is_player, team), shape ``[B, N, 2]``; the ball is the
            entity with ``team == 0`` (players are +/-1).
        max_dist: feet; the hinge engages when the nearest player is farther than this.
    Returns:
        scalar mean over (batch, time) of the squared ball-isolation overshoot. Returns
        a 0 tensor (graph-connected) when a batch element has no ball or no players.
    """
    is_ball = static[..., 1] == 0  # [B, N]
    is_player = ~is_ball  # [B, N]
    if not bool(is_ball.any()) or not bool(is_player.any()):
        return prediction_raw.sum() * 0.0
    # Per-batch ball/player counts can vary in principle; rely on the fixed NBA layout
    # (exactly one ball, ten players) and gather by mask along the entity axis.
    ball_xy = prediction_raw[:, :, is_ball[0], :]  # [B, T, 1, 2]
    player_xy = prediction_raw[:, :, is_player[0], :]  # [B, T, P, 2]
    dist = torch.linalg.vector_norm(player_xy - ball_xy, dim=-1)  # [B, T, P]
    nearest = dist.min(dim=-1).values  # [B, T]
    return torch.mean(torch.relu(nearest - max_dist) ** 2)
