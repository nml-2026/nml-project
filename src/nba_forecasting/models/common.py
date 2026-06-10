from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

import torch
from torch import nn

NUM_RELATION_TYPES = 3


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _static_features(context: torch.Tensor) -> torch.Tensor:
    return context[:, 0, :, -2:]


def _full_graph_mask(batch_size: int, num_entities: int, device: torch.device) -> torch.Tensor:
    mask = torch.ones(batch_size, num_entities, num_entities, device=device, dtype=torch.bool)
    eye = torch.eye(num_entities, device=device, dtype=torch.bool).unsqueeze(0)
    return mask & ~eye


def _semantic_graph_mask(static: torch.Tensor) -> torch.Tensor:
    teams = static[:, :, 1]
    team_i = teams.unsqueeze(2)
    team_j = teams.unsqueeze(1)
    same_team = (team_i == team_j) & (team_i != 0) & (team_j != 0)
    involves_ball = (team_i == 0) | (team_j == 0)
    num_entities = static.shape[1]
    mask = same_team | involves_ball
    eye = torch.eye(num_entities, device=static.device, dtype=torch.bool).unsqueeze(0)
    return mask & ~eye


def _knn_graph_mask(positions: torch.Tensor, k_neighbors: int) -> torch.Tensor:
    batch_size, num_entities, _ = positions.shape
    distances = torch.cdist(positions, positions)
    eye = torch.eye(num_entities, device=positions.device, dtype=torch.bool).unsqueeze(0)
    distances = distances.masked_fill(eye, float("inf"))
    k_neighbors = max(1, min(k_neighbors, num_entities - 1))
    neighbors = distances.topk(k=k_neighbors, largest=False).indices
    mask = torch.zeros(batch_size, num_entities, num_entities, device=positions.device, dtype=torch.bool)
    mask.scatter_(2, neighbors, True)
    return mask | mask.transpose(1, 2)


def _edge_features(positions: torch.Tensor, static: torch.Tensor, use_semantic_flags: bool) -> torch.Tensor:
    pos_i = positions.unsqueeze(2)
    pos_j = positions.unsqueeze(1)
    delta = pos_j - pos_i
    distance = torch.linalg.norm(delta, dim=-1, keepdim=True)
    features = [delta, distance]
    if use_semantic_flags:
        teams = static[:, :, 1]
        team_i = teams.unsqueeze(2)
        team_j = teams.unsqueeze(1)
        same_team = ((team_i == team_j) & (team_i != 0) & (team_j != 0)).float().unsqueeze(-1)
        involves_ball = ((team_i == 0) | (team_j == 0)).float().unsqueeze(-1)
        features.extend([same_team, involves_ball])
    return torch.cat(features, dim=-1)


class ForecasterHead(nn.Module):
    """Base seam for forecasting trajectory heads."""

    pass


class TypeConditionedDeltaHead(ForecasterHead):
    def __init__(self, hidden_dim: int, dropout: float, separate_ball_head: bool) -> None:
        super().__init__()
        self.separate_ball_head = separate_ball_head
        self.player_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.ball_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self.player_scale = nn.Parameter(torch.tensor(1.0))
        self.ball_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, hidden_state: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        batch_size, num_entities, hidden_dim = hidden_state.shape
        flat_hidden = hidden_state.reshape(batch_size * num_entities, hidden_dim)
        player_delta = self.player_head(flat_hidden).reshape(batch_size, num_entities, 2) * self.player_scale
        if not self.separate_ball_head:
            return player_delta
        ball_delta = self.ball_head(flat_hidden).reshape(batch_size, num_entities, 2) * self.ball_scale
        is_ball = (static[:, :, 1] == 0).unsqueeze(-1)
        return torch.where(is_ball, ball_delta, player_delta)


class MixtureParams(NamedTuple):
    """Parameters of a per-entity Gaussian mixture over trajectories.

    All tensors live in the model's (normalized) position space. ``K`` is the number of
    mixture components. The ``K`` modes share one weight per entity across the horizon, so
    each mode is a *coherent* trajectory rather than a per-step choice.
    """

    means: torch.Tensor  # [B, T, N, K, 2] per-mode position-mean trajectories
    log_sigma: torch.Tensor  # [B, T, N, K, 2] per-step log std-dev
    rho: torch.Tensor  # [B, T, N, K] per-step xy correlation in (-1, 1)
    log_pi: torch.Tensor  # [B, N, K] per-entity log mixture weights


class MixtureForecast(NamedTuple):
    params: MixtureParams
    student: torch.Tensor | None = None


@runtime_checkable
class MixtureForecaster(Protocol):
    def mixture_forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> MixtureParams | MixtureForecast: ...


class ForecastOutput(NamedTuple):
    point: torch.Tensor
    distribution: MixtureParams | None = None
    modes: torch.Tensor | None = None
    auxiliary_loss: torch.Tensor | None = None
    mode_logits: torch.Tensor | None = None
    oracle_student: torch.Tensor | None = None


@runtime_checkable
class SampleForecaster(Protocol):
    def sample_forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


def unwrap_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_orig_mod", model)


def _forecast_capability_model(model: nn.Module) -> nn.Module:
    return unwrap_model(model)


def forecast_model(
    model: nn.Module,
    context: torch.Tensor,
    *,
    future_xy: torch.Tensor | None = None,
    teacher_forcing_prob: float = 0.0,
    scheduled_sampling_prob: float = 0.0,
    mdn_point_estimate: str | None = None,
) -> ForecastOutput:
    capability_model = _forecast_capability_model(model)
    if mdn_point_estimate is not None:
        assert isinstance(capability_model, MixtureForecaster)
        mixture = capability_model.mixture_forward(
            context,
            future_xy=future_xy,
            teacher_forcing_prob=teacher_forcing_prob,
            scheduled_sampling_prob=scheduled_sampling_prob,
        )
        if isinstance(mixture, MixtureForecast):
            params = mixture.params
            student = mixture.student
        else:
            params = mixture
            student = None
        if mdn_point_estimate == "student":
            if student is None:
                raise RuntimeError("mdn_point_estimate='student' requires an oracle student head.")
            point = student
        else:
            point = reduce_mixture(params.means, params.log_pi, mdn_point_estimate)
        return ForecastOutput(point=point, distribution=params, modes=params.means, mode_logits=params.log_pi, oracle_student=student)
    raw: Any = model(
        context,
        future_xy=future_xy,
        teacher_forcing_prob=teacher_forcing_prob,
        scheduled_sampling_prob=scheduled_sampling_prob,
    )
    if isinstance(raw, ForecastOutput):
        return raw
    assert isinstance(raw, torch.Tensor)
    if isinstance(capability_model, SampleForecaster):
        sample_output = capability_model.sample_forward(
            context,
            future_xy=future_xy,
            teacher_forcing_prob=teacher_forcing_prob,
            scheduled_sampling_prob=scheduled_sampling_prob,
        )
        if isinstance(sample_output, tuple):
            modes, auxiliary_loss = sample_output
        else:
            modes = sample_output
            auxiliary_loss = None
        point = modes.mean(dim=3)
        return ForecastOutput(point=point, modes=modes, auxiliary_loss=auxiliary_loss)
    return ForecastOutput(point=raw)


class MixtureDeltaHead(ForecasterHead):
    """Gaussian-mixture trajectory head: K coherent modes with per-entity weights.

    Emits, per (step, entity, component), a position-*delta* mean (2), log-sigma (2) and
    correlation rho (1) from the per-step decoder state; and per (entity, component) a
    mixture logit from the per-entity summary state. The K modes share a single weight per
    entity across the horizon, so cumsumming each mode's per-step delta means yields K
    coherent trajectory hypotheses. ``sigma``/``rho`` are emitted per step as the local
    observation noise (they are not cumsumed).

    This mirrors the delta/cumsum inductive bias of ``TypeConditionedDeltaHead`` (the head
    outputs deltas; the model adds ``last_pos + cumsum(.)``), so a single-component mixture
    with fixed sigma reduces to the deterministic MSE decoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        num_components: int,
        min_log_sigma: float = -7.0,
        max_log_sigma: float = 5.0,
    ) -> None:
        super().__init__()
        if num_components < 1:
            raise ValueError(f"num_components must be >= 1, got {num_components}.")
        self.num_components = num_components
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma
        # 5 params per (component, step): mu_delta (2), log_sigma (2), rho (1).
        self.step_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_components * 5),
        )
        self.mixture_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, num_components),
        )
        self.delta_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        per_step_state: torch.Tensor,
        entity_state: torch.Tensor,
        batch_size: int,
        horizon_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return raw per-step (mu_delta, log_sigma, rho) and per-entity log_pi.

        Args:
            per_step_state: decoder state, shape ``[B*T, N, H]`` (flat index ``b*T + t``).
            entity_state: per-entity summary, shape ``[B, N, H]``.
            batch_size, horizon_size: used to unflatten the per-step axis.
        Returns:
            mu_delta ``[B, T, N, K, 2]``, log_sigma ``[B, T, N, K, 2]``,
            rho ``[B, T, N, K]``, log_pi ``[B, N, K]``.
        """
        _, num_entities, hidden_dim = per_step_state.shape
        raw = self.step_head(per_step_state.reshape(-1, hidden_dim))
        raw = raw.reshape(batch_size, horizon_size, num_entities, self.num_components, 5)
        mu_delta = raw[..., 0:2] * self.delta_scale
        log_sigma = raw[..., 2:4].clamp(self.min_log_sigma, self.max_log_sigma)
        rho = torch.tanh(raw[..., 4]) * 0.999  # keep strictly inside (-1, 1) for stability
        logits = self.mixture_head(entity_state)  # [B, N, K]
        log_pi = torch.log_softmax(logits, dim=-1)
        return mu_delta, log_sigma, rho, log_pi


def reduce_mixture(means: torch.Tensor, log_pi: torch.Tensor, point_estimate: str) -> torch.Tensor:
    """Collapse a trajectory mixture to a single ``[B, T, N, 2]`` point estimate.

    ``"mean"`` returns the per-entity pi-weighted mean of the mode means (the Bayes-optimal
    estimate for the competition's squared metric); ``"mode"`` returns the highest-weight
    mode's trajectory; ``"medoid"`` returns the mode with minimum aggregate distance to all
    other modes. ``means`` is ``[B, T, N, K, 2]`` and ``log_pi`` is ``[B, N, K]``.
    """
    if point_estimate == "mean":
        batch_size, num_entities, num_components = log_pi.shape
        weights = log_pi.exp().view(batch_size, 1, num_entities, num_components, 1)
        return (means * weights).sum(dim=3)
    if point_estimate == "mode":
        best = log_pi.argmax(dim=-1)  # [B, N]
        index = best.view(best.shape[0], 1, best.shape[1], 1, 1).expand(-1, means.shape[1], -1, -1, means.shape[-1])
        return torch.gather(means, 3, index).squeeze(3)
    if point_estimate == "medoid":
        # Pairwise summed squared distance over horizon and xy: [B, N, K, K].
        modes = means.permute(0, 2, 3, 1, 4)
        pairwise = (modes.unsqueeze(3) - modes.unsqueeze(2)).square().sum(dim=(-1, -2))
        best = pairwise.sum(dim=-1).argmin(dim=-1)
        index = best.view(best.shape[0], 1, best.shape[1], 1, 1).expand(-1, means.shape[1], -1, -1, means.shape[-1])
        return torch.gather(means, 3, index).squeeze(3)
    raise ValueError(f"Unknown point_estimate {point_estimate!r}. Expected 'mean', 'mode', or 'medoid'.")


def _next_autoregressive_inputs(
    current_positions: torch.Tensor,
    predicted_positions: torch.Tensor,
    target_positions: torch.Tensor,
    use_velocity: bool,
    teacher_forcing_prob: float,
    scheduled_sampling_prob: float,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    batch_size = current_positions.shape[0]
    if scheduled_sampling_prob > 0.0:
        use_model_mask = torch.rand(batch_size, 1, 1, device=current_positions.device) < scheduled_sampling_prob
        next_positions = torch.where(use_model_mask, predicted_positions, target_positions)
    elif teacher_forcing_prob > 0.0:
        use_teacher_mask = torch.rand(batch_size, 1, 1, device=current_positions.device) < teacher_forcing_prob
        next_positions = torch.where(use_teacher_mask, target_positions, predicted_positions)
    else:
        next_positions = predicted_positions
    next_velocity = next_positions - current_positions if use_velocity else None
    return next_positions, next_velocity
