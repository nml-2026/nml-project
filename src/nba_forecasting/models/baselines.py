from __future__ import annotations

import torch
from torch import nn

from .common import _next_autoregressive_inputs, _static_features


class NotebookExactGRUCell(nn.Module):
    def __init__(self, input_dim: int, state_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.pre_norm = nn.LayerNorm(input_dim)
        self.update_gate = nn.Sequential(
            nn.Linear(input_dim + state_dim, state_dim, bias=True),
            nn.Sigmoid(),
        )
        self.reset_gate = nn.Sequential(
            nn.Linear(input_dim + state_dim, state_dim, bias=True),
            nn.Sigmoid(),
        )
        self.candidate = nn.Sequential(
            nn.Linear(input_dim + state_dim, state_dim, bias=True),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor, hidden_prev: torch.Tensor) -> torch.Tensor:
        x = self.pre_norm(x)
        gate_input = torch.cat([x, hidden_prev], dim=1)
        update = self.update_gate(gate_input)
        reset = self.reset_gate(gate_input)
        candidate = self.candidate(torch.cat([x, reset * hidden_prev], dim=1))
        return update * hidden_prev + (1.0 - update) * candidate


class NotebookExactModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        horizon_size: int,
        context_size: int,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.rnn = NotebookExactGRUCell(input_dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, 2, bias=True),
        )

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        del future_xy, teacher_forcing_prob, scheduled_sampling_prob
        batch_size, _, num_entities, feature_dim = context.shape
        hidden_prev = torch.zeros(batch_size * num_entities, self.rnn.state_dim, device=context.device)
        static = context[:, 0, :, 2:].reshape(batch_size * num_entities, 2)
        total_steps = self.context_size + self.horizon_size
        predictions: list[torch.Tensor] = []
        current = torch.empty(0, device=context.device)
        for step in range(total_steps):
            if step < self.context_size:
                current = context[:, step].reshape(batch_size * num_entities, feature_dim)
            else:
                current = torch.cat([current, static], dim=1)
            hidden_prev = self.rnn(current, hidden_prev)
            if self.context_size - 1 <= step < total_steps - 1:
                current = self.proj(hidden_prev)
                predictions.append(current.reshape(batch_size, num_entities, 2))
        return torch.stack(predictions, dim=1)


class ReferenceGRUModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, horizon_size: int) -> None:
        super().__init__()
        self.horizon_size = horizon_size
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        del future_xy, teacher_forcing_prob, scheduled_sampling_prob
        batch_size, context_size, num_entities, feature_dim = context.shape
        static = _static_features(context).reshape(batch_size * num_entities, 2)
        hidden = torch.zeros(batch_size * num_entities, self.hidden_dim, device=context.device)
        previous_xy: torch.Tensor | None = None
        previous_previous_xy: torch.Tensor | None = None
        outputs = []
        for step in range(context_size + self.horizon_size - 1):
            if step < context_size:
                current_input = context[:, step].reshape(batch_size * num_entities, feature_dim)
            else:
                if previous_xy is None:
                    raise RuntimeError("Autoregressive rollout requires a previous prediction.")
                parts = [previous_xy]
                if feature_dim == 6:
                    velocity = torch.zeros_like(previous_xy) if previous_previous_xy is None else previous_xy - previous_previous_xy
                    parts.append(velocity)
                parts.append(static)
                current_input = torch.cat(parts, dim=-1)
            hidden = self.cell(current_input, hidden)
            if step >= context_size - 1:
                previous_previous_xy = previous_xy
                previous_xy = self.head(hidden)
                outputs.append(previous_xy.reshape(batch_size, num_entities, 2))
        return torch.stack(outputs, dim=1)


class JointGRUForecastModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_entities: int,
        context_size: int,
        horizon_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        del context_size
        self.num_entities = num_entities
        self.horizon_size = horizon_size
        self.encoder = nn.GRU(
            input_size=num_entities * input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_size * num_entities * 2),
        )

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        del future_xy, teacher_forcing_prob, scheduled_sampling_prob
        batch_size = context.shape[0]
        joint_context = context.reshape(batch_size, context.shape[1], -1)
        _, hidden = self.encoder(joint_context)
        decoded = self.head(hidden[-1])
        return decoded.reshape(batch_size, self.horizon_size, self.num_entities, 2)


class JointAutoregressiveGRUModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_entities: int,
        horizon_size: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        use_velocity: bool,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.horizon_size = horizon_size
        self.use_velocity = use_velocity
        self.encoder = nn.GRU(
            input_size=num_entities * input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.GRUCell(num_entities * input_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_entities * 2),
        )

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        batch_size = context.shape[0]
        static = _static_features(context)
        joint_context = context.reshape(batch_size, context.shape[1], -1)
        _, hidden = self.encoder(joint_context)
        hidden_state = hidden[-1]
        current_positions = context[:, -1, :, :2]
        current_velocity = context[:, -1, :, 2:4] if self.use_velocity else None
        outputs = []
        for step in range(self.horizon_size):
            parts = [current_positions]
            if self.use_velocity:
                if current_velocity is None:
                    current_velocity = torch.zeros_like(current_positions)
                parts.append(current_velocity)
            parts.append(static)
            decoder_input = torch.cat(parts, dim=-1).reshape(batch_size, -1)
            hidden_state = self.decoder(decoder_input, hidden_state)
            delta = self.head(hidden_state).reshape(batch_size, self.num_entities, 2)
            predicted_positions = current_positions + delta
            outputs.append(predicted_positions)
            if self.training and future_xy is not None:
                current_positions, current_velocity = _next_autoregressive_inputs(
                    current_positions=current_positions,
                    predicted_positions=predicted_positions,
                    target_positions=future_xy[:, step],
                    use_velocity=self.use_velocity,
                    teacher_forcing_prob=teacher_forcing_prob,
                    scheduled_sampling_prob=scheduled_sampling_prob,
                )
            else:
                current_velocity = predicted_positions - current_positions if self.use_velocity else None
                current_positions = predicted_positions
        return torch.stack(outputs, dim=1)
