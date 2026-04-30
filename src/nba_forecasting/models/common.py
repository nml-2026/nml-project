from __future__ import annotations

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


class TypeConditionedDeltaHead(nn.Module):
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
