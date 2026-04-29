from __future__ import annotations

import torch
from torch import nn

from .config import RunConfig


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


class SelectiveSSMLayer(nn.Module):
    def __init__(self, hidden_dim: int, conv_kernel: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 4)
        self.depthwise_conv = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=conv_kernel,
            padding=conv_kernel - 1,
            groups=hidden_dim,
        )
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        seq_len = sequence.shape[1]
        conv_input = sequence.transpose(1, 2)
        conv_output = self.depthwise_conv(conv_input)[..., :seq_len].transpose(1, 2)
        candidate, decay, input_gate, output_gate = self.in_proj(sequence + conv_output).chunk(4, dim=-1)
        candidate = torch.tanh(candidate)
        decay = torch.sigmoid(decay)
        input_gate = torch.sigmoid(input_gate)
        output_gate = torch.sigmoid(output_gate)
        state = torch.zeros(sequence.shape[0], sequence.shape[-1], device=sequence.device, dtype=sequence.dtype)
        outputs = []
        for step in range(seq_len):
            state = decay[:, step] * state + input_gate[:, step] * candidate[:, step]
            outputs.append(output_gate[:, step] * torch.tanh(state))
        return self.out_proj(self.dropout(torch.stack(outputs, dim=1)))


class CycleSSMBlock(nn.Module):
    def __init__(self, hidden_dim: int, conv_kernel: int, dropout: float) -> None:
        super().__init__()
        self.scan = SelectiveSSMLayer(hidden_dim, conv_kernel, dropout)
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        seq_len = sequence.shape[1]
        cycle_sequence = torch.cat([sequence, sequence.flip(1)], dim=1)
        cycle_output = self.scan(cycle_sequence)
        forward_pass = cycle_output[:, :seq_len]
        backward_pass = cycle_output[:, seq_len:].flip(1)
        fused = self.fuse(torch.cat([forward_pass, backward_pass], dim=-1))
        sequence = self.norm1(sequence + fused)
        return self.norm2(sequence + self.ffn(sequence))


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


def _radius_graph_mask(positions: torch.Tensor, radius: float) -> torch.Tensor:
    batch_size, num_entities, _ = positions.shape
    distances = torch.cdist(positions, positions)
    eye = torch.eye(num_entities, device=positions.device, dtype=torch.bool).unsqueeze(0)
    return (distances <= radius) & ~eye


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


def _relation_types(static: torch.Tensor) -> torch.Tensor:
    teams = static[:, :, 1]
    team_i = teams.unsqueeze(2)
    team_j = teams.unsqueeze(1)
    same_team = (team_i == team_j) & (team_i != 0) & (team_j != 0)
    involves_ball = (team_i == 0) | (team_j == 0)
    opponent = (~same_team) & (~involves_ball)
    relation = torch.zeros_like(same_team, dtype=torch.long)
    relation = relation.masked_fill(same_team, 1)
    relation = relation.masked_fill(opponent, 2)
    relation = relation.masked_fill(involves_ball, 3)
    num_entities = static.shape[1]
    eye = torch.eye(num_entities, device=static.device, dtype=torch.bool).unsqueeze(0)
    relation = relation.masked_fill(eye, 0)
    return relation


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


def _distance_sorted_social_inputs(
    node_state: torch.Tensor,
    positions: torch.Tensor,
    static: torch.Tensor,
    edge_proj: nn.Module,
    use_semantic_flags: bool,
) -> torch.Tensor:
    batch_size, num_entities, hidden_dim = node_state.shape
    edge_feat = _edge_features(positions, static, use_semantic_flags)
    order = torch.cdist(positions, positions).argsort(dim=-1)
    gather_node = order.unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
    gather_edge = order.unsqueeze(-1).expand(-1, -1, -1, edge_feat.shape[-1])
    sorted_nodes = torch.gather(node_state.unsqueeze(1).expand(-1, num_entities, -1, -1), 2, gather_node)
    sorted_edges = torch.gather(edge_feat, 2, gather_edge)
    return sorted_nodes + edge_proj(sorted_edges)


class MessagePassingLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        message_dim: int,
        dropout: float,
        num_relations: int = 0,
    ) -> None:
        super().__init__()
        self.num_relations = num_relations
        if num_relations > 0:
            self.relation_mlps = nn.ModuleList(
                MLP(2 * node_dim + edge_dim, message_dim, message_dim, dropout=dropout)
                for _ in range(num_relations)
            )
        else:
            self.message_mlp = MLP(2 * node_dim + edge_dim, message_dim, message_dim, dropout=dropout)
        self.update = nn.Sequential(
            nn.Linear(node_dim + message_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_feat: torch.Tensor,
        mask: torch.Tensor,
        relation_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_i = node_state.unsqueeze(2).expand(-1, -1, node_state.shape[1], -1)
        node_j = node_state.unsqueeze(1).expand(-1, node_state.shape[1], -1, -1)
        message_input = torch.cat([node_i, node_j, edge_feat], dim=-1)
        if self.num_relations > 0 and relation_ids is not None:
            messages = torch.zeros(
                node_state.shape[0],
                node_state.shape[1],
                node_state.shape[1],
                self.update[0].in_features - node_state.shape[-1],
                device=node_state.device,
                dtype=node_state.dtype,
            )
            for relation_idx, relation_mlp in enumerate(self.relation_mlps, start=1):
                relation_mask = mask & (relation_ids == relation_idx)
                relation_messages = relation_mlp(message_input)
                messages = messages + relation_messages * relation_mask.unsqueeze(-1)
        else:
            messages = self.message_mlp(message_input) * mask.unsqueeze(-1)
        denom = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(node_state.dtype)
        aggregated = messages.sum(dim=2) / denom
        updated = self.update(torch.cat([node_state, aggregated], dim=-1))
        return self.norm(node_state + updated)


class EdgeAwareAttentionLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        message_dim: int,
        dropout: float,
        num_relations: int = 0,
    ) -> None:
        super().__init__()
        self.query = nn.Linear(node_dim, message_dim)
        self.key = nn.Linear(node_dim, message_dim)
        self.value = nn.Linear(node_dim, message_dim)
        self.edge_key = nn.Linear(edge_dim, message_dim)
        self.edge_value = nn.Linear(edge_dim, message_dim)
        self.score = nn.Linear(message_dim, 1)
        self.update = nn.Sequential(
            nn.Linear(node_dim + message_dim, node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)
        self.scale = message_dim ** 0.5
        self.num_relations = num_relations
        if num_relations > 0:
            self.relation_key = nn.Embedding(num_relations + 1, message_dim)
            self.relation_value = nn.Embedding(num_relations + 1, message_dim)
            self.relation_bias = nn.Embedding(num_relations + 1, 1)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_feat: torch.Tensor,
        mask: torch.Tensor,
        relation_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        query = self.query(node_state).unsqueeze(2)
        key = self.key(node_state).unsqueeze(1)
        value = self.value(node_state).unsqueeze(1)
        edge_key = self.edge_key(edge_feat)
        edge_value = self.edge_value(edge_feat)
        logits = torch.tanh(query + key + edge_key)
        if self.num_relations > 0 and relation_ids is not None:
            logits = logits + self.relation_key(relation_ids)
        logits = self.score(logits).squeeze(-1) / self.scale
        if self.num_relations > 0 and relation_ids is not None:
            logits = logits + self.relation_bias(relation_ids).squeeze(-1)
        logits = logits.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(logits, dim=2)
        attention = torch.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
        aggregated_values = value + edge_value
        if self.num_relations > 0 and relation_ids is not None:
            aggregated_values = aggregated_values + self.relation_value(relation_ids)
        aggregated = torch.sum(attention.unsqueeze(-1) * aggregated_values, dim=2)
        updated = self.update(torch.cat([node_state, aggregated], dim=-1))
        return self.norm(node_state + updated)


def _make_message_layers(
    message_passing_type: str,
    node_dim: int,
    edge_dim: int,
    message_dim: int,
    num_layers: int,
    dropout: float,
    num_relations: int = 0,
) -> nn.ModuleList:
    layer_cls = MessagePassingLayer if message_passing_type == "mpnn" else EdgeAwareAttentionLayer
    return nn.ModuleList(
        layer_cls(node_dim, edge_dim, message_dim, dropout, num_relations=num_relations) for _ in range(num_layers)
    )


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
                    velocity = (
                        torch.zeros_like(previous_xy)
                        if previous_previous_xy is None
                        else previous_xy - previous_previous_xy
                    )
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


class HorizonQueryDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        horizon_size: int,
        dropout: float,
        num_heads: int,
        separate_ball_head: bool,
    ) -> None:
        super().__init__()
        self.horizon_size = horizon_size
        self.horizon_embed = nn.Embedding(horizon_size, hidden_dim)
        self.query_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.delta_head = TypeConditionedDeltaHead(hidden_dim, dropout, separate_ball_head)

    def forward(self, hidden_state: torch.Tensor, static: torch.Tensor, last_positions: torch.Tensor) -> torch.Tensor:
        batch_size, num_entities, hidden_dim = hidden_state.shape
        horizon_tokens = self.horizon_embed(
            torch.arange(self.horizon_size, device=hidden_state.device)
        ).view(1, self.horizon_size, 1, hidden_dim)
        entity_tokens = hidden_state.unsqueeze(1).expand(-1, self.horizon_size, -1, -1)
        query = self.query_proj(
            torch.cat([entity_tokens, horizon_tokens.expand(batch_size, -1, num_entities, -1)], dim=-1)
        ).reshape(batch_size, self.horizon_size * num_entities, hidden_dim)
        attn_out, _ = self.cross_attn(query, hidden_state, hidden_state, need_weights=False)
        decoded = self.norm1(query + attn_out)
        decoded = self.norm2(decoded + self.ffn(decoded))
        decoded = decoded.reshape(batch_size * self.horizon_size, num_entities, hidden_dim)
        repeated_static = static.unsqueeze(1).expand(-1, self.horizon_size, -1, -1).reshape(
            batch_size * self.horizon_size,
            num_entities,
            2,
        )
        deltas = self.delta_head(decoded, repeated_static).reshape(batch_size, self.horizon_size, num_entities, 2)
        return last_positions.unsqueeze(1) + torch.cumsum(deltas, dim=1)


class HorizonSSMDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        horizon_size: int,
        dropout: float,
        num_layers: int,
        conv_kernel: int,
        separate_ball_head: bool,
    ) -> None:
        super().__init__()
        self.horizon_size = horizon_size
        self.horizon_embed = nn.Embedding(horizon_size, hidden_dim)
        self.input_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layers = nn.ModuleList(
            CycleSSMBlock(hidden_dim=hidden_dim, conv_kernel=conv_kernel, dropout=dropout) for _ in range(num_layers)
        )
        self.delta_head = TypeConditionedDeltaHead(hidden_dim, dropout, separate_ball_head)

    def forward(self, hidden_state: torch.Tensor, static: torch.Tensor, last_positions: torch.Tensor) -> torch.Tensor:
        batch_size, num_entities, hidden_dim = hidden_state.shape
        horizon_tokens = self.horizon_embed(
            torch.arange(self.horizon_size, device=hidden_state.device)
        ).view(1, self.horizon_size, 1, hidden_dim)
        entity_tokens = hidden_state.unsqueeze(1).expand(-1, self.horizon_size, -1, -1)
        sequence = self.input_proj(
            torch.cat([entity_tokens, horizon_tokens.expand(batch_size, -1, num_entities, -1)], dim=-1)
        )
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch_size * num_entities, self.horizon_size, hidden_dim)
        for layer in self.layers:
            sequence = layer(sequence)
        decoded = sequence.reshape(batch_size, num_entities, self.horizon_size, hidden_dim).permute(0, 2, 1, 3)
        repeated_static = static.unsqueeze(1).expand(-1, self.horizon_size, -1, -1).reshape(
            batch_size * self.horizon_size,
            num_entities,
            2,
        )
        deltas = self.delta_head(
            decoded.reshape(batch_size * self.horizon_size, num_entities, hidden_dim),
            repeated_static,
        ).reshape(batch_size, self.horizon_size, num_entities, 2)
        return last_positions.unsqueeze(1) + torch.cumsum(deltas, dim=1)


class SocialSSMForecastModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_entities: int,
        horizon_size: int,
        hidden_dim: int,
        dropout: float,
        use_semantic_flags: bool,
        use_entity_embeddings: bool,
        entity_embedding_dim: int,
        temporal_ssm_layers: int,
        social_ssm_layers: int,
        decoder_ssm_layers: int,
        ssm_conv_kernel: int,
        separate_ball_head: bool,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.use_semantic_flags = use_semantic_flags
        self.use_entity_embeddings = use_entity_embeddings
        node_input_dim = input_dim + (entity_embedding_dim if use_entity_embeddings else 0)
        edge_dim = 5 if use_semantic_flags else 3
        if use_entity_embeddings:
            self.is_player_embedding = nn.Embedding(2, entity_embedding_dim)
            self.team_embedding = nn.Embedding(3, entity_embedding_dim)
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.temporal_layers = nn.ModuleList(
            CycleSSMBlock(hidden_dim=hidden_dim, conv_kernel=ssm_conv_kernel, dropout=dropout)
            for _ in range(temporal_ssm_layers)
        )
        self.social_edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.social_layers = nn.ModuleList(
            CycleSSMBlock(hidden_dim=hidden_dim, conv_kernel=ssm_conv_kernel, dropout=dropout)
            for _ in range(social_ssm_layers)
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.decoder = HorizonSSMDecoder(
            hidden_dim=hidden_dim,
            horizon_size=horizon_size,
            dropout=dropout,
            num_layers=decoder_ssm_layers,
            conv_kernel=ssm_conv_kernel,
            separate_ball_head=separate_ball_head,
        )

    def _entity_embedding(self, static: torch.Tensor) -> torch.Tensor:
        if not self.use_entity_embeddings:
            raise RuntimeError("Entity embeddings were requested while disabled.")
        is_player_idx = static[:, :, 0].long().clamp_(0, 1)
        team_idx = (static[:, :, 1].long() + 1).clamp_(0, 2)
        return self.is_player_embedding(is_player_idx) + self.team_embedding(team_idx)

    def _augment_features(self, node_features: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if not self.use_entity_embeddings:
            return node_features
        dynamic = node_features[:, :, :-2]
        return torch.cat([dynamic, static, self._entity_embedding(static)], dim=-1)

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        del future_xy, teacher_forcing_prob, scheduled_sampling_prob
        batch_size, context_size, num_entities, _ = context.shape
        static = _static_features(context)
        encoded = self.node_proj(
            self._augment_features(
                context.reshape(batch_size * context_size, num_entities, -1),
                static.unsqueeze(1).expand(-1, context_size, -1, -1).reshape(batch_size * context_size, num_entities, 2),
            )
        ).reshape(batch_size, context_size, num_entities, -1)
        temporal = encoded.permute(0, 2, 1, 3).reshape(batch_size * num_entities, context_size, -1)
        for layer in self.temporal_layers:
            temporal = layer(temporal)
        temporal_state = temporal[:, -1].reshape(batch_size, num_entities, -1)
        social = _distance_sorted_social_inputs(
            temporal_state,
            positions=context[:, -1, :, :2],
            static=static,
            edge_proj=self.social_edge_proj,
            use_semantic_flags=self.use_semantic_flags,
        ).reshape(batch_size * num_entities, num_entities, -1)
        for layer in self.social_layers:
            social = layer(social)
        social_state = social[:, 0].reshape(batch_size, num_entities, -1)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([temporal_state, social_state], dim=-1)))
        fused = self.fusion_norm(gate * temporal_state + (1.0 - gate) * social_state)
        return self.decoder(fused, static, context[:, -1, :, :2])


class StaticGraphDirectModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_entities: int,
        horizon_size: int,
        hidden_dim: int,
        message_dim: int,
        num_layers: int,
        dropout: float,
        use_semantic_flags: bool,
        message_passing_type: str,
        use_relation_types: bool,
        use_entity_embeddings: bool,
        entity_embedding_dim: int,
        edge_radius: float | None,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.horizon_size = horizon_size
        self.use_semantic_flags = use_semantic_flags
        self.use_relation_types = use_relation_types
        self.use_entity_embeddings = use_entity_embeddings
        self.edge_radius = edge_radius
        edge_dim = 5 if use_semantic_flags else 3
        node_input_dim = input_dim + (entity_embedding_dim if use_entity_embeddings else 0)
        if use_entity_embeddings:
            self.is_player_embedding = nn.Embedding(2, entity_embedding_dim)
            self.team_embedding = nn.Embedding(3, entity_embedding_dim)
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.layers = _make_message_layers(
            message_passing_type,
            hidden_dim,
            edge_dim,
            message_dim,
            num_layers,
            dropout,
            num_relations=NUM_RELATION_TYPES if use_relation_types else 0,
        )
        self.temporal = nn.GRU(
            input_size=num_entities * hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon_size * num_entities * 2),
        )

    def _entity_embedding(self, static: torch.Tensor) -> torch.Tensor:
        if not self.use_entity_embeddings:
            raise RuntimeError("Entity embeddings were requested while disabled.")
        is_player_idx = static[:, :, 0].long().clamp_(0, 1)
        team_idx = (static[:, :, 1].long() + 1).clamp_(0, 2)
        return self.is_player_embedding(is_player_idx) + self.team_embedding(team_idx)

    def _augment_features(self, node_features: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if not self.use_entity_embeddings:
            return node_features
        dynamic = node_features[:, :, :-2]
        return torch.cat([dynamic, static, self._entity_embedding(static)], dim=-1)

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        del future_xy, teacher_forcing_prob, scheduled_sampling_prob
        batch_size, context_size, num_entities, _ = context.shape
        static = _static_features(context)
        mask = _full_graph_mask(batch_size, num_entities, context.device)
        if self.edge_radius is not None:
            radius_mask = _radius_graph_mask(context[:, -1, :, :2], self.edge_radius)
            mask = mask & radius_mask
        relation_ids = _relation_types(static) if self.use_relation_types else None
        encoded_steps = []
        for step in range(context_size):
            positions = context[:, step, :, :2]
            edge_feat = _edge_features(positions, static, self.use_semantic_flags)
            step_mask = mask
            if self.edge_radius is not None:
                step_mask = step_mask & _radius_graph_mask(positions, self.edge_radius)
            node_state = self.node_proj(self._augment_features(context[:, step], static))
            for layer in self.layers:
                node_state = layer(node_state, edge_feat, step_mask, relation_ids)
            encoded_steps.append(node_state.reshape(batch_size, -1))
        temporal_input = torch.stack(encoded_steps, dim=1)
        _, hidden = self.temporal(temporal_input)
        decoded = self.head(hidden[-1])
        return decoded.reshape(batch_size, self.horizon_size, self.num_entities, 2)


class DynamicGraphBackbone(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_entities: int,
        hidden_dim: int,
        message_dim: int,
        num_layers: int,
        dropout: float,
        k_neighbors: int,
        use_semantic_flags: bool,
        dynamic_edges: bool,
        use_velocity: bool,
        message_passing_type: str,
        use_relation_types: bool,
        use_entity_embeddings: bool,
        entity_embedding_dim: int,
        edge_radius: float | None,
    ) -> None:
        super().__init__()
        self.num_entities = num_entities
        self.k_neighbors = k_neighbors
        self.use_semantic_flags = use_semantic_flags
        self.dynamic_edges = dynamic_edges
        self.use_velocity = use_velocity
        self.use_relation_types = use_relation_types
        self.use_entity_embeddings = use_entity_embeddings
        self.edge_radius = edge_radius
        edge_dim = 5 if use_semantic_flags else 3
        node_input_dim = input_dim + (entity_embedding_dim if use_entity_embeddings else 0)
        if use_entity_embeddings:
            self.is_player_embedding = nn.Embedding(2, entity_embedding_dim)
            self.team_embedding = nn.Embedding(3, entity_embedding_dim)
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.layers = _make_message_layers(
            message_passing_type,
            hidden_dim,
            edge_dim,
            message_dim,
            num_layers,
            dropout,
            num_relations=NUM_RELATION_TYPES if use_relation_types else 0,
        )
        self.node_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def _entity_embedding(self, static: torch.Tensor) -> torch.Tensor:
        if not self.use_entity_embeddings:
            raise RuntimeError("Entity embeddings were requested while disabled.")
        is_player_idx = static[:, :, 0].long().clamp_(0, 1)
        team_idx = (static[:, :, 1].long() + 1).clamp_(0, 2)
        return self.is_player_embedding(is_player_idx) + self.team_embedding(team_idx)

    def _augment_features(self, node_features: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if not self.use_entity_embeddings:
            return node_features
        dynamic = node_features[:, :, :-2]
        return torch.cat([dynamic, static, self._entity_embedding(static)], dim=-1)

    def _dynamic_mask(self, positions: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if not self.dynamic_edges:
            if self.use_semantic_flags:
                mask = _semantic_graph_mask(static)
            else:
                mask = _full_graph_mask(positions.shape[0], positions.shape[1], positions.device)
        else:
            mask = _knn_graph_mask(positions, self.k_neighbors)
            if self.use_semantic_flags:
                mask = mask | _semantic_graph_mask(static)
        if self.edge_radius is not None:
            mask = mask & _radius_graph_mask(positions, self.edge_radius)
        return mask

    def _encode_step(
        self,
        node_features: torch.Tensor,
        hidden_state: torch.Tensor,
        positions: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        node_state = self._graph_message_pass(node_features, positions, static)
        batch_size, num_entities, hidden_dim = node_state.shape
        encoded = self.node_gru(
            node_state.reshape(batch_size * num_entities, hidden_dim),
            hidden_state.reshape(batch_size * num_entities, hidden_dim),
        )
        return encoded.reshape(batch_size, num_entities, hidden_dim)

    def _graph_message_pass(
        self,
        node_features: torch.Tensor,
        positions: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        mask = self._dynamic_mask(positions, static)
        relation_ids = _relation_types(static) if self.use_relation_types else None
        edge_feat = _edge_features(positions, static, self.use_semantic_flags)
        node_state = self.node_proj(self._augment_features(node_features, static))
        for layer in self.layers:
            node_state = layer(node_state, edge_feat, mask, relation_ids)
        return node_state

    def _assemble_features(
        self,
        positions: torch.Tensor,
        velocity: torch.Tensor | None,
        static: torch.Tensor,
    ) -> torch.Tensor:
        parts = [positions]
        if self.use_velocity:
            if velocity is None:
                velocity = torch.zeros_like(positions)
            parts.append(velocity)
        parts.append(static)
        return torch.cat(parts, dim=-1)

    def encode_context(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, context_size, num_entities, _ = context.shape
        hidden_dim = self.node_gru.hidden_size
        hidden_state = torch.zeros(batch_size, num_entities, hidden_dim, device=context.device)
        static = _static_features(context)
        for step in range(context_size):
            node_features = context[:, step]
            positions = node_features[:, :, :2]
            hidden_state = self._encode_step(node_features, hidden_state, positions, static)
        return hidden_state, static


class DynamicGraphAutoregressiveModel(DynamicGraphBackbone):
    def __init__(
        self,
        input_dim: int,
        num_entities: int,
        horizon_size: int,
        hidden_dim: int,
        message_dim: int,
        num_layers: int,
        dropout: float,
        k_neighbors: int,
        use_semantic_flags: bool,
        dynamic_edges: bool,
        use_velocity: bool,
        message_passing_type: str,
        separate_ball_head: bool,
        use_relation_types: bool,
        use_entity_embeddings: bool,
        entity_embedding_dim: int,
        edge_radius: float | None,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_entities=num_entities,
            hidden_dim=hidden_dim,
            message_dim=message_dim,
            num_layers=num_layers,
            dropout=dropout,
            k_neighbors=k_neighbors,
            use_semantic_flags=use_semantic_flags,
            dynamic_edges=dynamic_edges,
            use_velocity=use_velocity,
            message_passing_type=message_passing_type,
            use_relation_types=use_relation_types,
            use_entity_embeddings=use_entity_embeddings,
            entity_embedding_dim=entity_embedding_dim,
            edge_radius=edge_radius,
        )
        self.horizon_size = horizon_size
        self.delta_head = TypeConditionedDeltaHead(hidden_dim, dropout, separate_ball_head)

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        hidden_state, static = self.encode_context(context)
        current_positions = context[:, -1, :, :2]
        current_velocity = context[:, -1, :, 2:4] if self.use_velocity else None
        outputs = []
        for step in range(self.horizon_size):
            node_features = self._assemble_features(current_positions, current_velocity, static)
            hidden_state = self._encode_step(node_features, hidden_state, current_positions, static)
            delta = self.delta_head(hidden_state, static)
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


class DynamicGraphDirectModel(DynamicGraphBackbone):
    def __init__(
        self,
        input_dim: int,
        num_entities: int,
        horizon_size: int,
        hidden_dim: int,
        message_dim: int,
        num_layers: int,
        dropout: float,
        k_neighbors: int,
        use_semantic_flags: bool,
        dynamic_edges: bool,
        use_velocity: bool,
        message_passing_type: str,
        separate_ball_head: bool,
        use_relation_types: bool,
        use_entity_embeddings: bool,
        entity_embedding_dim: int,
        edge_radius: float | None,
        direct_decoder_heads: int,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_entities=num_entities,
            hidden_dim=hidden_dim,
            message_dim=message_dim,
            num_layers=num_layers,
            dropout=dropout,
            k_neighbors=k_neighbors,
            use_semantic_flags=use_semantic_flags,
            dynamic_edges=dynamic_edges,
            use_velocity=use_velocity,
            message_passing_type=message_passing_type,
            use_relation_types=use_relation_types,
            use_entity_embeddings=use_entity_embeddings,
            entity_embedding_dim=entity_embedding_dim,
            edge_radius=edge_radius,
        )
        self.decoder = HorizonQueryDecoder(
            hidden_dim=hidden_dim,
            horizon_size=horizon_size,
            dropout=dropout,
            num_heads=direct_decoder_heads,
            separate_ball_head=separate_ball_head,
        )

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        del future_xy, teacher_forcing_prob, scheduled_sampling_prob
        hidden_state, static = self.encode_context(context)
        last_positions = context[:, -1, :, :2]
        return self.decoder(hidden_state, static, last_positions)


class GraphSSMDirectModel(DynamicGraphBackbone):
    def __init__(
        self,
        input_dim: int,
        num_entities: int,
        horizon_size: int,
        hidden_dim: int,
        message_dim: int,
        num_layers: int,
        dropout: float,
        k_neighbors: int,
        use_semantic_flags: bool,
        dynamic_edges: bool,
        use_velocity: bool,
        message_passing_type: str,
        separate_ball_head: bool,
        use_relation_types: bool,
        use_entity_embeddings: bool,
        entity_embedding_dim: int,
        edge_radius: float | None,
        temporal_ssm_layers: int,
        social_ssm_layers: int,
        decoder_ssm_layers: int,
        ssm_conv_kernel: int,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            num_entities=num_entities,
            hidden_dim=hidden_dim,
            message_dim=message_dim,
            num_layers=num_layers,
            dropout=dropout,
            k_neighbors=k_neighbors,
            use_semantic_flags=use_semantic_flags,
            dynamic_edges=dynamic_edges,
            use_velocity=use_velocity,
            message_passing_type=message_passing_type,
            use_relation_types=use_relation_types,
            use_entity_embeddings=use_entity_embeddings,
            entity_embedding_dim=entity_embedding_dim,
            edge_radius=edge_radius,
        )
        edge_dim = 5 if use_semantic_flags else 3
        self.temporal_layers = nn.ModuleList(
            CycleSSMBlock(hidden_dim=hidden_dim, conv_kernel=ssm_conv_kernel, dropout=dropout)
            for _ in range(temporal_ssm_layers)
        )
        self.social_edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.social_layers = nn.ModuleList(
            CycleSSMBlock(hidden_dim=hidden_dim, conv_kernel=ssm_conv_kernel, dropout=dropout)
            for _ in range(social_ssm_layers)
        )
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)
        self.decoder = HorizonSSMDecoder(
            hidden_dim=hidden_dim,
            horizon_size=horizon_size,
            dropout=dropout,
            num_layers=decoder_ssm_layers,
            conv_kernel=ssm_conv_kernel,
            separate_ball_head=separate_ball_head,
        )

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        del future_xy, teacher_forcing_prob, scheduled_sampling_prob
        batch_size, context_size, num_entities, _ = context.shape
        static = _static_features(context)
        encoded_steps = []
        for step in range(context_size):
            node_features = context[:, step]
            positions = node_features[:, :, :2]
            encoded_steps.append(self._graph_message_pass(node_features, positions, static))
        temporal = torch.stack(encoded_steps, dim=1).permute(0, 2, 1, 3).reshape(batch_size * num_entities, context_size, -1)
        for layer in self.temporal_layers:
            temporal = layer(temporal)
        temporal_state = temporal[:, -1].reshape(batch_size, num_entities, -1)
        social = _distance_sorted_social_inputs(
            temporal_state,
            positions=context[:, -1, :, :2],
            static=static,
            edge_proj=self.social_edge_proj,
            use_semantic_flags=self.use_semantic_flags,
        ).reshape(batch_size * num_entities, num_entities, -1)
        for layer in self.social_layers:
            social = layer(social)
        social_state = social[:, 0].reshape(batch_size, num_entities, -1)
        gate = torch.sigmoid(self.fusion_gate(torch.cat([temporal_state, social_state], dim=-1)))
        fused = self.fusion_norm(gate * temporal_state + (1.0 - gate) * social_state)
        return self.decoder(fused, static, context[:, -1, :, :2])


def build_model(config: RunConfig) -> nn.Module:
    experiment = config.experiment
    model_cfg = config.model
    model_name = experiment.model_name
    common_kwargs = {
        "input_dim": config.input_dim,
        "num_entities": experiment.num_entities,
        "horizon_size": experiment.horizon_size,
        "hidden_dim": model_cfg.hidden_dim,
        "message_dim": model_cfg.message_dim,
        "num_layers": model_cfg.num_layers,
        "dropout": model_cfg.dropout,
    }
    if model_name == "e0_reference":
        return ReferenceGRUModel(
            input_dim=config.input_dim,
            hidden_dim=model_cfg.hidden_dim,
            horizon_size=experiment.horizon_size,
        )
    if model_name == "e0_notebook_exact":
        return NotebookExactModel(
            input_dim=config.input_dim,
            hidden_dim=model_cfg.hidden_dim,
            horizon_size=experiment.horizon_size,
            context_size=experiment.context_size,
        )
    if model_name == "e1_joint_gru":
        return JointGRUForecastModel(
            input_dim=config.input_dim,
            num_entities=experiment.num_entities,
            context_size=experiment.context_size,
            horizon_size=experiment.horizon_size,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
        )
    if model_name == "e1_joint_autoreg_gru":
        return JointAutoregressiveGRUModel(
            input_dim=config.input_dim,
            num_entities=experiment.num_entities,
            horizon_size=experiment.horizon_size,
            hidden_dim=model_cfg.hidden_dim,
            num_layers=model_cfg.num_layers,
            dropout=model_cfg.dropout,
            use_velocity=config.features.use_velocity,
        )
    if model_name == "e1_social_ssm":
        return SocialSSMForecastModel(
            input_dim=config.input_dim,
            num_entities=experiment.num_entities,
            horizon_size=experiment.horizon_size,
            hidden_dim=model_cfg.hidden_dim,
            dropout=model_cfg.dropout,
            use_semantic_flags=model_cfg.use_semantic_flags,
            use_entity_embeddings=model_cfg.use_entity_embeddings,
            entity_embedding_dim=model_cfg.entity_embedding_dim,
            temporal_ssm_layers=model_cfg.temporal_ssm_layers,
            social_ssm_layers=model_cfg.social_ssm_layers,
            decoder_ssm_layers=model_cfg.decoder_ssm_layers,
            ssm_conv_kernel=model_cfg.ssm_conv_kernel,
            separate_ball_head=model_cfg.separate_ball_head,
        )
    if model_name == "e2_static_graph":
        return StaticGraphDirectModel(
            **common_kwargs,
            use_semantic_flags=model_cfg.use_semantic_flags,
            message_passing_type=model_cfg.message_passing_type,
            use_relation_types=model_cfg.use_relation_types,
            use_entity_embeddings=model_cfg.use_entity_embeddings,
            entity_embedding_dim=model_cfg.entity_embedding_dim,
            edge_radius=model_cfg.edge_radius,
        )
    if model_name in {"e3_dynamic_graph", "e3_graph_ssm"}:
        dynamic_kwargs = dict(
            **common_kwargs,
            k_neighbors=model_cfg.k_neighbors,
            use_semantic_flags=model_cfg.use_semantic_flags,
            dynamic_edges=model_cfg.dynamic_edges,
            use_velocity=config.features.use_velocity,
            message_passing_type=model_cfg.message_passing_type,
            separate_ball_head=model_cfg.separate_ball_head,
            use_relation_types=model_cfg.use_relation_types,
            use_entity_embeddings=model_cfg.use_entity_embeddings,
            entity_embedding_dim=model_cfg.entity_embedding_dim,
            edge_radius=model_cfg.edge_radius,
        )
    if model_name == "e3_dynamic_graph":
        if model_cfg.decoder_type == "direct":
            return DynamicGraphDirectModel(
                **dynamic_kwargs,
                direct_decoder_heads=model_cfg.direct_decoder_heads,
            )
        return DynamicGraphAutoregressiveModel(**dynamic_kwargs)
    if model_name == "e3_graph_ssm":
        return GraphSSMDirectModel(
            **dynamic_kwargs,
            temporal_ssm_layers=model_cfg.temporal_ssm_layers,
            social_ssm_layers=model_cfg.social_ssm_layers,
            decoder_ssm_layers=model_cfg.decoder_ssm_layers,
            ssm_conv_kernel=model_cfg.ssm_conv_kernel,
        )
    raise ValueError(f"Unknown model name {model_name!r}.")
