from __future__ import annotations

import torch
from torch import nn

from .common import NUM_RELATION_TYPES, TypeConditionedDeltaHead, _next_autoregressive_inputs, _static_features
from .graph_layers import (
    _edge_features,
    _full_graph_mask,
    _knn_graph_mask,
    _make_message_layers,
    _radius_graph_mask,
    _relation_types,
    _semantic_graph_mask,
)


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
        horizon_tokens = self.horizon_embed(torch.arange(self.horizon_size, device=hidden_state.device)).view(
            1, self.horizon_size, 1, hidden_dim
        )
        entity_tokens = hidden_state.unsqueeze(1).expand(-1, self.horizon_size, -1, -1)
        query = self.query_proj(torch.cat([entity_tokens, horizon_tokens.expand(batch_size, -1, num_entities, -1)], dim=-1)).reshape(
            batch_size, self.horizon_size * num_entities, hidden_dim
        )
        attn_out, _ = self.cross_attn(query, hidden_state, hidden_state, need_weights=False)
        decoded = self.norm1(query + attn_out)
        decoded = self.norm2(decoded + self.ffn(decoded))
        decoded = decoded.reshape(batch_size * self.horizon_size, num_entities, hidden_dim)
        repeated_static = (
            static.unsqueeze(1)
            .expand(-1, self.horizon_size, -1, -1)
            .reshape(
                batch_size * self.horizon_size,
                num_entities,
                2,
            )
        )
        deltas = self.delta_head(decoded, repeated_static).reshape(batch_size, self.horizon_size, num_entities, 2)
        return last_positions.unsqueeze(1) + torch.cumsum(deltas, dim=1)


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
