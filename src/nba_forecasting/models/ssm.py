from __future__ import annotations

import torch
from torch import nn

from .common import TypeConditionedDeltaHead, _static_features
from .graph import DynamicGraphBackbone
from .graph_layers import _distance_sorted_social_inputs
from .ssm_blocks import CycleSSMBlock


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
        horizon_tokens = self.horizon_embed(torch.arange(self.horizon_size, device=hidden_state.device)).view(
            1, self.horizon_size, 1, hidden_dim
        )
        entity_tokens = hidden_state.unsqueeze(1).expand(-1, self.horizon_size, -1, -1)
        sequence = self.input_proj(torch.cat([entity_tokens, horizon_tokens.expand(batch_size, -1, num_entities, -1)], dim=-1))
        sequence = sequence.permute(0, 2, 1, 3).reshape(batch_size * num_entities, self.horizon_size, hidden_dim)
        for layer in self.layers:
            sequence = layer(sequence)
        decoded = sequence.reshape(batch_size, num_entities, self.horizon_size, hidden_dim).permute(0, 2, 1, 3)
        repeated_static = (
            static.unsqueeze(1)
            .expand(-1, self.horizon_size, -1, -1)
            .reshape(
                batch_size * self.horizon_size,
                num_entities,
                2,
            )
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
            CycleSSMBlock(hidden_dim=hidden_dim, conv_kernel=ssm_conv_kernel, dropout=dropout) for _ in range(temporal_ssm_layers)
        )
        self.social_edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.social_layers = nn.ModuleList(
            CycleSSMBlock(hidden_dim=hidden_dim, conv_kernel=ssm_conv_kernel, dropout=dropout) for _ in range(social_ssm_layers)
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
            CycleSSMBlock(hidden_dim=hidden_dim, conv_kernel=ssm_conv_kernel, dropout=dropout) for _ in range(temporal_ssm_layers)
        )
        self.social_edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.social_layers = nn.ModuleList(
            CycleSSMBlock(hidden_dim=hidden_dim, conv_kernel=ssm_conv_kernel, dropout=dropout) for _ in range(social_ssm_layers)
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
