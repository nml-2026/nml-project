from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from .common import (
    MixtureDeltaHead,
    MixtureForecast,
    MixtureParams,
    TypeConditionedDeltaHead,
    _edge_features,
    _full_graph_mask,
    _knn_graph_mask,
    _semantic_graph_mask,
    _static_features,
    reduce_mixture,
)


class GraphMultiheadAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int,
        dropout: float,
        use_edge_bias: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}.")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.use_edge_bias = use_edge_bias
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.edge_bias = nn.Linear(edge_dim, num_heads)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_state: torch.Tensor, edge_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, num_entities, _ = node_state.shape
        query = self.query(node_state).reshape(batch_size, num_entities, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(node_state).reshape(batch_size, num_entities, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(node_state).reshape(batch_size, num_entities, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if self.use_edge_bias:
            logits = logits + self.edge_bias(edge_feat).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        attention = torch.nan_to_num(attention, nan=0.0, posinf=0.0, neginf=0.0)
        attended = torch.matmul(self.dropout(attention), value).transpose(1, 2).reshape(batch_size, num_entities, self.hidden_dim)
        return self.output(attended)


class GraphTransformerSpatialBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: float,
        use_edge_bias: bool = True,
    ) -> None:
        super().__init__()
        ffn_dim = int(hidden_dim * ffn_multiplier)
        self.attention = GraphMultiheadAttention(
            hidden_dim,
            edge_dim,
            num_heads,
            dropout,
            use_edge_bias=use_edge_bias,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, node_state: torch.Tensor, edge_feat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attention(node_state, edge_feat, mask)
        node_state = self.norm1(node_state + attn_out)
        ffn_out = self.ffn(node_state)
        return self.norm2(node_state + ffn_out)


class TemporalTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: float,
    ) -> None:
        super().__init__()
        ffn_dim = int(hidden_dim * ffn_multiplier)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        sequence = self.norm1(sequence + attended)
        ffn_out = self.ffn(sequence)
        return self.norm2(sequence + ffn_out)


class VirtualNodeRouter(nn.Module):
    """Learned global nodes that exchange context with all real entities."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        ffn_multiplier: float,
        num_virtual_nodes: int,
    ) -> None:
        super().__init__()
        if num_virtual_nodes < 1:
            raise ValueError("num_virtual_nodes must be >= 1.")
        self.num_virtual_nodes = num_virtual_nodes
        ffn_dim = int(hidden_dim * ffn_multiplier)
        self.virtual_nodes = nn.Parameter(torch.randn(num_virtual_nodes, hidden_dim) * (hidden_dim**-0.5))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, node_state: torch.Tensor) -> torch.Tensor:
        batch_size = node_state.shape[0]
        virtual = self.virtual_nodes.unsqueeze(0).expand(batch_size, -1, -1)
        augmented = torch.cat([virtual, node_state], dim=1)
        attended, _ = self.attention(augmented, augmented, augmented, need_weights=False)
        augmented = self.norm1(augmented + attended)
        augmented = self.norm2(augmented + self.ffn(augmented))
        return augmented[:, self.num_virtual_nodes :]


class GraphTransformerForecastModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_entities: int,
        context_size: int,
        horizon_size: int,
        hidden_dim: int,
        dropout: float,
        k_neighbors: int,
        use_semantic_flags: bool,
        dynamic_edges: bool,
        separate_ball_head: bool,
        use_entity_embeddings: bool,
        entity_embedding_dim: int,
        transformer_heads: int,
        spatial_layers: int,
        temporal_layers: int,
        decoder_layers: int,
        transformer_ffn_multiplier: float,
        graph_ablation_mode: str,
        num_virtual_nodes: int = 0,
        head_type: str = "delta",
        mdn_components: int = 1,
        mdn_point_estimate: str = "mean",
        oracle_student_head: bool = False,
    ) -> None:
        super().__init__()
        if graph_ablation_mode not in {"normal", "no_graph", "shuffled_graph"}:
            raise ValueError("graph_ablation_mode must be one of: normal, no_graph, shuffled_graph.")
        if head_type not in {"delta", "mdn"}:
            raise ValueError("head_type must be one of: delta, mdn.")
        if mdn_point_estimate not in {"mean", "mode", "medoid", "student"}:
            raise ValueError("mdn_point_estimate must be one of: mean, mode, medoid, student.")
        if mdn_point_estimate == "student" and not oracle_student_head:
            raise ValueError("mdn_point_estimate='student' requires model.head.oracle_student_head=true.")
        if num_virtual_nodes < 0:
            raise ValueError("model.graph_transformer.num_virtual_nodes must be >= 0.")
        self.head_type = head_type
        self.mdn_components = mdn_components
        self.mdn_point_estimate = mdn_point_estimate
        self.num_virtual_nodes = num_virtual_nodes
        self.num_entities = num_entities
        self.horizon_size = horizon_size
        self.k_neighbors = k_neighbors
        self.use_semantic_flags = use_semantic_flags
        self.dynamic_edges = dynamic_edges
        self.use_entity_embeddings = use_entity_embeddings
        self.graph_ablation_mode = graph_ablation_mode
        edge_dim = 5 if use_semantic_flags else 3
        node_input_dim = input_dim + (entity_embedding_dim if use_entity_embeddings else 0)
        if use_entity_embeddings:
            self.is_player_embedding = nn.Embedding(2, entity_embedding_dim)
            self.team_embedding = nn.Embedding(3, entity_embedding_dim)
        self.node_proj = nn.Linear(node_input_dim, hidden_dim)
        self.time_embed = nn.Embedding(context_size, hidden_dim)
        self.horizon_embed = nn.Embedding(horizon_size, hidden_dim)
        use_edge_bias = graph_ablation_mode != "no_graph"
        self.spatial_layers = nn.ModuleList(
            GraphTransformerSpatialBlock(
                hidden_dim,
                edge_dim,
                transformer_heads,
                dropout,
                transformer_ffn_multiplier,
                use_edge_bias=use_edge_bias,
            )
            for _ in range(spatial_layers)
        )
        self.temporal_layers = nn.ModuleList(
            TemporalTransformerBlock(
                hidden_dim,
                transformer_heads,
                dropout,
                transformer_ffn_multiplier,
            )
            for _ in range(temporal_layers)
        )
        self.decoder_layers = nn.ModuleList(
            GraphTransformerSpatialBlock(
                hidden_dim,
                edge_dim,
                transformer_heads,
                dropout,
                transformer_ffn_multiplier,
                use_edge_bias=use_edge_bias,
            )
            for _ in range(decoder_layers)
        )
        self.decoder_input = nn.Linear(hidden_dim * 2, hidden_dim)
        self.virtual_router = (
            VirtualNodeRouter(hidden_dim, transformer_heads, dropout, transformer_ffn_multiplier, num_virtual_nodes)
            if num_virtual_nodes > 0
            else None
        )
        if head_type == "mdn":
            self.delta_head = MixtureDeltaHead(hidden_dim, dropout, mdn_components)
            self.oracle_student_head = TypeConditionedDeltaHead(hidden_dim, dropout, separate_ball_head) if oracle_student_head else None
        else:
            self.delta_head = TypeConditionedDeltaHead(hidden_dim, dropout, separate_ball_head)
            self.oracle_student_head = None

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

    def _base_graph_mask(self, positions: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        if self.graph_ablation_mode == "no_graph":
            return _full_graph_mask(positions.shape[0], positions.shape[1], positions.device)
        if not self.dynamic_edges:
            if self.use_semantic_flags:
                mask = _semantic_graph_mask(static)
            else:
                mask = _full_graph_mask(positions.shape[0], positions.shape[1], positions.device)
        else:
            mask = _knn_graph_mask(positions, self.k_neighbors)
            if self.use_semantic_flags:
                mask = mask | _semantic_graph_mask(static)
        if self.graph_ablation_mode == "shuffled_graph":
            permutation = torch.arange(positions.shape[1], device=positions.device).roll(1)
            mask = mask.index_select(2, permutation)
            eye = torch.eye(positions.shape[1], device=positions.device, dtype=torch.bool).unsqueeze(0)
            mask = mask & ~eye
        return mask

    def graph_inputs(self, positions: torch.Tensor, static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self._base_graph_mask(positions, static)
        edge_feat = _edge_features(positions, static, self.use_semantic_flags)
        if self.graph_ablation_mode == "no_graph":
            edge_feat = torch.zeros_like(edge_feat)
        return edge_feat, mask

    def _apply_spatial_layer(
        self,
        encoded: torch.Tensor,
        positions: torch.Tensor,
        static: torch.Tensor,
        layer: GraphTransformerSpatialBlock,
        edge_inputs: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size, context_size, num_entities, hidden_dim = encoded.shape
        flat_state = encoded.reshape(batch_size * context_size, num_entities, hidden_dim)
        if edge_inputs is None:
            flat_positions = positions.reshape(batch_size * context_size, num_entities, 2)
            flat_static = static.unsqueeze(1).expand(-1, context_size, -1, -1).reshape(batch_size * context_size, num_entities, 2)
            edge_feat, mask = self.graph_inputs(flat_positions, flat_static)
        else:
            edge_feat, mask = edge_inputs
        flat_state = layer(flat_state, edge_feat, mask)
        if self.virtual_router is not None:
            flat_state = self.virtual_router(flat_state)
        return flat_state.reshape(batch_size, context_size, num_entities, hidden_dim)

    def _apply_temporal_layer(self, encoded: torch.Tensor, layer: TemporalTransformerBlock) -> torch.Tensor:
        batch_size, context_size, num_entities, hidden_dim = encoded.shape
        temporal = encoded.permute(0, 2, 1, 3).reshape(batch_size * num_entities, context_size, hidden_dim)
        temporal = layer(temporal)
        return temporal.reshape(batch_size, num_entities, context_size, hidden_dim).permute(0, 2, 1, 3)

    def _encode_decode(self, context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared encoder + horizon decoder, head-agnostic.

        Returns ``(flat_decoded [B*T, N, H], entity_state [B, N, H], static [B, N, 2],
        last_pos [B, N, 2])``. Both the deterministic delta head and the MDN head consume
        these; factoring it keeps the two heads byte-identical up to the final projection.
        """
        batch_size, context_size, num_entities, _ = context.shape
        static = _static_features(context)
        flat_static = static.unsqueeze(1).expand(-1, context_size, -1, -1).reshape(batch_size * context_size, num_entities, 2)
        encoded = self.node_proj(self._augment_features(context.reshape(batch_size * context_size, num_entities, -1), flat_static)).reshape(
            batch_size, context_size, num_entities, -1
        )
        time_tokens = self.time_embed(torch.arange(context_size, device=context.device)).view(1, context_size, 1, -1)
        encoded = encoded + time_tokens
        layer_count = max(len(self.spatial_layers), len(self.temporal_layers))
        encoder_edge_inputs = None
        if self.spatial_layers:
            flat_positions = context[:, :, :, :2].reshape(batch_size * context_size, num_entities, 2)
            encoder_edge_inputs = self.graph_inputs(flat_positions, flat_static)
        for layer_idx in range(layer_count):
            if layer_idx < len(self.spatial_layers):
                spatial_layer: Any = self.spatial_layers[layer_idx]
                encoded = self._apply_spatial_layer(
                    encoded,
                    context[:, :, :, :2],
                    static,
                    spatial_layer,
                    edge_inputs=encoder_edge_inputs,
                )
            if layer_idx < len(self.temporal_layers):
                temporal_layer: Any = self.temporal_layers[layer_idx]
                encoded = self._apply_temporal_layer(encoded, temporal_layer)
        entity_state = encoded[:, -1]
        horizon_tokens = self.horizon_embed(torch.arange(self.horizon_size, device=context.device)).view(1, self.horizon_size, 1, -1)
        entity_tokens = entity_state.unsqueeze(1).expand(-1, self.horizon_size, -1, -1)
        decoded = self.decoder_input(torch.cat([entity_tokens, horizon_tokens.expand(batch_size, -1, num_entities, -1)], dim=-1))
        flat_decoded = decoded.reshape(batch_size * self.horizon_size, num_entities, -1)
        edge_feat_once, mask_once = self.graph_inputs(context[:, -1, :, :2], static)
        edge_feat = (
            edge_feat_once.unsqueeze(1)
            .expand(-1, self.horizon_size, -1, -1, -1)
            .reshape(batch_size * self.horizon_size, num_entities, num_entities, -1)
        )
        mask = (
            mask_once.unsqueeze(1).expand(-1, self.horizon_size, -1, -1).reshape(batch_size * self.horizon_size, num_entities, num_entities)
        )
        for layer in self.decoder_layers:
            flat_decoded = layer(flat_decoded, edge_feat, mask)
            if self.virtual_router is not None:
                flat_decoded = self.virtual_router(flat_decoded)
        return flat_decoded, entity_state, static, context[:, -1, :, :2]

    def _mixture_params(
        self,
        flat_decoded: torch.Tensor,
        entity_state: torch.Tensor,
        last_pos: torch.Tensor,
    ) -> MixtureParams:
        batch_size, num_entities, _ = entity_state.shape
        mu_delta, log_sigma, rho, log_pi = self.delta_head(flat_decoded, entity_state, batch_size, self.horizon_size)
        # Each mode's coherent trajectory = last position + cumsum of its per-step deltas.
        means = last_pos.view(batch_size, 1, num_entities, 1, 2) + torch.cumsum(mu_delta, dim=1)
        return MixtureParams(means=means, log_sigma=log_sigma, rho=rho, log_pi=log_pi)

    def mixture_forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> MixtureParams | MixtureForecast:
        """Full Gaussian-mixture parameters (MDN head only). Used by the NLL training loss
        and the minADE_K evaluation pass; the deterministic pipeline uses ``forward``."""
        del future_xy, teacher_forcing_prob, scheduled_sampling_prob
        if self.head_type != "mdn":
            raise RuntimeError("mixture_forward is only available when head_type == 'mdn'.")
        flat_decoded, entity_state, static, last_pos = self._encode_decode(context)
        params = self._mixture_params(flat_decoded, entity_state, last_pos)
        if self.oracle_student_head is None:
            return params
        student = self._oracle_student_prediction(flat_decoded, static, last_pos)
        return MixtureForecast(params=params, student=student)

    def _oracle_student_prediction(
        self,
        flat_decoded: torch.Tensor,
        static: torch.Tensor,
        last_pos: torch.Tensor,
    ) -> torch.Tensor:
        if self.oracle_student_head is None:
            raise RuntimeError("Oracle student head is disabled.")
        batch_size, num_entities, _ = static.shape
        decoder_static = static.unsqueeze(1).expand(-1, self.horizon_size, -1, -1)
        repeated_static = decoder_static.reshape(batch_size * self.horizon_size, num_entities, 2)
        deltas = self.oracle_student_head(flat_decoded, repeated_static).reshape(batch_size, self.horizon_size, num_entities, 2)
        return last_pos.unsqueeze(1) + torch.cumsum(deltas, dim=1)

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        del future_xy, teacher_forcing_prob, scheduled_sampling_prob
        batch_size, _, num_entities, _ = context.shape
        flat_decoded, entity_state, static, last_pos = self._encode_decode(context)
        if self.head_type == "mdn":
            params = self._mixture_params(flat_decoded, entity_state, last_pos)
            if self.mdn_point_estimate == "student":
                return self._oracle_student_prediction(flat_decoded, static, last_pos)
            return reduce_mixture(params.means, params.log_pi, self.mdn_point_estimate)
        decoder_static = static.unsqueeze(1).expand(-1, self.horizon_size, -1, -1)
        repeated_static = decoder_static.reshape(batch_size * self.horizon_size, num_entities, 2)
        deltas = self.delta_head(flat_decoded, repeated_static).reshape(batch_size, self.horizon_size, num_entities, 2)
        return last_pos.unsqueeze(1) + torch.cumsum(deltas, dim=1)


from ..config import GraphTransformerModelConfig, RunConfig  # noqa: E402


def build_graph_transformer(config: RunConfig) -> GraphTransformerForecastModel:
    if not isinstance(config.model, GraphTransformerModelConfig):
        raise TypeError("graph_transformer requires GraphTransformerModelConfig.")
    experiment = config.experiment
    common = config.model.common
    graph = config.model.graph
    transformer = config.model.graph_transformer
    head = config.model.head
    return GraphTransformerForecastModel(
        input_dim=config.input_dim,
        num_entities=experiment.num_entities,
        context_size=experiment.context_size,
        horizon_size=experiment.horizon_size,
        hidden_dim=common.hidden_dim,
        dropout=common.dropout,
        k_neighbors=graph.k_neighbors,
        use_semantic_flags=graph.use_semantic_flags,
        dynamic_edges=graph.dynamic_edges,
        separate_ball_head=graph.separate_ball_head,
        use_entity_embeddings=graph.use_entity_embeddings,
        entity_embedding_dim=graph.entity_embedding_dim,
        transformer_heads=transformer.transformer_heads,
        spatial_layers=transformer.spatial_layers,
        temporal_layers=transformer.temporal_layers,
        decoder_layers=transformer.decoder_layers,
        transformer_ffn_multiplier=transformer.transformer_ffn_multiplier,
        graph_ablation_mode=transformer.graph_ablation_mode,
        num_virtual_nodes=transformer.num_virtual_nodes,
        head_type=head.head_type,
        mdn_components=head.mdn_components,
        mdn_point_estimate=head.mdn_point_estimate,
        oracle_student_head=head.oracle_student_head,
    )
