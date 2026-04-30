from __future__ import annotations

import torch
from torch import nn

from .common import MLP


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
                MLP(2 * node_dim + edge_dim, message_dim, message_dim, dropout=dropout) for _ in range(num_relations)
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
        self.scale = message_dim**0.5
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
    return nn.ModuleList(layer_cls(node_dim, edge_dim, message_dim, dropout, num_relations=num_relations) for _ in range(num_layers))
