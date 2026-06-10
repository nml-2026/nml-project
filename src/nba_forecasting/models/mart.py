"""MART-style hypergraph Transformer blocks.

This module adapts substantial portions of the official MART implementation:
https://github.com/gist-ailab/MART

MIT License
Copyright (c) 2024 GIST-AILAB

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ..config import MartModelConfig, RunConfig
from .common import ForecasterHead, ForecastOutput, MixtureParams


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: tuple[int, ...] = (1024, 512),
        dropout: float = -1.0,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_size, output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx != len(self.layers) - 1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = F.dropout(x, p=min(0.1, self.dropout / 3) if idx == 1 else self.dropout, training=self.training)
        return x


class PositionalAgentEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, dropout: float = 0.1, max_t_len: int = 200, concat: bool = True) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.concat = concat
        self.d_model = d_model
        if concat:
            self.fc = nn.Linear(2 * d_model, d_model)
        self.register_buffer("pe", self.build_pos_enc(max_t_len), persistent=False)

    def build_pos_enc(self, max_len: int) -> torch.Tensor:
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        return pe

    def forward(self, x: torch.Tensor, num_a: int, t_offset: int = 0) -> torch.Tensor:
        num_t = x.shape[1]
        pos_enc = self.pe[t_offset : num_t + t_offset].unsqueeze(0).repeat(num_a, 1, 1)
        if self.concat:
            x = self.fc(torch.cat([x, pos_enc], dim=-1))
        else:
            x = x + pos_enc
        return self.dropout(x)


class RTTransformerLayer(nn.Module):
    def __init__(
        self,
        attention_layer: nn.Module,
        node_dim: int = 64,
        node_hidden_dim: int = 128,
        edge_dim: int = 64,
        edge_hidden_dim_1: int = 128,
        edge_hidden_dim_2: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attention_layer = attention_layer
        self.dropout = nn.Dropout(dropout)
        self.linear_net_n = nn.Sequential(nn.Linear(node_dim, node_hidden_dim), nn.ReLU(inplace=True), nn.Linear(node_hidden_dim, node_dim))
        self.norm1_n = nn.LayerNorm(node_dim)
        self.norm2_n = nn.LayerNorm(node_dim)
        self.linear_net1_e = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim * 2, edge_hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Linear(edge_hidden_dim_1, edge_dim),
        )
        self.linear_net2_e = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim_2), nn.ReLU(inplace=True), nn.Linear(edge_hidden_dim_2, edge_dim)
        )
        self.norm1_e = nn.LayerNorm(edge_dim)
        self.norm2_e = nn.LayerNorm(edge_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = self.norm1_n(node_features + self.dropout(self.attention_layer(node_features, edge_features)))
        node_features = self.norm2_n(node_features + self.dropout(self.linear_net_n(node_features)))

        lin0: Any = self.linear_net1_e[0]
        activation: Any = self.linear_net1_e[1]
        lin1: Any = self.linear_net1_e[2]
        edge_dim = edge_features.shape[-1]
        node_dim = node_features.shape[-1]
        weight = lin0.weight
        bias = lin0.bias
        w_edge = weight[:, :edge_dim]
        w_reverse = weight[:, edge_dim : 2 * edge_dim]
        w_source = weight[:, 2 * edge_dim : 2 * edge_dim + node_dim]
        w_target = weight[:, 2 * edge_dim + node_dim :]

        hidden = F.linear(edge_features, w_edge, bias)
        hidden = hidden + F.linear(edge_features.transpose(1, 2), w_reverse, None)
        hidden = hidden + F.linear(node_features, w_source, None).unsqueeze(1)
        hidden = hidden + F.linear(node_features, w_target, None).unsqueeze(2)
        edge_features = self.norm1_e(edge_features + self.dropout(lin1(activation(hidden))))
        edge_features = self.norm2_e(edge_features + self.dropout(self.linear_net2_e(edge_features)))
        return node_features, edge_features


class MARTAttentionLayer(nn.Module):
    def __init__(self, num_heads: int, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        if node_dim % num_heads != 0:
            raise ValueError(f"node_dim={node_dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.proj_qkv_n = nn.Linear(node_dim, 3 * node_dim)
        self.proj_qkv_e = nn.Linear(edge_dim, 3 * node_dim)
        self.proj_o = nn.Linear(node_dim, node_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_features.shape
        qkv_n = self.proj_qkv_n(node_features)
        qkv_e = self.proj_qkv_e(edge_features)

        qkv_n = qkv_n.reshape(batch_size, num_nodes, self.num_heads, 3 * self.head_dim).permute(0, 2, 1, 3)
        q_n, k_n, v_n = qkv_n.chunk(3, dim=-1)

        qkv_e = qkv_e.reshape(batch_size, num_nodes, num_nodes, self.num_heads, 3 * self.head_dim).permute(0, 3, 1, 2, 4)
        q_e, k_e, v_e = qkv_e.chunk(3, dim=-1)

        logits = torch.einsum("bhid,bhjd->bhij", q_n, k_n)
        logits = logits + (q_n.unsqueeze(3) * k_e).sum(dim=-1)
        logits = logits + (q_e * k_n.unsqueeze(2)).sum(dim=-1)
        logits = logits + (q_e * k_e).sum(dim=-1)
        att_dist = F.softmax(logits * self.scale, dim=-1)

        node_part = torch.matmul(att_dist, v_n)
        edge_part = (att_dist.unsqueeze(-1) * v_e).sum(dim=3)
        new_nodes = (node_part + edge_part).permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.num_heads * self.head_dim)
        return self.proj_o(new_nodes)


class MARTRT(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        num_heads: int = 4,
        node_dim: int = 64,
        node_hidden_dim: int = 128,
        edge_dim: int = 64,
        edge_hidden_dim_1: int = 128,
        edge_hidden_dim_2: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer = RTTransformerLayer(
            MARTAttentionLayer(num_heads, node_dim, edge_dim),
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.node2edge_mlp = MLP(input_dim=node_dim * 2, output_dim=node_dim, hidden_size=(node_hidden_dim,))

    def init_edge(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        receivers = x.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, -1)
        senders = x.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, -1)
        return self.node2edge_mlp(torch.cat([receivers, senders], dim=-1))

    def forward(self, node_features: torch.Tensor, edge_features_: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        del edge_features_
        edge_features = self.init_edge(node_features)
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features)
        return node_features, edge_features


class MARTRTNoEdgeInit(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        num_heads: int = 4,
        node_dim: int = 64,
        node_hidden_dim: int = 128,
        edge_dim: int = 64,
        edge_hidden_dim_1: int = 128,
        edge_hidden_dim_2: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer = RTTransformerLayer(
            MARTAttentionLayer(num_heads, node_dim, edge_dim),
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features)
        return node_features, edge_features


class BinaryThresholdFunctionType1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output: torch.Tensor = grad_outputs[0]
        (input_tensor,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[torch.abs(input_tensor) > 0.5] = 0
        return grad_input


class BinaryThresholdFunctionType2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output: torch.Tensor = grad_outputs[0]
        (input_tensor,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        close = torch.abs(input_tensor) <= 0.5
        grad_input[close] *= 2 - 4.0 * torch.abs(input_tensor[close])
        grad_input[~close] = 0
        return grad_input


class BinaryThresholdFunctionType3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input_tensor)
        return (input_tensor > 0).float()

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output: torch.Tensor = grad_outputs[0]
        (input_tensor,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        close = torch.abs(input_tensor) <= 0.4
        mid = torch.logical_and(torch.abs(input_tensor) > 0.4, torch.abs(input_tensor) <= 1)
        grad_input[close] *= 2 - 4.0 * torch.abs(input_tensor[close])
        grad_input[mid] *= 0.4
        grad_input[torch.abs(input_tensor) > 1] = 0
        return grad_input


class BinaryThreshold(nn.Module):
    def __init__(self, function_type: int = 1) -> None:
        super().__init__()
        self.function_type = function_type

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.function_type == 1:
            return BinaryThresholdFunctionType1.apply(input_tensor)
        if self.function_type == 2:
            return BinaryThresholdFunctionType2.apply(input_tensor)
        if self.function_type == 3:
            return BinaryThresholdFunctionType3.apply(input_tensor)
        raise NotImplementedError


class AdaptiveGroupEstimator(nn.Module):
    def __init__(self, function_type: int = 1) -> None:
        super().__init__()
        self.th = nn.Parameter(torch.tensor([0.5]))
        self.ste = BinaryThreshold(function_type=function_type)

    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        temp = F.normalize(node_features, p=2, dim=2)
        corr_mat = torch.matmul(temp, temp.permute(0, 2, 1))
        return self.ste(corr_mat - self.th.clamp(-0.9999, 0.9999))


class HRTAttentionLayer(nn.Module):
    def __init__(self, num_heads: int, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        if node_dim % num_heads != 0:
            raise ValueError(f"node_dim={node_dim} must be divisible by num_heads={num_heads}.")
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.proj_qkv_n = nn.Linear(node_dim, 3 * node_dim)
        self.proj_qkv_e = nn.Linear(edge_dim, 3 * node_dim)
        self.proj_o = nn.Linear(node_dim, node_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = node_features.shape
        _, num_edges, _ = edge_features.shape
        qkv_n = self.proj_qkv_n(node_features)
        qkv_e = self.proj_qkv_e(edge_features)

        qkv_n = qkv_n.reshape(batch_size, num_nodes, self.num_heads, 3 * self.head_dim).permute(0, 2, 1, 3)
        q_n, k_n, v_n = qkv_n.chunk(3, dim=-1)

        qkv_e = qkv_e.reshape(batch_size, num_edges, self.num_heads, 3 * self.head_dim).permute(0, 2, 1, 3)
        q_e, k_e, v_e = qkv_e.chunk(3, dim=-1)

        div = torch.sum(group, dim=1)[:, None, :, None].clamp_min(1.0)
        q = q_n + torch.einsum("bmn,bhnc->bhmc", group.transpose(-2, -1), q_e) / div
        k = k_n + torch.einsum("bmn,bhnc->bhmc", group.transpose(-2, -1), k_e) / div
        value = v_n + torch.einsum("bmn,bhnc->bhmc", group.transpose(-2, -1), v_e) / div

        new_nodes = F.scaled_dot_product_attention(q, k, value, dropout_p=0.0, scale=self.scale)
        new_nodes = new_nodes.permute(0, 2, 1, 3).reshape(batch_size, num_nodes, self.num_heads * self.head_dim)
        return self.proj_o(new_nodes)


class HRTTransformerLayer(nn.Module):
    def __init__(
        self,
        num_heads: int = 4,
        node_dim: int = 64,
        node_hidden_dim: int = 128,
        edge_dim: int = 64,
        edge_hidden_dim_1: int = 128,
        edge_hidden_dim_2: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.node_attention_layer = HRTAttentionLayer(num_heads, node_dim, edge_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_net_n = nn.Sequential(nn.Linear(node_dim, node_hidden_dim), nn.ReLU(inplace=True), nn.Linear(node_hidden_dim, node_dim))
        self.norm1_n = nn.LayerNorm(node_dim)
        self.norm2_n = nn.LayerNorm(node_dim)
        self.linear_net1_e = nn.Sequential(
            nn.Linear(node_dim + edge_dim, edge_hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Linear(edge_hidden_dim_1, edge_dim),
        )
        self.linear_net2_e = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim_2), nn.ReLU(inplace=True), nn.Linear(edge_hidden_dim_2, edge_dim)
        )
        self.norm1_e = nn.LayerNorm(edge_dim)
        self.norm2_e = nn.LayerNorm(edge_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor, group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_features = self.norm1_n(node_features + self.dropout(self.node_attention_layer(node_features, edge_features, group)))
        node_features = self.norm2_n(node_features + self.dropout(self.linear_net_n(node_features)))

        div = torch.sum(group, dim=-1)[:, :, None].clamp_min(1.0)
        concatenated_inputs = torch.cat((edge_features, torch.matmul(group, node_features) / div), dim=-1)
        edge_features = self.norm1_e(edge_features + self.dropout(self.linear_net1_e(concatenated_inputs)))
        edge_features = self.norm2_e(edge_features + self.dropout(self.linear_net2_e(edge_features)))
        return node_features, edge_features


class HRT(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        num_heads: int = 4,
        node_dim: int = 64,
        node_hidden_dim: int = 128,
        edge_dim: int = 64,
        edge_hidden_dim_1: int = 128,
        edge_hidden_dim_2: int = 128,
        dropout: float = 0.0,
        scale: int = 2,
        function_type: int = 1,
    ) -> None:
        super().__init__()
        self.scale = scale
        layer = HRTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.node2edge_mlp = MLP(input_dim=node_dim, output_dim=node_dim, hidden_size=(node_hidden_dim,))
        if self.scale > 1:
            self.group_gen = AdaptiveGroupEstimator(function_type=function_type)

    def init_pair_adj(self, num_agents: int, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        receivers = torch.eye(num_agents, device=device, dtype=dtype).repeat_interleave(num_agents, dim=0)
        senders = torch.eye(num_agents, device=device, dtype=dtype).repeat(num_agents, 1)
        return (receivers + senders).unsqueeze(0).repeat(batch_size, 1, 1)

    def init_edge(self, x: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        div = torch.sum(group, dim=-1)[:, :, None].clamp_min(1.0)
        return self.node2edge_mlp(torch.matmul(group, x) / div)

    def forward(
        self,
        node_features: torch.Tensor,
        _edge_features,
        group: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del _edge_features, group
        if self.scale == 1:
            group = self.init_pair_adj(node_features.shape[1], node_features.shape[0], node_features.device, node_features.dtype)
        else:
            group = self.group_gen(node_features)
        edge_features = self.init_edge(node_features, group)
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, group)
        return node_features, edge_features, group


class HRTNoEdgeInit(nn.Module):
    def __init__(
        self,
        num_layers: int = 3,
        num_heads: int = 4,
        node_dim: int = 64,
        node_hidden_dim: int = 128,
        edge_dim: int = 64,
        edge_hidden_dim_1: int = 128,
        edge_hidden_dim_2: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layer = HRTTransformerLayer(
            num_heads,
            node_dim,
            node_hidden_dim,
            edge_dim,
            edge_hidden_dim_1,
            edge_hidden_dim_2,
            dropout,
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(
        self,
        node_features: torch.Tensor,
        edge_features: torch.Tensor,
        group: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            node_features, edge_features = layer(node_features, edge_features, group)
        return node_features, edge_features, group


class ParallelMARTDecoder(ForecasterHead):
    def __init__(
        self,
        model_dim: int,
        horizon_size: int,
        decoder_hidden_dim: int,
        sample_k: int,
        multiplier: int = 3,
        variant: str = "direct",
    ) -> None:
        super().__init__()
        if variant not in {"direct", "type_conditioned"}:
            raise ValueError("mart_decoder_variant must be one of: direct, type_conditioned.")
        self.horizon_size = horizon_size
        self.sample_k = sample_k
        self.variant = variant
        dims = (model_dim * multiplier, decoder_hidden_dim, decoder_hidden_dim // 2, horizon_size * 2)
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=True):
            weight = nn.Parameter(torch.empty(sample_k, out_dim, in_dim))
            bias = nn.Parameter(torch.empty(sample_k, out_dim))
            for sample_idx in range(sample_k):
                nn.init.kaiming_uniform_(weight[sample_idx], a=math.sqrt(5))
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight[sample_idx])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(bias[sample_idx], -bound, bound)
            self.weights.append(weight)
            self.biases.append(bias)
        self.type_bias = nn.Parameter(torch.zeros(2, sample_k, horizon_size, 2)) if variant == "type_conditioned" else None

    def forward(
        self,
        final_feature: torch.Tensor,
        cur_location: torch.Tensor,
        *,
        static: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = final_feature
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            if x.ndim == 2:
                x = torch.einsum("bd,kod->bko", x, weight) + bias.unsqueeze(0)
            else:
                x = torch.einsum("bkd,kod->bko", x, weight) + bias.unsqueeze(0)
            if idx != len(self.weights) - 1:
                x = F.relu(x)
        outputs = x.view(-1, self.sample_k, self.horizon_size, 2)
        if self.variant == "type_conditioned":
            if static is None or self.type_bias is None:
                raise RuntimeError("static features are required for type_conditioned MART decoder.")
            is_ball = (static[..., 1] == 0).reshape(-1).long()
            outputs = outputs + self.type_bias[is_ball]
        return outputs + cur_location.unsqueeze(1)


class MARTForecastModel(nn.Module):
    """MART-style model adapted from the official `gist-ailab/MART` PyTorch code."""

    def __init__(
        self,
        input_dim: int,
        num_entities: int,
        context_size: int,
        horizon_size: int,
        hidden_dim: int,
        message_dim: int,
        num_layers: int,
        dropout: float,
        transformer_heads: int,
        mart_function_type: int,
        mart_hyper_scale: int,
        mart_decoder_hidden_dim: int,
        mart_sample_k: int = 1,
        point_aggregation: str = "mean",
        decoder_variant: str = "direct",
        mode_distribution: str = "none",
        mixture_min_log_sigma: float = -7.0,
        mixture_max_log_sigma: float = 5.0,
        distribution_detach_features: bool = False,
    ) -> None:
        super().__init__()
        if mart_sample_k < 1:
            raise ValueError(f"mart_sample_k must be >= 1, got {mart_sample_k}.")
        if point_aggregation not in {"mean", "learned_softmax", "argmax", "distribution_softmax", "distribution_argmax"}:
            raise ValueError(
                "mart_point_aggregation must be one of: mean, learned_softmax, argmax, distribution_softmax, distribution_argmax."
            )
        if mode_distribution not in {"none", "gaussian"}:
            raise ValueError("mart_mode_distribution must be one of: none, gaussian.")
        if point_aggregation in {"distribution_softmax", "distribution_argmax"} and mode_distribution != "gaussian":
            raise ValueError("distribution MART point aggregation requires mart_mode_distribution='gaussian'.")
        del num_entities
        module_args: dict[str, Any] = {
            "num_layers": 1,
            "num_heads": transformer_heads,
            "node_dim": hidden_dim,
            "node_hidden_dim": message_dim,
            "edge_dim": hidden_dim,
            "edge_hidden_dim_1": message_dim,
            "edge_hidden_dim_2": message_dim,
            "dropout": dropout,
        }
        hrt_module_args = module_args
        self.horizon_size = horizon_size
        self.sample_k = mart_sample_k
        self.point_aggregation = point_aggregation
        self.mixture_min_log_sigma = mixture_min_log_sigma
        self.mixture_max_log_sigma = mixture_max_log_sigma
        self.distribution_detach_features = distribution_detach_features
        self.input_fc = nn.Linear(input_dim, hidden_dim)
        self.input_fc2 = nn.Linear(hidden_dim * context_size, hidden_dim)
        self.pos_encoder = PositionalAgentEncoding(hidden_dim, dropout, concat=True)
        self.pair_encoders = nn.ModuleList(
            MARTRT(**module_args) if idx == 0 else MARTRTNoEdgeInit(**module_args) for idx in range(num_layers)
        )
        self.hyper_encoders = nn.ModuleList(
            HRT(**hrt_module_args, function_type=mart_function_type, scale=mart_hyper_scale)
            if idx == 0
            else HRTNoEdgeInit(**hrt_module_args)
            for idx in range(num_layers)
        )
        self.heads = ParallelMARTDecoder(
            hidden_dim,
            horizon_size,
            mart_decoder_hidden_dim,
            mart_sample_k,
            multiplier=3,
            variant=decoder_variant,
        )
        self.mode_scorer = nn.Linear(hidden_dim * 3, mart_sample_k) if point_aggregation in {"learned_softmax", "argmax"} else None
        self.distribution_scorer = (
            nn.Linear(hidden_dim * 3, mart_sample_k)
            if mode_distribution == "gaussian" or point_aggregation in {"distribution_softmax", "distribution_argmax"}
            else None
        )
        self.distribution_head = (
            nn.Sequential(
                nn.LayerNorm(hidden_dim * 3),
                nn.Linear(hidden_dim * 3, mart_decoder_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mart_decoder_hidden_dim, horizon_size * mart_sample_k * 3),
            )
            if mode_distribution == "gaussian"
            else None
        )

    def _sample_forward_with_logits(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, MixtureParams | None]:
        del future_xy, teacher_forcing_prob, scheduled_sampling_prob
        batch_size, context_size, num_agents, _ = context.shape
        cur_pos = context[:, -1, :, :2].reshape(batch_size * num_agents, 1, 2).contiguous()
        static = context[:, 0, :, -2:]
        inputs = context.permute(0, 2, 1, 3).reshape(batch_size * num_agents, context_size, -1).contiguous()

        inputs_fc = self.input_fc(inputs).view(batch_size * num_agents, context_size, -1)
        inputs_pos = self.pos_encoder(inputs_fc, num_a=batch_size * num_agents)
        n_initial = self.input_fc2(inputs_pos.contiguous().view(batch_size, num_agents, context_size * inputs_pos.shape[-1]))

        n_pair, e_pair = n_initial, None
        n_group, e_group, group = n_initial, None, None
        for pair_encoder, hyper_encoder in zip(self.pair_encoders, self.hyper_encoders, strict=True):
            n_pair, e_pair = pair_encoder(n_pair, e_pair)
            n_group, e_group, group = hyper_encoder(n_group, e_group, group)

        n_final = torch.cat([n_initial, n_pair, n_group], dim=-1)
        flat_final = n_final.reshape(batch_size * num_agents, -1)
        modes = self.heads(flat_final, cur_pos, static=static).view(batch_size, num_agents, self.sample_k, self.horizon_size, 2)
        modes = modes.permute(0, 3, 1, 2, 4).contiguous()
        logits = self.mode_scorer(flat_final).view(batch_size, num_agents, self.sample_k) if self.mode_scorer is not None else None
        distribution = None
        if self.distribution_head is not None:
            if self.distribution_scorer is None:
                raise RuntimeError("MART Gaussian mode distribution requires a distribution scorer.")
            distribution_features = flat_final.detach() if self.distribution_detach_features else flat_final
            raw_distribution = self.distribution_head(distribution_features).view(
                batch_size, num_agents, self.horizon_size, self.sample_k, 3
            )
            raw_distribution = raw_distribution.permute(0, 2, 1, 3, 4).contiguous()
            log_sigma = raw_distribution[..., 0:2].clamp(self.mixture_min_log_sigma, self.mixture_max_log_sigma)
            rho = torch.tanh(raw_distribution[..., 2]) * 0.999
            distribution_logits = self.distribution_scorer(distribution_features).view(batch_size, num_agents, self.sample_k)
            log_pi = torch.log_softmax(distribution_logits, dim=-1)
            distribution = MixtureParams(means=modes, log_sigma=log_sigma, rho=rho, log_pi=log_pi)
        return modes, logits, distribution

    def sample_forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> torch.Tensor:
        modes, _, _ = self._sample_forward_with_logits(
            context,
            future_xy=future_xy,
            teacher_forcing_prob=teacher_forcing_prob,
            scheduled_sampling_prob=scheduled_sampling_prob,
        )
        return modes

    def _aggregate_modes(
        self,
        modes: torch.Tensor,
        mode_logits: torch.Tensor | None,
        distribution: MixtureParams | None,
    ) -> torch.Tensor:
        if self.point_aggregation == "mean":
            return modes.mean(dim=3)
        if self.point_aggregation in {"distribution_softmax", "distribution_argmax"}:
            if distribution is None:
                raise RuntimeError("distribution params are required for distribution MART point aggregation.")
            if self.point_aggregation == "distribution_softmax":
                weights = distribution.log_pi.exp().view(
                    distribution.log_pi.shape[0], 1, distribution.log_pi.shape[1], distribution.log_pi.shape[2], 1
                )
                return (modes * weights).sum(dim=3)
            best = distribution.log_pi.argmax(dim=-1)
            index = best.view(best.shape[0], 1, best.shape[1], 1, 1).expand(-1, modes.shape[1], -1, -1, modes.shape[-1])
            return torch.gather(modes, 3, index).squeeze(3)
        if mode_logits is None:
            raise RuntimeError("mode logits are required for learned MART point aggregation.")
        if self.point_aggregation == "learned_softmax":
            weights = torch.softmax(mode_logits, dim=-1).view(mode_logits.shape[0], 1, mode_logits.shape[1], mode_logits.shape[2], 1)
            return (modes * weights).sum(dim=3)
        best = mode_logits.argmax(dim=-1)
        index = best.view(best.shape[0], 1, best.shape[1], 1, 1).expand(-1, modes.shape[1], -1, -1, modes.shape[-1])
        return torch.gather(modes, 3, index).squeeze(3)

    def forward(
        self,
        context: torch.Tensor,
        future_xy: torch.Tensor | None = None,
        teacher_forcing_prob: float = 0.0,
        scheduled_sampling_prob: float = 0.0,
    ) -> ForecastOutput:
        modes, mode_logits, distribution = self._sample_forward_with_logits(
            context,
            future_xy=future_xy,
            teacher_forcing_prob=teacher_forcing_prob,
            scheduled_sampling_prob=scheduled_sampling_prob,
        )
        return ForecastOutput(
            point=self._aggregate_modes(modes, mode_logits, distribution), distribution=distribution, modes=modes, mode_logits=mode_logits
        )


def build_mart_hypergraph(config: RunConfig) -> MARTForecastModel:
    if not isinstance(config.model, MartModelConfig):
        raise TypeError("mart_hypergraph requires MartModelConfig.")
    experiment = config.experiment
    common = config.model.common
    transformer = config.model.graph_transformer
    mart = config.model.mart
    return MARTForecastModel(
        input_dim=config.input_dim,
        num_entities=experiment.num_entities,
        context_size=experiment.context_size,
        horizon_size=experiment.horizon_size,
        hidden_dim=common.hidden_dim,
        message_dim=common.message_dim,
        num_layers=common.num_layers,
        dropout=common.dropout,
        transformer_heads=transformer.transformer_heads,
        mart_function_type=mart.mart_function_type,
        mart_hyper_scale=mart.mart_hyper_scale,
        mart_decoder_hidden_dim=mart.mart_decoder_hidden_dim,
        mart_sample_k=mart.mart_sample_k,
        point_aggregation=mart.mart_point_aggregation,
        decoder_variant=mart.mart_decoder_variant,
        mode_distribution=mart.mart_mode_distribution,
        mixture_min_log_sigma=mart.mart_mixture_min_log_sigma,
        mixture_max_log_sigma=mart.mart_mixture_max_log_sigma,
        distribution_detach_features=mart.mart_distribution_detach_features,
    )
