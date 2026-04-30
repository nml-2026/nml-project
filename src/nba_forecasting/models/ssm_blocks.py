from __future__ import annotations

import torch
from torch import nn


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
