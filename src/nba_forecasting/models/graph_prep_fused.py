from __future__ import annotations

import torch

from .graph_layers import _edge_features, _full_graph_mask, _knn_graph_mask, _radius_graph_mask, _semantic_graph_mask

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only when Triton is unavailable
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _graph_prep_kernel(
        positions,
        static,
        edge_out,
        mask_out,
        num_entities: tl.constexpr,
        position_dim: tl.constexpr,
        static_dim: tl.constexpr,
        edge_dim: tl.constexpr,
        k_neighbors: tl.constexpr,
        use_flags: tl.constexpr,
        dynamic_edges: tl.constexpr,
        has_radius: tl.constexpr,
        edge_radius2: tl.constexpr,
        ablation_mode: tl.constexpr,
        block_n: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        offs = tl.arange(0, block_n)
        valid = offs < num_entities

        x = tl.load(positions + (batch_idx * num_entities + offs) * position_dim + 0, mask=valid, other=0.0).to(tl.float32)
        y = tl.load(positions + (batch_idx * num_entities + offs) * position_dim + 1, mask=valid, other=0.0).to(tl.float32)
        dx = x[None, :] - x[:, None]
        dy = y[None, :] - y[:, None]
        dist2 = dx * dx + dy * dy
        pair_valid = valid[:, None] & valid[None, :]
        eye = offs[:, None] == offs[None, :]

        if ablation_mode == 1:
            graph_mask = pair_valid & ~eye
        else:
            if dynamic_edges:
                dist_for_rank = tl.where(pair_valid & ~eye, dist2, float("inf"))
                rank = tl.zeros((block_n, block_n), dtype=tl.int32)
                for other_idx in tl.static_range(0, block_n):
                    other_dist = tl.sum(tl.where(offs == other_idx, dist_for_rank, 0.0), axis=1)
                    rank += tl.where(other_dist[:, None] < dist_for_rank, 1, 0)
                knn_directed = rank < k_neighbors
                graph_mask = (knn_directed | tl.trans(knn_directed)) & pair_valid & ~eye
            else:
                graph_mask = pair_valid & ~eye
                if use_flags:
                    graph_mask = graph_mask & False

            if use_flags:
                team = tl.load(static + (batch_idx * num_entities + offs) * static_dim + 1, mask=valid, other=0)
                same_team = (team[:, None] == team[None, :]) & (team[:, None] != 0) & (team[None, :] != 0)
                involves_ball = (team[:, None] == 0) | (team[None, :] == 0)
                semantic = (same_team | involves_ball) & pair_valid & ~eye
                graph_mask = graph_mask | semantic

            if has_radius:
                graph_mask = graph_mask & (dist2 <= edge_radius2)

            if ablation_mode == 2:
                shifted_j = (offs[None, :] + 1) % num_entities
                shuffled = (
                    tl.load(
                        mask_out + (batch_idx * num_entities + offs[:, None]) * num_entities + shifted_j,
                        mask=pair_valid,
                        other=0,
                    )
                    != 0
                )
                graph_mask = shuffled & pair_valid & ~eye

        edge_base = ((batch_idx * num_entities + offs[:, None]) * num_entities + offs[None, :]) * edge_dim
        if ablation_mode == 1:
            dx = tl.zeros((block_n, block_n), dtype=tl.float32)
            dy = tl.zeros((block_n, block_n), dtype=tl.float32)
            dist = tl.zeros((block_n, block_n), dtype=tl.float32)
        else:
            dist = tl.sqrt(dist2)
        tl.store(edge_out + edge_base + 0, dx, mask=pair_valid)
        tl.store(edge_out + edge_base + 1, dy, mask=pair_valid)
        tl.store(edge_out + edge_base + 2, dist, mask=pair_valid)

        if use_flags:
            team = tl.load(static + (batch_idx * num_entities + offs) * static_dim + 1, mask=valid, other=0)
            same_team_f = ((team[:, None] == team[None, :]) & (team[:, None] != 0) & (team[None, :] != 0)).to(tl.float32)
            involves_ball_f = ((team[:, None] == 0) | (team[None, :] == 0)).to(tl.float32)
            if ablation_mode == 1:
                same_team_f = tl.zeros((block_n, block_n), dtype=tl.float32)
                involves_ball_f = tl.zeros((block_n, block_n), dtype=tl.float32)
            tl.store(edge_out + edge_base + 3, same_team_f, mask=pair_valid)
            tl.store(edge_out + edge_base + 4, involves_ball_f, mask=pair_valid)

        tl.store(mask_out + (batch_idx * num_entities + offs[:, None]) * num_entities + offs[None, :], graph_mask, mask=pair_valid)


def _graph_inputs_reference(
    positions: torch.Tensor,
    static: torch.Tensor,
    *,
    k_neighbors: int,
    use_semantic_flags: bool,
    dynamic_edges: bool,
    edge_radius: float | None,
    ablation_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if ablation_mode == "no_graph":
        mask = _full_graph_mask(positions.shape[0], positions.shape[1], positions.device)
    elif not dynamic_edges:
        if use_semantic_flags:
            mask = _semantic_graph_mask(static)
        else:
            mask = _full_graph_mask(positions.shape[0], positions.shape[1], positions.device)
    else:
        mask = _knn_graph_mask(positions, k_neighbors)
        if use_semantic_flags:
            mask = mask | _semantic_graph_mask(static)

    if edge_radius is not None:
        mask = mask & _radius_graph_mask(positions, edge_radius)

    if ablation_mode == "shuffled_graph":
        permutation = torch.arange(positions.shape[1], device=positions.device).roll(1)
        mask = mask.index_select(2, permutation)
        eye = torch.eye(positions.shape[1], device=positions.device, dtype=torch.bool).unsqueeze(0)
        mask = mask & ~eye

    edge_feat = _edge_features(positions, static, use_semantic_flags)
    if ablation_mode == "no_graph":
        edge_feat = torch.zeros_like(edge_feat)
    return edge_feat, mask


def graph_inputs_fused(
    positions: torch.Tensor,
    static: torch.Tensor,
    *,
    k_neighbors: int,
    use_semantic_flags: bool,
    dynamic_edges: bool,
    edge_radius: float | None,
    ablation_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if triton is None or not positions.is_cuda or positions.ndim != 3 or positions.shape[-1] != 2 or positions.shape[1] > 16:
        return _graph_inputs_reference(
            positions,
            static,
            k_neighbors=k_neighbors,
            use_semantic_flags=use_semantic_flags,
            dynamic_edges=dynamic_edges,
            edge_radius=edge_radius,
            ablation_mode=ablation_mode,
        )

    if ablation_mode not in {"normal", "no_graph", "shuffled_graph"}:
        raise ValueError("ablation_mode must be one of: normal, no_graph, shuffled_graph.")
    batch_size, num_entities, position_dim = positions.shape
    edge_dim = position_dim + 1 + (2 if use_semantic_flags else 0)
    positions_c = positions.contiguous()
    static_c = static.contiguous()
    edge_feat = torch.empty((batch_size, num_entities, num_entities, edge_dim), device=positions.device, dtype=positions.dtype)
    mask = torch.empty((batch_size, num_entities, num_entities), device=positions.device, dtype=torch.bool)
    mode_id = {"normal": 0, "no_graph": 1, "shuffled_graph": 2}[ablation_mode]

    if mode_id == 2:
        _, mask = _graph_inputs_reference(
            positions,
            static,
            k_neighbors=k_neighbors,
            use_semantic_flags=use_semantic_flags,
            dynamic_edges=dynamic_edges,
            edge_radius=edge_radius,
            ablation_mode="normal",
        )

    _graph_prep_kernel[(batch_size,)](
        positions_c,
        static_c,
        edge_feat,
        mask,
        num_entities=num_entities,
        position_dim=position_dim,
        static_dim=static.shape[-1],
        edge_dim=edge_dim,
        k_neighbors=max(1, min(k_neighbors, num_entities - 1)),
        use_flags=use_semantic_flags,
        dynamic_edges=dynamic_edges,
        has_radius=edge_radius is not None,
        edge_radius2=0.0 if edge_radius is None else float(edge_radius * edge_radius),
        ablation_mode=mode_id,
        block_n=16,
        num_warps=4,
        num_stages=3,
    )
    return edge_feat, mask
