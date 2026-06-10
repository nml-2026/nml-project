from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nba_forecasting.config import RunConfig
from tests.helpers import IS_PLAYER, TEAM_VALUES, write_synthetic_dataset


@pytest.fixture
def synthetic_dataset_path(tmp_path: Path) -> Path:
    """Generates a standard synthetic dataset in a temporary path."""
    return write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24, 21, 23, 25, 26, 27, 28, 29], test_count=2)


def make_config(
    model_name: str,
    data_dir: str = "data",
    use_velocity: bool = True,
    dynamic_edges: bool = True,
    use_semantic_flags: bool = True,
    decoder_type: str = "autoregressive",
    use_relation_types: bool = False,
    use_entity_embeddings: bool = False,
) -> RunConfig:
    """Helper to generate a base RunConfig for testing."""
    return RunConfig.from_dict(
        {
            "experiment": {
                "name": model_name,
                "data_dir": data_dir,
                "output_root": "runs",
                "split_path": "artifacts/splits/default_split.json",
                "device": "cpu",
                "context_size": 8,
                "horizon_size": 12,
                "num_entities": 11,
                "checkpoint_metric": "ade",
                "seed": 0,
            },
            "features": {"use_velocity": use_velocity},
            "model": {
                "family": model_name,
                "common": {"hidden_dim": 64, "message_dim": 64, "num_layers": 2, "dropout": 0.0},
                "graph": {
                    "k_neighbors": 3,
                    "use_semantic_flags": use_semantic_flags,
                    "dynamic_edges": dynamic_edges,
                    "use_relation_types": use_relation_types,
                    "use_entity_embeddings": use_entity_embeddings,
                    "entity_embedding_dim": 8,
                    "decoder_type": decoder_type,
                    "direct_decoder_heads": 2,
                    "separate_ball_head": False,
                },
            },
            "training": {
                "epochs": 2,
                "batch_size": 4,
                "lr": 0.01,
                "weight_decay": 0.0,
                "grad_clip": 1.0,
                "num_workers": 0,
                "patience": 2,
                "log_every": 10,
                "teacher_forcing_prob": 0.0,
                "scheduled_sampling_final_prob": 0.0,
                "ema_decay": 0.0,
                "swa_start_pct": 0.0,
                "train_window_sampling": "all_windows",
                "val_window_sampling": "all_windows",
            },
            "evaluation": {"num_examples": 2},
        }
    )


def make_context(batch_size: int, use_velocity: bool) -> torch.Tensor:
    """Helper to generate a batch context tensor."""
    time = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1)
    entity = torch.arange(11, dtype=torch.float32).view(1, 1, 11, 1)
    x = 0.2 * time + entity
    y = -0.1 * time + 0.05 * entity
    positions = torch.cat([x, y], dim=-1).expand(batch_size, -1, -1, -1)
    static = torch.stack([IS_PLAYER, TEAM_VALUES], dim=-1).view(1, 1, 11, 2).expand(batch_size, 8, -1, -1)
    if not use_velocity:
        return torch.cat([positions, static], dim=-1)
    velocity = torch.zeros_like(positions)
    velocity[:, 1:] = positions[:, 1:] - positions[:, :-1]
    return torch.cat([positions, velocity, static], dim=-1)
