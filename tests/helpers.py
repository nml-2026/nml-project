from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import pandas as pd
import torch

TEAM_VALUES = torch.tensor([0, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1], dtype=torch.float32)
IS_PLAYER = (TEAM_VALUES != 0).float()

PUBLICATION_SCRIPTS = frozenset(
    {
        "run_tune.sh",
        "collect_results.py",
        "demo.sh",
    }
)


@runtime_checkable
class HasGraphInputs(Protocol):
    def graph_inputs(self, positions: torch.Tensor, static: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


def make_synthetic_sequence(length: int, offset: float = 0.0) -> torch.Tensor:
    time = torch.arange(length, dtype=torch.float32).view(length, 1, 1)
    entity = torch.arange(11, dtype=torch.float32).view(1, 11, 1)
    x = offset + entity + 0.25 * time
    y = -offset + 0.1 * entity - 0.15 * time
    is_player = IS_PLAYER.view(1, 11, 1).expand(length, -1, -1)
    team = TEAM_VALUES.view(1, 11, 1).expand(length, -1, -1)
    return torch.cat([x, y, is_player, team], dim=-1)


def write_synthetic_dataset(root: Path, train_lengths: list[int], test_count: int = 2) -> Path:
    data_dir = root / "data"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for seq_id, length in enumerate(train_lengths):
        torch.save(make_synthetic_sequence(length, offset=float(seq_id)), train_dir / f"{seq_id}.pt")
    for seq_id in range(test_count):
        torch.save(make_synthetic_sequence(8, offset=float(seq_id + 100))[:8], test_dir / f"{seq_id}.pt")

    pd.DataFrame({"id": list(range(len(train_lengths)))}).to_csv(data_dir / "train.csv", index=False)
    pd.DataFrame({"id": list(range(test_count))}).to_csv(data_dir / "test.csv", index=False)

    submission_columns = ["id"] + [
        f"entity_{entity_idx}_time_{time_idx}_{axis}" for time_idx in range(12) for entity_idx in range(11) for axis in ("x", "y")
    ]
    pd.DataFrame([[seq_id] + [0.0] * (len(submission_columns) - 1) for seq_id in range(test_count)], columns=submission_columns).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
    )
    return data_dir


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data: list) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return self.data[index]
