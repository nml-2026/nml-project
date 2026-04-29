from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json
import random

import torch
from torch.utils.data import Dataset, Sampler


@dataclass(slots=True)
class Normalizer:
    mu: torch.Tensor
    sigma: torch.Tensor

    def normalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        return (xy - self.mu) / self.sigma

    def denormalize_xy(self, xy: torch.Tensor) -> torch.Tensor:
        return xy * self.sigma + self.mu

    def state_dict(self) -> dict[str, list[float]]:
        return {
            "mu": self.mu.tolist(),
            "sigma": self.sigma.tolist(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> "Normalizer":
        return cls(
            mu=torch.tensor(state["mu"], dtype=torch.float32),
            sigma=torch.tensor(state["sigma"], dtype=torch.float32),
        )


def collect_sequence_paths(split_dir: str | Path) -> dict[int, Path]:
    root = Path(split_dir)
    paths: dict[int, Path] = {}
    for path in sorted(root.glob("*.pt"), key=lambda value: int(value.stem)):
        paths[int(path.stem)] = path.resolve()
    if not paths:
        raise FileNotFoundError(f"No .pt files found under {root}.")
    return paths


def _validate_split(split: dict[str, list[int]], available_ids: set[int]) -> None:
    seen: set[int] = set()
    for split_name in ("train", "val", "holdout"):
        ids = split.get(split_name, [])
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"Split {split_name} overlaps with previous splits: {sorted(overlap)[:5]}")
        unknown = set(ids) - available_ids
        if unknown:
            raise ValueError(f"Split {split_name} contains unknown ids: {sorted(unknown)[:5]}")
        seen.update(ids)
    if seen != available_ids:
        missing = available_ids - seen
        raise ValueError(f"Split is missing ids: {sorted(missing)[:5]}")


def freeze_sequence_split(train_dir: str | Path, split_path: str | Path, seed: int = 0) -> dict[str, list[int]]:
    split_file = Path(split_path)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    sequence_paths = collect_sequence_paths(train_dir)
    available_ids = set(sequence_paths)
    if split_file.exists():
        split = json.loads(split_file.read_text())
        _validate_split(split, available_ids)
        return split

    ids = sorted(available_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_total = len(ids)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    split = {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "holdout": sorted(ids[n_train + n_val:]),
    }
    _validate_split(split, available_ids)
    split_file.write_text(json.dumps(split, indent=2, sort_keys=True))
    return split


def freeze_kfold_sequence_splits(
    train_dir: str | Path,
    output_dir: str | Path,
    num_folds: int,
    seed: int = 0,
    base_split_path: str | Path | None = None,
    include_holdout: bool = False,
) -> list[Path]:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2 for cross-validation.")

    train_root = Path(train_dir)
    split_root = Path(output_dir)
    split_root.mkdir(parents=True, exist_ok=True)
    sequence_paths = collect_sequence_paths(train_root)
    available_ids = set(sequence_paths)

    if base_split_path is not None:
        base_split = freeze_sequence_split(train_root, base_split_path, seed=seed)
        pool_ids = sorted(base_split["train"] + base_split["val"])
        holdout_ids = [] if include_holdout else sorted(base_split["holdout"])
        if include_holdout:
            pool_ids.extend(base_split["holdout"])
            pool_ids = sorted(pool_ids)
    else:
        base_split = None
        pool_ids = sorted(available_ids)
        holdout_ids = []

    if num_folds > len(pool_ids):
        raise ValueError(f"num_folds={num_folds} exceeds the CV pool size of {len(pool_ids)} sequences.")

    rng = random.Random(seed)
    shuffled_ids = list(pool_ids)
    rng.shuffle(shuffled_ids)
    fold_sizes = [len(shuffled_ids) // num_folds] * num_folds
    for idx in range(len(shuffled_ids) % num_folds):
        fold_sizes[idx] += 1

    fold_paths: list[Path] = []
    cursor = 0
    for fold_idx, fold_size in enumerate(fold_sizes):
        val_ids = sorted(shuffled_ids[cursor:cursor + fold_size])
        cursor += fold_size
        train_ids = sorted(seq_id for seq_id in pool_ids if seq_id not in set(val_ids))
        split = {
            "train": train_ids,
            "val": val_ids,
            "holdout": holdout_ids,
        }
        _validate_split(split, available_ids)
        fold_path = split_root / f"fold_{fold_idx}.json"
        fold_path.write_text(json.dumps(split, indent=2, sort_keys=True))
        fold_paths.append(fold_path.resolve())

    summary = {
        "num_folds": num_folds,
        "seed": seed,
        "base_split_path": str(Path(base_split_path).resolve()) if base_split_path is not None else None,
        "include_holdout": include_holdout,
        "pool_size": len(pool_ids),
        "holdout_size": len(holdout_ids),
        "fold_paths": [str(path) for path in fold_paths],
    }
    if base_split is not None:
        summary["base_holdout"] = holdout_ids
    (split_root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    return fold_paths


def compute_normalizer(sequence_paths: Iterable[Path]) -> Normalizer:
    total_count = 0
    total_sum = torch.zeros(2, dtype=torch.float64)
    total_squared_sum = torch.zeros(2, dtype=torch.float64)
    for path in sequence_paths:
        seq = torch.load(path, map_location="cpu", weights_only=False).float()
        xy = seq[:, :, :2]
        total_count += xy.shape[0] * xy.shape[1]
        total_sum += xy.sum(dim=(0, 1), dtype=torch.float64)
        total_squared_sum += (xy ** 2).sum(dim=(0, 1), dtype=torch.float64)

    mu = total_sum / total_count
    var = total_squared_sum / total_count - (mu ** 2)
    sigma = torch.sqrt(var.clamp_min(1e-6))
    return Normalizer(mu=mu.float(), sigma=sigma.float())


def _build_feature_tensor(seq: torch.Tensor, normalizer: Normalizer, use_velocity: bool) -> tuple[torch.Tensor, torch.Tensor]:
    raw_xy = seq[:, :, :2].float()
    static = seq[:, :, 2:].float()
    normalized_xy = normalizer.normalize_xy(raw_xy)
    features = [normalized_xy]
    if use_velocity:
        velocity = torch.zeros_like(normalized_xy)
        velocity[1:] = normalized_xy[1:] - normalized_xy[:-1]
        features.append(velocity)
    features.append(static)
    return torch.cat(features, dim=-1), raw_xy


class SequenceWindowDataset(Dataset):
    def __init__(
        self,
        sequence_paths: Iterable[Path],
        normalizer: Normalizer,
        context_size: int,
        horizon_size: int,
        use_velocity: bool,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.use_velocity = use_velocity
        self.sequence_paths = list(sequence_paths)
        self.samples: list[dict[str, torch.Tensor | int]] = []
        self.window_index: list[tuple[int, int]] = []
        self._load_sequences(normalizer)

    def _load_sequences(self, normalizer: Normalizer) -> None:
        for path in self.sequence_paths:
            seq_id = int(path.stem)
            seq = torch.load(path, map_location="cpu", weights_only=False).float()
            if seq.shape[0] < self.window_size:
                continue
            features, raw_xy = _build_feature_tensor(seq, normalizer, self.use_velocity)
            static = seq[0, :, 2:].float()
            sample = {
                "sequence_id": seq_id,
                "features": features,
                "raw_xy": raw_xy,
                "static": static,
            }
            sample_idx = len(self.samples)
            self.samples.append(sample)
            max_start = seq.shape[0] - self.window_size
            self.window_index.extend((sample_idx, start) for start in range(max_start + 1))

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        sample_idx, start = self.window_index[index]
        sample = self.samples[sample_idx]
        features = sample["features"]
        raw_xy = sample["raw_xy"]
        context_end = start + self.context_size
        future_end = context_end + self.horizon_size
        return {
            "sequence_id": int(sample["sequence_id"]),
            "start_idx": start,
            "context": features[start:context_end],
            "future_xy": features[context_end:future_end, :, :2],
            "future_xy_raw": raw_xy[context_end:future_end],
            "context_raw_xy": raw_xy[start:context_end],
            "static": sample["static"],
        }


class NotebookSequenceDataset(Dataset):
    def __init__(
        self,
        sequence_paths: Iterable[Path],
        normalizer: Normalizer,
        context_size: int,
        horizon_size: int,
        use_velocity: bool,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.use_velocity = use_velocity
        self.sequence_paths = list(sequence_paths)
        self.samples: list[dict[str, torch.Tensor | int]] = []
        self.max_start: list[int] = []
        self._load_sequences(normalizer)

    def _load_sequences(self, normalizer: Normalizer) -> None:
        for path in self.sequence_paths:
            seq_id = int(path.stem)
            seq = torch.load(path, map_location="cpu", weights_only=False).float()
            if seq.shape[0] < self.window_size:
                continue
            features, raw_xy = _build_feature_tensor(seq, normalizer, self.use_velocity)
            self.samples.append(
                {
                    "sequence_id": seq_id,
                    "features": features,
                    "raw_xy": raw_xy,
                    "static": seq[0, :, 2:].float(),
                }
            )
            self.max_start.append(max(0, seq.shape[0] - self.window_size))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: tuple[int, int] | int) -> dict[str, torch.Tensor | int]:
        if isinstance(index, tuple):
            sample_idx, start = index
        else:
            sample_idx, start = index, 0
        sample = self.samples[sample_idx]
        features = sample["features"]
        raw_xy = sample["raw_xy"]
        context_end = start + self.context_size
        future_end = context_end + self.horizon_size
        return {
            "sequence_id": int(sample["sequence_id"]),
            "start_idx": start,
            "context": features[start:context_end],
            "future_xy": features[context_end:future_end, :, :2],
            "future_xy_raw": raw_xy[context_end:future_end],
            "context_raw_xy": raw_xy[start:context_end],
            "static": sample["static"],
        }


class NotebookRandomWindowSampler(Sampler[tuple[int, int]]):
    def __init__(self, max_start: list[int], seed: int = 0, shuffle: bool = True) -> None:
        self.max_start = max_start
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.generator = torch.Generator().manual_seed(seed)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.generator.manual_seed(self.seed + epoch)

    def __iter__(self):
        num_sequences = len(self.max_start)
        if self.shuffle:
            order = torch.randperm(num_sequences, generator=self.generator).tolist()
        else:
            order = list(range(num_sequences))
        for sample_idx in order:
            max_start = self.max_start[sample_idx]
            start = int(torch.randint(0, max_start + 1, size=(), generator=self.generator).item())
            yield sample_idx, start

    def __len__(self) -> int:
        return len(self.max_start)


class TestSequenceDataset(Dataset):
    def __init__(self, test_dir: str | Path, normalizer: Normalizer, use_velocity: bool) -> None:
        super().__init__()
        self.sequence_paths = list(collect_sequence_paths(test_dir).values())
        self.samples: list[dict[str, torch.Tensor | int]] = []
        self.use_velocity = use_velocity
        self._load_sequences(normalizer)

    def _load_sequences(self, normalizer: Normalizer) -> None:
        for path in self.sequence_paths:
            seq_id = int(path.stem)
            seq = torch.load(path, map_location="cpu", weights_only=False).float()
            features, raw_xy = _build_feature_tensor(seq, normalizer, self.use_velocity)
            self.samples.append(
                {
                    "sequence_id": seq_id,
                    "context": features,
                    "context_raw_xy": raw_xy,
                    "static": seq[0, :, 2:].float(),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | int]:
        return self.samples[index]
