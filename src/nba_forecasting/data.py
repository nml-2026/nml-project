from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypedDict

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import RunConfig


class WindowFields(TypedDict):
    context: torch.Tensor
    future_xy: torch.Tensor
    future_xy_raw: torch.Tensor
    context_raw_xy: torch.Tensor


class SupportsNormalizationStats(Protocol):
    mu: torch.Tensor
    sigma: torch.Tensor


class WindowTransform(Protocol):
    def apply_window(self, window: WindowFields, generator: torch.Generator) -> WindowFields: ...


@dataclass(slots=True)
class TransformPipeline:
    transforms: tuple[WindowTransform, ...] = ()

    def apply_window(self, window: WindowFields, generator: torch.Generator) -> WindowFields:
        for transform in self.transforms:
            window = transform.apply_window(window, generator)
        return window


@dataclass(slots=True)
class CourtReflectionTransform:
    use_velocity: bool
    shift_x: float
    shift_y: float
    reflect_prob: float = 0.0
    reflect_x_prob: float = 0.0

    @classmethod
    def from_normalization(
        cls,
        stats: SupportsNormalizationStats,
        *,
        use_velocity: bool,
        reflect_prob: float = 0.0,
        reflect_x_prob: float = 0.0,
    ) -> CourtReflectionTransform:
        shift_x, shift_y = reflect_shifts(stats)
        return cls(
            use_velocity=use_velocity,
            shift_x=shift_x,
            shift_y=shift_y,
            reflect_prob=reflect_prob,
            reflect_x_prob=reflect_x_prob,
        )

    def apply_window(self, window: WindowFields, generator: torch.Generator) -> WindowFields:
        flip_x = self.reflect_x_prob > 0.0 and torch.rand((), generator=generator).item() < self.reflect_x_prob
        flip_y = self.reflect_prob > 0.0 and torch.rand((), generator=generator).item() < self.reflect_prob
        if not flip_x and not flip_y:
            return window
        return reflect_window(
            window,
            use_velocity=self.use_velocity,
            shift_x=self.shift_x,
            shift_y=self.shift_y,
            flip_x=flip_x,
            flip_y=flip_y,
        )

    def reflect_context(self, context: torch.Tensor, *, flip_x: bool, flip_y: bool) -> torch.Tensor:
        return reflect_context_tensor(
            context,
            use_velocity=self.use_velocity,
            shift_x=self.shift_x,
            shift_y=self.shift_y,
            flip_x=flip_x,
            flip_y=flip_y,
        )

    def inverse_prediction(self, prediction: torch.Tensor, *, flip_x: bool, flip_y: bool) -> torch.Tensor:
        return reflect_prediction_xy(
            prediction,
            shift_x=self.shift_x,
            shift_y=self.shift_y,
            flip_x=flip_x,
            flip_y=flip_y,
        )


@dataclass(slots=True)
class ContextNoiseTransform:
    use_velocity: bool
    sigma: torch.Tensor
    std_ft: float = 0.0

    @classmethod
    def from_normalization(
        cls,
        stats: SupportsNormalizationStats,
        *,
        use_velocity: bool,
        std_ft: float = 0.0,
    ) -> ContextNoiseTransform:
        return cls(use_velocity=use_velocity, sigma=stats.sigma.float(), std_ft=std_ft)

    def apply_window(self, window: WindowFields, generator: torch.Generator) -> WindowFields:
        if self.std_ft <= 0.0:
            return window
        context = window["context"].clone()
        context_raw_xy = window["context_raw_xy"].clone()
        noise_raw = (
            torch.randn(context_raw_xy.shape, generator=generator, device=context_raw_xy.device, dtype=context_raw_xy.dtype) * self.std_ft
        )
        context_raw_xy = context_raw_xy + noise_raw
        context[..., :2] = context[..., :2] + noise_raw / self.sigma.to(context.device).view(1, 1, 2)
        if self.use_velocity:
            _recompute_context_velocity_(context)
        return {
            "context": context,
            "future_xy": window["future_xy"],
            "future_xy_raw": window["future_xy_raw"],
            "context_raw_xy": context_raw_xy,
        }


@dataclass(slots=True)
class ContextFrameDropoutTransform:
    use_velocity: bool
    prob: float = 0.0
    mode: str = "previous"

    def apply_window(self, window: WindowFields, generator: torch.Generator) -> WindowFields:
        if self.prob <= 0.0:
            return window
        if self.mode not in {"previous", "zero_velocity"}:
            raise ValueError("context_frame_dropout_mode must be one of: previous, zero_velocity.")
        context = window["context"].clone()
        context_raw_xy = window["context_raw_xy"].clone()
        if context.shape[0] <= 1:
            return window
        mask = torch.rand((context.shape[0] - 1,), generator=generator, device=context.device) < self.prob
        if self.mode == "previous":
            for step in range(1, context.shape[0] - 1):
                if bool(mask[step]):
                    context[step, :, :2] = context[step - 1, :, :2]
                    context_raw_xy[step] = context_raw_xy[step - 1]
            if bool(mask[0]):
                context[0, :, :2] = context[1, :, :2]
                context_raw_xy[0] = context_raw_xy[1]
            if self.use_velocity:
                _recompute_context_velocity_(context)
        elif self.use_velocity:
            context[:-1, :, 2:4] = torch.where(mask.view(-1, 1, 1), torch.zeros_like(context[:-1, :, 2:4]), context[:-1, :, 2:4])
        return {
            "context": context,
            "future_xy": window["future_xy"],
            "future_xy_raw": window["future_xy_raw"],
            "context_raw_xy": context_raw_xy,
        }


def _recompute_context_velocity_(context: torch.Tensor) -> None:
    if context.shape[-1] < 4:
        return
    velocity = context[:, :, 2:4]
    velocity.zero_()
    velocity[1:] = context[1:, :, :2] - context[:-1, :, :2]


def reflect_shifts(stats: SupportsNormalizationStats) -> tuple[float, float]:
    shift_x = float(2.0 * stats.mu[0].item() / stats.sigma[0].item())
    shift_y = float(2.0 * stats.mu[1].item() / stats.sigma[1].item())
    return shift_x, shift_y


def reflect_window(
    window: WindowFields,
    *,
    use_velocity: bool,
    shift_x: float,
    shift_y: float,
    flip_x: bool,
    flip_y: bool,
) -> WindowFields:
    context = window["context"].clone()
    future_xy = window["future_xy"].clone()
    future_xy_raw = window["future_xy_raw"].clone()
    context_raw_xy = window["context_raw_xy"].clone()

    axes: list[tuple[int, int, float]] = []
    if flip_x:
        axes.append((0, 2, shift_x))
    if flip_y:
        axes.append((1, 3, shift_y))
    for pos_c, vel_c, shift in axes:
        context[..., pos_c] = -context[..., pos_c] - shift
        if use_velocity:
            context[..., vel_c] = -context[..., vel_c]
        future_xy[..., pos_c] = -future_xy[..., pos_c] - shift
        future_xy_raw[..., pos_c] = -future_xy_raw[..., pos_c]
        context_raw_xy[..., pos_c] = -context_raw_xy[..., pos_c]
    return {
        "context": context,
        "future_xy": future_xy,
        "future_xy_raw": future_xy_raw,
        "context_raw_xy": context_raw_xy,
    }


def reflect_context_tensor(
    context: torch.Tensor,
    *,
    use_velocity: bool,
    shift_x: float,
    shift_y: float,
    flip_x: bool,
    flip_y: bool,
) -> torch.Tensor:
    context = context.clone()
    if flip_x:
        context[..., 0] = -context[..., 0] - shift_x
        if use_velocity:
            context[..., 2] = -context[..., 2]
    if flip_y:
        context[..., 1] = -context[..., 1] - shift_y
        if use_velocity:
            context[..., 3] = -context[..., 3]
    return context


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
    def from_state_dict(cls, state: dict[str, list[float]]) -> Normalizer:
        return cls(
            mu=torch.tensor(state["mu"], dtype=torch.float32),
            sigma=torch.tensor(state["sigma"], dtype=torch.float32),
        )


class SequenceSample(TypedDict):
    sequence_id: int
    features: torch.Tensor
    raw_xy: torch.Tensor
    static: torch.Tensor


class WindowSample(TypedDict):
    sequence_id: int
    start_idx: int
    context: torch.Tensor
    future_xy: torch.Tensor
    future_xy_raw: torch.Tensor
    context_raw_xy: torch.Tensor
    static: torch.Tensor


def collect_sequence_paths(split_dir: str | Path) -> dict[int, Path]:
    root = Path(split_dir)
    paths: dict[int, Path] = {}
    for path in sorted(root.glob("*.pt"), key=lambda value: int(value.stem)):
        paths[int(path.stem)] = path.resolve()
    if not paths:
        raise FileNotFoundError(f"No .pt files found under {root}.")
    return paths


def consolidated_sequence_path(split_dir: str | Path) -> Path:
    root = Path(split_dir)
    return root.parent / f"{root.name}_consolidated.pt"


def _load_consolidated_sequences(path: str | Path) -> dict[int, torch.Tensor]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError(f"Consolidated dataset must be a dict, got {type(data).__name__}: {path}")
    return {int(seq_id): seq.float() for seq_id, seq in data.items()}


def collect_sequence_ids(split_dir: str | Path) -> set[int]:
    root = Path(split_dir)
    consolidated_path = consolidated_sequence_path(root)
    if consolidated_path.exists():
        return set(_load_consolidated_sequences(consolidated_path))
    if root.exists():
        paths = sorted(root.glob("*.pt"), key=lambda value: int(value.stem))
        if paths:
            return {int(path.stem) for path in paths}
    raise FileNotFoundError(f"No .pt files found under {root} and no consolidated file found at {consolidated_path}.")


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
    available_ids = collect_sequence_ids(train_dir)
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
        "val": sorted(ids[n_train : n_train + n_val]),
        "holdout": sorted(ids[n_train + n_val :]),
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
    available_ids = collect_sequence_ids(train_root)

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
        val_ids = sorted(shuffled_ids[cursor : cursor + fold_size])
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

    summary: dict[str, Any] = {
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


def load_sequences_from_dir_or_consolidated(split_dir: str | Path) -> dict[int, torch.Tensor]:
    """Loads all sequences from a split directory. If a consolidated file exists, loads it.
    Otherwise, loads individual files, consolidates them, saves the consolidated file to parent dir, and returns it.
    """
    import time

    split_dir = Path(split_dir)
    consolidated_path = consolidated_sequence_path(split_dir)

    if consolidated_path.exists():
        try:
            print(f"Loading consolidated dataset from {consolidated_path}...")
            start_time = time.perf_counter()
            data = _load_consolidated_sequences(consolidated_path)
            elapsed = time.perf_counter() - start_time
            print(f"Loaded consolidated dataset in {elapsed:.4f} seconds.")
            return data
        except Exception as e:
            print(f"Warning: Failed to load consolidated file {consolidated_path} due to {e}. Falling back to individual files.")

    print(f"Consolidated file not found or failed to load. Loading individual files from {split_dir}...")
    start_time = time.perf_counter()
    paths = collect_sequence_paths(split_dir)
    loaded_data = {}
    for seq_id, path in paths.items():
        loaded_data[seq_id] = torch.load(path, map_location="cpu", weights_only=False).float()
    elapsed = time.perf_counter() - start_time
    print(f"Loaded {len(loaded_data)} individual files in {elapsed:.4f} seconds.")

    try:
        consolidated_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(loaded_data, consolidated_path)
        print(f"Saved consolidated dataset to {consolidated_path}")
    except Exception as e:
        print(f"Warning: Failed to save consolidated file to {consolidated_path} due to {e}")

    return loaded_data


def consolidate_sequence_dir(split_dir: str | Path, output_path: str | Path | None = None, *, overwrite: bool = False) -> Path:
    source_dir = Path(split_dir)
    output = Path(output_path) if output_path is not None else consolidated_sequence_path(source_dir)
    if output.exists() and not overwrite:
        data = _load_consolidated_sequences(output)
        if not data:
            raise ValueError(f"Existing consolidated dataset is empty: {output}")
        return output.resolve()

    paths = collect_sequence_paths(source_dir)
    data = {seq_id: torch.load(path, map_location="cpu", weights_only=False).float() for seq_id, path in paths.items()}
    if not data:
        raise FileNotFoundError(f"No sequences found under {source_dir}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, output)
    return output.resolve()


def prepare_remote_dataset(
    data_dir: str | Path = "data",
    split_path: str | Path = "artifacts/splits/default_split.json",
    *,
    seed: int = 0,
    overwrite: bool = False,
) -> dict[str, object]:
    root = Path(data_dir)
    train_consolidated = consolidate_sequence_dir(root / "train", overwrite=overwrite)
    test_consolidated = consolidate_sequence_dir(root / "test", overwrite=overwrite)
    split = freeze_sequence_split(root / "train", split_path, seed=seed)
    return {
        "data_dir": str(root.resolve()),
        "train_consolidated": str(train_consolidated),
        "test_consolidated": str(test_consolidated),
        "split_path": str(Path(split_path).resolve()),
        "train_sequences": len(split["train"]),
        "val_sequences": len(split["val"]),
        "holdout_sequences": len(split["holdout"]),
    }


def _iter_sequences(sequences: Iterable[Path] | dict[int, torch.Tensor]) -> Iterable[tuple[int, torch.Tensor]]:
    if isinstance(sequences, dict):
        for seq_id, seq in sequences.items():
            if isinstance(seq, torch.Tensor):
                yield seq_id, seq.float()
    else:
        for path in sequences:
            seq_id = int(path.stem)
            seq = torch.load(path, map_location="cpu", weights_only=False).float()
            yield seq_id, seq


def compute_normalizer(sequences: Iterable[Path | torch.Tensor]) -> Normalizer:
    total_count = 0
    total_sum = torch.zeros(2, dtype=torch.float64)
    total_squared_sum = torch.zeros(2, dtype=torch.float64)
    for item in sequences:
        if isinstance(item, torch.Tensor):
            seq = item.float()
        else:
            seq = torch.load(item, map_location="cpu", weights_only=False).float()
        xy = seq[:, :, :2]
        total_count += xy.shape[0] * xy.shape[1]
        total_sum += xy.sum(dim=(0, 1), dtype=torch.float64)
        total_squared_sum += (xy**2).sum(dim=(0, 1), dtype=torch.float64)

    mu = total_sum / total_count
    var = total_squared_sum / total_count - (mu**2)
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


def _reflect_window(
    context: torch.Tensor,
    future_xy: torch.Tensor,
    future_xy_raw: torch.Tensor,
    context_raw_xy: torch.Tensor,
    use_velocity: bool,
    shift_x: float,
    shift_y: float,
    flip_x: bool,
    flip_y: bool,
) -> WindowFields:
    """Apply a court reflection to one window for xy reflection augmentation.

    The dataset spans the full court with both attack directions in ~50/50 balance, so
    both reflections are valid label-preserving plays: ``y -> -y`` is the left/right
    mirror about the centerline, ``x -> -x`` reverses the attack direction. Applying each
    independently with its own probability yields the full dihedral group {I, Rx, Ry,
    RxRy}. Team/static labels are never flipped (basketball is symmetric under both).

    Tensors passed in are views into cached per-sequence tensors, so we clone before
    mutating. With ``n = (r - mu) / sigma``, mapping raw ``r -> -r`` along an axis gives
    ``n -> -n - 2*mu/sigma`` (= ``-n - shift``); velocity is a difference of normalized
    positions, so the shift cancels and that component simply negates.
    """
    return reflect_window(
        {
            "context": context,
            "future_xy": future_xy,
            "future_xy_raw": future_xy_raw,
            "context_raw_xy": context_raw_xy,
        },
        use_velocity=use_velocity,
        shift_x=shift_x,
        shift_y=shift_y,
        flip_x=flip_x,
        flip_y=flip_y,
    )


def reflect_context(
    context: torch.Tensor,
    use_velocity: bool,
    shift_x: float,
    shift_y: float,
    flip_x: bool,
    flip_y: bool,
) -> torch.Tensor:
    """Reflect a context feature window (the input arm of ``_reflect_window``).

    Position channels (0=x, 1=y) map ``n -> -n - shift``; the matching velocity channels
    (2=x, 3=y) simply negate (the shift cancels in a difference of positions). Static
    channels are untouched. Used by inference-time TTA; ``reflect_prob=0`` callers never
    reach here.
    """
    return reflect_context_tensor(
        context,
        use_velocity=use_velocity,
        shift_x=shift_x,
        shift_y=shift_y,
        flip_x=flip_x,
        flip_y=flip_y,
    )


def reflect_prediction_xy(
    prediction: torch.Tensor,
    shift_x: float,
    shift_y: float,
    flip_x: bool,
    flip_y: bool,
) -> torch.Tensor:
    """Apply the (involutive) reflection to predicted normalized xy positions.

    ``prediction`` has shape ``[..., 2]`` (channels 0=x, 1=y). The mirror ``n -> -n - shift``
    is its own inverse, so the same operation that reflects the context input also maps a
    prediction made in the reflected frame back to the original frame.
    """
    prediction = prediction.clone()
    if flip_x:
        prediction[..., 0] = -prediction[..., 0] - shift_x
    if flip_y:
        prediction[..., 1] = -prediction[..., 1] - shift_y
    return prediction


def _reflection_variants(reflect_x_prob: float, reflect_y_prob: float) -> list[tuple[bool, bool]]:
    x_values = (False, True) if reflect_x_prob > 0.0 else (False,)
    y_values = (False, True) if reflect_y_prob > 0.0 else (False,)
    return [(flip_x, flip_y) for flip_x in x_values for flip_y in y_values]


def build_training_pipeline(config: RunConfig, normalizer: Normalizer) -> TransformPipeline:
    transforms = _augmentation_transforms(config, normalizer)
    reflection = config.features.reflection
    if reflection.mode == "random":
        transforms.append(
            CourtReflectionTransform.from_normalization(
                normalizer,
                use_velocity=config.features.use_velocity,
                reflect_prob=reflection.y_prob,
                reflect_x_prob=reflection.x_prob,
            )
        )
    return TransformPipeline(tuple(transforms))


def _augmentation_transforms(config: RunConfig, normalizer: Normalizer) -> list[WindowTransform]:
    transforms: list[WindowTransform] = []
    augmentation = config.features.augmentation
    if augmentation.context_noise_std_ft > 0.0:
        transforms.append(
            ContextNoiseTransform.from_normalization(
                normalizer,
                use_velocity=config.features.use_velocity,
                std_ft=augmentation.context_noise_std_ft,
            )
        )
    if augmentation.context_frame_dropout_prob > 0.0:
        transforms.append(
            ContextFrameDropoutTransform(
                use_velocity=config.features.use_velocity,
                prob=augmentation.context_frame_dropout_prob,
                mode=augmentation.context_frame_dropout_mode,
            )
        )
    return transforms


class TrajectoryWindowSource(Dataset):
    def __init__(
        self,
        sequence_paths: Iterable[Path] | dict[int, torch.Tensor],
        normalizer: Normalizer,
        context_size: int,
        horizon_size: int,
        use_velocity: bool,
        transform_pipeline: TransformPipeline | None = None,
        entity_permutation_prob: float = 0.0,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.horizon_size = horizon_size
        self.window_size = context_size + horizon_size
        self.use_velocity = use_velocity
        self.window_transform: WindowTransform = transform_pipeline or TransformPipeline()
        self.entity_permutation_prob = entity_permutation_prob
        if not 0.0 <= entity_permutation_prob <= 1.0:
            raise ValueError("features.augmentation.agent_permutation_prob must be in [0, 1].")
        self._gen = torch.Generator().manual_seed(seed)
        self.sequences = sequence_paths
        self.samples: list[SequenceSample] = []
        self._load_sequences(normalizer)

    def _load_sequences(self, normalizer: Normalizer) -> None:
        for seq_id, seq in _iter_sequences(self.sequences):
            if seq.shape[0] < self.window_size:
                continue
            features, raw_xy = _build_feature_tensor(seq, normalizer, self.use_velocity)
            static = seq[0, :, 2:].float()
            sample: SequenceSample = {
                "sequence_id": seq_id,
                "features": features,
                "raw_xy": raw_xy,
                "static": static,
            }
            self.samples.append(sample)

    def _window_for(self, sample_idx: int, start: int, reflection_variant: tuple[bool, bool] | None = None) -> WindowSample:
        sample = self.samples[sample_idx]
        features = sample["features"]
        raw_xy = sample["raw_xy"]
        context_end = start + self.context_size
        future_end = context_end + self.horizon_size
        window: WindowFields = {
            "context": features[start:context_end],
            "future_xy": features[context_end:future_end, :, :2],
            "future_xy_raw": raw_xy[context_end:future_end],
            "context_raw_xy": raw_xy[start:context_end],
        }
        window = self._apply_window_transform(window, reflection_variant)
        static = sample["static"]
        if self.entity_permutation_prob > 0.0 and torch.rand((), generator=self._gen).item() < self.entity_permutation_prob:
            window, static = self._permute_window_entities(window, static)
        return {
            "sequence_id": int(sample["sequence_id"]),
            "start_idx": start,
            "context": window["context"],
            "future_xy": window["future_xy"],
            "future_xy_raw": window["future_xy_raw"],
            "context_raw_xy": window["context_raw_xy"],
            "static": static,
        }

    def _apply_window_transform(self, window: WindowFields, reflection_variant: tuple[bool, bool] | None) -> WindowFields:
        del reflection_variant
        return self.window_transform.apply_window(window, self._gen)

    def _permute_window_entities(self, window: WindowFields, static: torch.Tensor) -> tuple[WindowFields, torch.Tensor]:
        num_entities = static.shape[0]
        ball_candidates = torch.nonzero(static[:, 1] == 0, as_tuple=False).flatten()
        ball_idx = int(ball_candidates[0].item()) if ball_candidates.numel() == 1 else -1
        player_indices = [idx for idx in range(num_entities) if idx != ball_idx]
        shuffled = torch.tensor(player_indices, dtype=torch.long)
        shuffled = shuffled[torch.randperm(len(player_indices), generator=self._gen)]
        perm = torch.arange(num_entities, dtype=torch.long)
        perm[player_indices] = shuffled
        if ball_idx >= 0:
            perm[ball_idx] = ball_idx
        return (
            {
                "context": window["context"][:, perm],
                "future_xy": window["future_xy"][:, perm],
                "future_xy_raw": window["future_xy_raw"][:, perm],
                "context_raw_xy": window["context_raw_xy"][:, perm],
            },
            static[perm],
        )


class SequenceWindowDataset(TrajectoryWindowSource):
    def __init__(
        self,
        sequence_paths: Iterable[Path] | dict[int, torch.Tensor],
        normalizer: Normalizer,
        context_size: int,
        horizon_size: int,
        use_velocity: bool,
        reflect_prob: float = 0.0,
        reflect_x_prob: float = 0.0,
        reflection_mode: str = "random",
        transform_pipeline: TransformPipeline | None = None,
        entity_permutation_prob: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.reflection_transform = CourtReflectionTransform.from_normalization(
            normalizer,
            use_velocity=use_velocity,
            reflect_prob=reflect_prob,
            reflect_x_prob=reflect_x_prob,
        )
        self.reflection_mode = reflection_mode
        if reflection_mode not in {"random", "expand"}:
            raise ValueError("features.reflection.mode must be one of: random, expand.")
        pipeline = transform_pipeline or TransformPipeline((self.reflection_transform,) if reflection_mode == "random" else ())
        self.reflection_variants = _reflection_variants(reflect_x_prob, reflect_prob) if reflection_mode == "expand" else [(False, False)]
        super().__init__(
            sequence_paths=sequence_paths,
            normalizer=normalizer,
            context_size=context_size,
            horizon_size=horizon_size,
            use_velocity=use_velocity,
            transform_pipeline=pipeline,
            entity_permutation_prob=entity_permutation_prob,
            seed=seed,
        )
        self.window_index: list[tuple[int, int, int]] = []
        for sample_idx, sample in enumerate(self.samples):
            features = sample["features"]
            max_start = features.shape[0] - self.window_size
            self.window_index.extend(
                (sample_idx, start, variant_idx) for start in range(max_start + 1) for variant_idx in range(len(self.reflection_variants))
            )

    def __len__(self) -> int:
        return len(self.window_index)

    def __getitem__(self, index: int) -> WindowSample:
        sample_idx, start, variant_idx = self.window_index[index]
        return self._window_for(sample_idx, start, self.reflection_variants[variant_idx])

    def _apply_window_transform(self, window: WindowFields, reflection_variant: tuple[bool, bool] | None) -> WindowFields:
        if self.reflection_mode != "expand":
            return super()._apply_window_transform(window, reflection_variant)
        window = self.window_transform.apply_window(window, self._gen)
        assert reflection_variant is not None
        flip_x, flip_y = reflection_variant
        if not flip_x and not flip_y:
            return window
        return reflect_window(
            window,
            use_velocity=self.use_velocity,
            shift_x=self.reflection_transform.shift_x,
            shift_y=self.reflection_transform.shift_y,
            flip_x=flip_x,
            flip_y=flip_y,
        )


class NotebookSequenceDataset(TrajectoryWindowSource):
    def __init__(
        self,
        sequence_paths: Iterable[Path] | dict[int, torch.Tensor],
        normalizer: Normalizer,
        context_size: int,
        horizon_size: int,
        use_velocity: bool,
        reflect_prob: float = 0.0,
        reflect_x_prob: float = 0.0,
        reflection_mode: str = "random",
        transform_pipeline: TransformPipeline | None = None,
        entity_permutation_prob: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.reflection_transform = CourtReflectionTransform.from_normalization(
            normalizer,
            use_velocity=use_velocity,
            reflect_prob=reflect_prob,
            reflect_x_prob=reflect_x_prob,
        )
        self.reflection_mode = reflection_mode
        if reflection_mode not in {"random", "expand"}:
            raise ValueError("features.reflection.mode must be one of: random, expand.")
        pipeline = transform_pipeline or TransformPipeline((self.reflection_transform,) if reflection_mode == "random" else ())
        self.reflection_variants = _reflection_variants(reflect_x_prob, reflect_prob) if reflection_mode == "expand" else [(False, False)]
        super().__init__(
            sequence_paths=sequence_paths,
            normalizer=normalizer,
            context_size=context_size,
            horizon_size=horizon_size,
            use_velocity=use_velocity,
            transform_pipeline=pipeline,
            entity_permutation_prob=entity_permutation_prob,
            seed=seed,
        )
        self.max_start = [max(0, sample["features"].shape[0] - self.window_size) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples) * len(self.reflection_variants)

    def __getitem__(self, index: tuple[int, int] | int) -> WindowSample:
        if isinstance(index, tuple):
            sample_idx, start = index
            variant_idx = 0
        else:
            sample_idx = index // len(self.reflection_variants)
            variant_idx = index % len(self.reflection_variants)
            start = 0
        return self._window_for(sample_idx, start, self.reflection_variants[variant_idx])

    def _apply_window_transform(self, window: WindowFields, reflection_variant: tuple[bool, bool] | None) -> WindowFields:
        if self.reflection_mode != "expand":
            return super()._apply_window_transform(window, reflection_variant)
        window = self.window_transform.apply_window(window, self._gen)
        assert reflection_variant is not None
        flip_x, flip_y = reflection_variant
        if not flip_x and not flip_y:
            return window
        return reflect_window(
            window,
            use_velocity=self.use_velocity,
            shift_x=self.reflection_transform.shift_x,
            shift_y=self.reflection_transform.shift_y,
            flip_x=flip_x,
            flip_y=flip_y,
        )


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
    def __init__(self, test_dir: str | Path | dict[int, torch.Tensor], normalizer: Normalizer, use_velocity: bool) -> None:
        super().__init__()
        self.sequences = test_dir
        self.samples: list[dict[str, torch.Tensor | int]] = []
        self.use_velocity = use_velocity
        self._load_sequences(normalizer)

    def _load_sequences(self, normalizer: Normalizer) -> None:
        if isinstance(self.sequences, str | Path):
            sequences = load_sequences_from_dir_or_consolidated(self.sequences)
        else:
            sequences = self.sequences

        for seq_id, seq in _iter_sequences(sequences):
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


def _denormalize_predictions(prediction: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
    mu = normalizer.mu.to(prediction.device).view(1, 1, 1, 2)
    sigma = normalizer.sigma.to(prediction.device).view(1, 1, 1, 2)
    return prediction * sigma + mu


def prepare_split_and_normalizer(config: RunConfig) -> tuple[dict[str, list[int]], dict[int, torch.Tensor], Normalizer]:
    train_dir = Path(config.experiment.data_dir) / "train"
    split = freeze_sequence_split(train_dir, config.experiment.split_path, seed=config.experiment.seed)
    all_sequences = load_sequences_from_dir_or_consolidated(train_dir)
    normalizer = compute_normalizer(all_sequences[seq_id] for seq_id in split["train"])
    return split, all_sequences, normalizer


def build_dataloaders(
    config: RunConfig,
    split: dict[str, list[int]],
    all_sequences: dict[int, torch.Tensor],
    normalizer: Normalizer,
    *,
    num_workers: int | None = None,
    dataloader_kwargs: dict[str, int | bool] | None = None,
) -> tuple[SequenceWindowDataset | NotebookSequenceDataset, SequenceWindowDataset | NotebookSequenceDataset, DataLoader, DataLoader]:
    from .runtime_env import configure_training_cpu_runtime, dataloader_worker_kwargs

    if num_workers is None:
        cpu_runtime = configure_training_cpu_runtime(config.training.num_workers)
        resolved_workers = cpu_runtime.num_workers
        loader_kwargs = dataloader_kwargs or dataloader_worker_kwargs(resolved_workers)
    else:
        resolved_workers = num_workers
        loader_kwargs = dataloader_kwargs or dataloader_worker_kwargs(resolved_workers)

    train_reflection = config.features.reflection

    def build_split_loader(
        sequence_ids: list[int],
        sampling_strategy: str,
        shuffle: bool,
        reflect_prob: float = 0.0,
        reflect_x_prob: float = 0.0,
        reflection_mode: str = "random",
        transform_pipeline: TransformPipeline | None = None,
        entity_permutation_prob: float = 0.0,
    ) -> tuple[SequenceWindowDataset | NotebookSequenceDataset, DataLoader]:
        sequence_data = {seq_id: all_sequences[seq_id] for seq_id in sequence_ids}
        if sampling_strategy == "all_windows":
            dataset = SequenceWindowDataset(
                sequence_paths=sequence_data,
                normalizer=normalizer,
                context_size=config.experiment.context_size,
                horizon_size=config.experiment.horizon_size,
                use_velocity=config.features.use_velocity,
                reflect_prob=reflect_prob,
                reflect_x_prob=reflect_x_prob,
                reflection_mode=reflection_mode,
                transform_pipeline=transform_pipeline,
                entity_permutation_prob=entity_permutation_prob,
                seed=config.experiment.seed,
            )
            loader = DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=shuffle,
                num_workers=resolved_workers,
                **loader_kwargs,
            )
            return dataset, loader
        if sampling_strategy == "random_window_per_sequence":
            if reflection_mode == "expand":
                raise ValueError("features.reflection.mode='expand' requires training.train_window_sampling='all_windows'.")
            dataset = NotebookSequenceDataset(
                sequence_paths=sequence_data,
                normalizer=normalizer,
                context_size=config.experiment.context_size,
                horizon_size=config.experiment.horizon_size,
                use_velocity=config.features.use_velocity,
                reflect_prob=reflect_prob,
                reflect_x_prob=reflect_x_prob,
                reflection_mode=reflection_mode,
                transform_pipeline=transform_pipeline,
                entity_permutation_prob=entity_permutation_prob,
                seed=config.experiment.seed,
            )
            loader = DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                sampler=NotebookRandomWindowSampler(dataset.max_start, seed=config.experiment.seed, shuffle=shuffle),
                num_workers=resolved_workers,
                **loader_kwargs,
            )
            return dataset, loader
        raise ValueError(f"Unknown window sampling strategy {sampling_strategy!r}.")

    train_dataset, train_loader = build_split_loader(
        split["train"],
        config.training.train_window_sampling,
        shuffle=True,
        reflect_prob=train_reflection.y_prob,
        reflect_x_prob=train_reflection.x_prob,
        reflection_mode=train_reflection.mode,
        transform_pipeline=build_training_pipeline(config, normalizer),
        entity_permutation_prob=config.features.augmentation.agent_permutation_prob,
    )
    val_dataset, val_loader = build_split_loader(
        split["val"],
        config.training.val_window_sampling,
        shuffle=False,
        reflect_prob=0.0,
        reflect_x_prob=0.0,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def build_eval_dataloader(
    config: RunConfig,
    dataset: torch.utils.data.Dataset,
    *,
    shuffle: bool = False,
) -> DataLoader:
    from .runtime_env import configure_training_cpu_runtime, dataloader_worker_kwargs

    cpu_runtime = configure_training_cpu_runtime(config.training.num_workers)
    workers = cpu_runtime.num_workers
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=shuffle,
        num_workers=workers,
        **dataloader_worker_kwargs(workers),
    )
