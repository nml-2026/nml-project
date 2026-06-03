from __future__ import annotations

import copy
import csv
import json
import os
import random
import shutil
import statistics
import sys
import time
import traceback
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - fallback for minimal environments
    tqdm = None

from .config import RunConfig, apply_overrides, load_config, resolve_config_paths
from .data import (
    Normalizer,
    NotebookRandomWindowSampler,
    NotebookSequenceDataset,
    SequenceWindowDataset,
    TestSequenceDataset,
    compute_normalizer,
    freeze_kfold_sequence_splits,
    freeze_sequence_split,
    load_sequences_from_dir_or_consolidated,
)
from .metrics import compute_raw_metrics, normalized_mse_loss, per_example_mse
from .models import build_model
from .plots import save_history_plot, save_prediction_plot


@dataclass(slots=True)
class ExampleRecord:
    error: float
    sequence_id: int
    start_idx: int
    context_xy: torch.Tensor
    target_xy: torch.Tensor
    pred_xy: torch.Tensor
    static: torch.Tensor


METRIC_KEYS = ("loss", "mse", "ade", "rmse")


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def update(self, source_model: torch.nn.Module) -> None:
        with torch.no_grad():
            ema_state = self.model.state_dict()
            source_state = source_model.state_dict()
            for key, ema_value in ema_state.items():
                source_value = source_state[key].detach()
                if torch.is_floating_point(ema_value):
                    ema_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
                else:
                    ema_value.copy_(source_value)


def _validation_model(
    model: torch.nn.Module,
    ema: ExponentialMovingAverage | None,
    swa_model: AveragedModel | None,
    swa_active: bool,
) -> tuple[torch.nn.Module, str]:
    if swa_model is not None and swa_active:
        return swa_model.module, "swa"
    if ema is not None:
        return ema.model, "ema"
    return model, "raw"


def _progress_disabled() -> bool:
    if os.environ.get("NBA_FORECASTING_DISABLE_TQDM") == "1":
        return True
    return not sys.stderr.isatty()


def _progress_bar(iterable, *, total: int | None, desc: str, leave: bool = False):
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        disable=_progress_disabled(),
    )


def _mixed_precision_enabled(device: torch.device, requested: bool) -> bool:
    return requested and device.type == "cuda"


def _cuda_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()


def _cuda_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def _model_parameter_l2(model: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                total += float(parameter.detach().float().square().sum().item())
    return total**0.5


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this environment. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}, "
            f"torch.version.cuda={torch.version.cuda!r}, "
            f"torch.cuda.device_count()={torch.cuda.device_count()}."
        )
    if device_arg == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this environment.")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _configure_torch_dynamo_for_compile() -> None:
    """Raise Dynamo cache limits so graph-transformer forward does not thrash recompiles."""
    import torch._dynamo

    if torch._dynamo.config.cache_size_limit < 64:
        torch._dynamo.config.cache_size_limit = 64
    accumulated = getattr(torch._dynamo.config, "accumulated_cache_size_limit", None)
    if isinstance(accumulated, int) and accumulated < 256:
        torch._dynamo.config.accumulated_cache_size_limit = 256


def _apply_optimization_backend(model: torch.nn.Module, backend: str) -> torch.nn.Module:
    if backend == "none":
        return model
    if backend == "compile":
        _configure_torch_dynamo_for_compile()
        return torch.compile(model, mode="reduce-overhead")
    if backend == "compile_default":
        _configure_torch_dynamo_for_compile()
        return torch.compile(model, mode="default")
    if backend == "cudagraph":
        raise NotImplementedError(
            "optimization_backend='cudagraph' requires static-buffer training-loop capture and is not wired yet. "
            "Use optimization_backend='compile' for CUDA Graph based reduce-overhead capture."
        )
    raise ValueError("optimization_backend must be one of: none, compile, compile_default, cudagraph.")


def _build_lr_scheduler(
    config: RunConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.training.lr_scheduler == "none":
        return None
    if config.training.lr_scheduler == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.training.lr_milestones,
            gamma=config.training.lr_gamma,
        )
    raise ValueError("training.lr_scheduler must be one of: none, multistep.")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_history_csv(path: str | Path, history: list[dict[str, float]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = list(history[0].keys())
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _write_rows_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _prepare_split_and_normalizer(config: RunConfig) -> tuple[dict[str, list[int]], dict[int, torch.Tensor], Normalizer]:
    train_dir = Path(config.experiment.data_dir) / "train"
    split = freeze_sequence_split(train_dir, config.experiment.split_path, seed=config.experiment.seed)
    all_sequences = load_sequences_from_dir_or_consolidated(train_dir)
    normalizer = compute_normalizer(all_sequences[seq_id] for seq_id in split["train"])
    return split, all_sequences, normalizer


def _build_dataloaders(
    config: RunConfig,
    split: dict[str, list[int]],
    all_sequences: dict[int, torch.Tensor],
    normalizer: Normalizer,
) -> tuple[Dataset, Dataset, DataLoader, DataLoader]:
    def build_split_loader(
        sequence_ids: list[int],
        sampling_strategy: str,
        shuffle: bool,
    ) -> tuple[Dataset, DataLoader]:
        sequence_data = {seq_id: all_sequences[seq_id] for seq_id in sequence_ids}
        if sampling_strategy == "all_windows":
            dataset = SequenceWindowDataset(
                sequence_paths=sequence_data,
                normalizer=normalizer,
                context_size=config.experiment.context_size,
                horizon_size=config.experiment.horizon_size,
                use_velocity=config.features.use_velocity,
            )
            loader = DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                shuffle=shuffle,
                num_workers=config.training.num_workers,
            )
            return dataset, loader
        if sampling_strategy == "random_window_per_sequence":
            dataset = NotebookSequenceDataset(
                sequence_paths=sequence_data,
                normalizer=normalizer,
                context_size=config.experiment.context_size,
                horizon_size=config.experiment.horizon_size,
                use_velocity=config.features.use_velocity,
            )
            loader = DataLoader(
                dataset,
                batch_size=config.training.batch_size,
                sampler=NotebookRandomWindowSampler(dataset.max_start, seed=config.experiment.seed, shuffle=shuffle),
                num_workers=config.training.num_workers,
            )
            return dataset, loader
        raise ValueError(
            f"Unknown window sampling strategy {sampling_strategy!r}. Expected one of {{'all_windows', 'random_window_per_sequence'}}."
        )

    train_dataset, train_loader = build_split_loader(
        split["train"],
        config.training.train_window_sampling,
        shuffle=True,
    )
    val_dataset, val_loader = build_split_loader(
        split["val"],
        config.training.val_window_sampling,
        shuffle=False,
    )
    return train_dataset, val_dataset, train_loader, val_loader


def _denormalize_predictions(prediction: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
    mu = normalizer.mu.to(prediction.device).view(1, 1, 1, 2)
    sigma = normalizer.sigma.to(prediction.device).view(1, 1, 1, 2)
    return prediction * sigma + mu


def _update_example_records(
    records: list[ExampleRecord],
    batch: dict[str, torch.Tensor],
    prediction_raw: torch.Tensor,
    errors: torch.Tensor,
    keep: int,
    reverse: bool,
) -> list[ExampleRecord]:
    for idx in range(errors.shape[0]):
        records.append(
            ExampleRecord(
                error=float(errors[idx].item()),
                sequence_id=int(batch["sequence_id"][idx].item()),
                start_idx=int(batch["start_idx"][idx].item()),
                context_xy=batch["context_raw_xy"][idx].detach().cpu(),
                target_xy=batch["future_xy_raw"][idx].detach().cpu(),
                pred_xy=prediction_raw[idx].detach().cpu(),
                static=batch["static"][idx].detach().cpu(),
            )
        )
    records.sort(key=lambda record: record.error, reverse=reverse)
    return records[:keep]


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    num_examples: int = 0,
    max_batches: int | None = None,
    mixed_precision: bool = False,
) -> tuple[dict[str, float], dict[str, list[ExampleRecord]]]:
    model.eval()
    totals = {"loss": 0.0, "mse": 0.0, "ade": 0.0, "rmse": 0.0}
    batches = 0
    best_examples: list[ExampleRecord] = []
    worst_examples: list[ExampleRecord] = []
    total_batches = max_batches if max_batches is not None else len(dataloader)
    with torch.no_grad():
        iterator = _progress_bar(
            enumerate(dataloader, start=1),
            total=total_batches,
            desc="eval",
            leave=False,
        )
        for batch_idx, batch in iterator:
            context = batch["context"].to(device)
            target = batch["future_xy"].to(device)
            with _cuda_autocast(mixed_precision):
                prediction = model(context)
                loss = normalized_mse_loss(prediction, target)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Model produced NaN or Inf predictions during evaluation.")
            prediction_raw = _denormalize_predictions(prediction, normalizer)
            target_raw = batch["future_xy_raw"].to(device)
            metrics = compute_raw_metrics(prediction_raw, target_raw)
            totals["loss"] += float(loss.item())
            for key, value in metrics.items():
                totals[key] += value
            if tqdm is not None and not _progress_disabled():
                iterator.set_postfix(loss=f"{loss.item():.4f}", ade=f"{metrics['ade']:.2f}")
            if num_examples > 0:
                errors = per_example_mse(prediction_raw, target_raw)
                best_examples = _update_example_records(
                    best_examples, batch, prediction_raw.cpu(), errors.cpu(), keep=num_examples, reverse=False
                )
                worst_examples = _update_example_records(
                    worst_examples, batch, prediction_raw.cpu(), errors.cpu(), keep=num_examples, reverse=True
                )
            batches += 1
            if max_batches is not None and batch_idx >= max_batches:
                break
    averaged = {key: value / max(batches, 1) for key, value in totals.items()}
    return averaged, {"best": best_examples, "worst": worst_examples}


def _train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    teacher_forcing_prob: float,
    scheduled_sampling_prob: float,
    ema: ExponentialMovingAverage | None = None,
    max_batches: int | None = None,
    scaler: Any | None = None,
    mixed_precision: bool = False,
    timing_callback=None,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    total_batches = max_batches if max_batches is not None else len(dataloader)
    iterator = _progress_bar(
        enumerate(dataloader, start=1),
        total=total_batches,
        desc="train",
        leave=False,
    )
    for batch_idx, batch in iterator:
        if batch_idx == 1 and timing_callback is not None:
            timing_callback("first_batch_retrieved")
        context = batch["context"].to(device)
        target = batch["future_xy"].to(device)
        if batch_idx == 1 and timing_callback is not None:
            timing_callback("first_step_start")
        with _cuda_autocast(mixed_precision):
            prediction = model(
                context,
                future_xy=target,
                teacher_forcing_prob=teacher_forcing_prob,
                scheduled_sampling_prob=scheduled_sampling_prob,
            )
            loss = normalized_mse_loss(prediction, target)
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Model produced NaN or Inf predictions during training.")
        optimizer.zero_grad()
        if scaler is not None and mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if ema is not None:
            ema.update(model)
        if batch_idx == 1 and timing_callback is not None:
            timing_callback("first_forward_backward_optimizer_step_done", model_param_l2=_model_parameter_l2(model))
        total_loss += float(loss.item())
        steps += 1
        if tqdm is not None and not _progress_disabled():
            iterator.set_postfix(loss=f"{loss.item():.4f}")
        if max_batches is not None and batch_idx >= max_batches:
            break
    return total_loss / max(steps, 1)


def _scheduled_sampling_probability(config: RunConfig, epoch: int) -> float:
    final_prob = config.training.scheduled_sampling_final_prob
    if final_prob <= 0.0:
        return 0.0
    if config.training.epochs <= 1:
        return final_prob
    return final_prob * float(epoch - 1) / float(config.training.epochs - 1)


def _optional_batch_limit(raw_limit: int | None) -> int | None:
    if raw_limit is None or raw_limit <= 0:
        return None
    return raw_limit


def _run_dir(config: RunConfig) -> Path:
    root = Path(config.experiment.output_root) / config.experiment.name / _timestamp()
    root.mkdir(parents=True, exist_ok=False)
    return root.resolve()


def _checkpoint_run_dir(resume_state: dict[str, Any] | None) -> Path | None:
    if resume_state is None:
        return None
    raw_run_dir = resume_state.get("run_dir")
    if not raw_run_dir:
        return None
    return Path(raw_run_dir).expanduser().resolve()


def _assert_resume_compatible(config: RunConfig, resume_state: dict[str, Any], resume_checkpoint: Path) -> None:
    checkpoint_config = resume_state.get("config", {})
    checkpoint_experiment = checkpoint_config.get("experiment", {})
    expected = {
        "model_name": config.experiment.model_name,
        "context_size": config.experiment.context_size,
        "horizon_size": config.experiment.horizon_size,
        "num_entities": config.experiment.num_entities,
    }
    actual = {key: checkpoint_experiment.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"Resume checkpoint {resume_checkpoint} is not compatible with the requested config. Expected {expected}, found {actual}."
        )


def _training_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any | None,
    scaler: Any | None,
    ema: ExponentialMovingAverage | None,
    swa_model: AveragedModel | None,
    config: RunConfig,
    split: dict[str, list[int]],
    normalizer: Normalizer,
    epoch: int,
    best_epoch: int,
    best_metric: float,
    patience: int,
    history: list[dict[str, float]],
    mixed_precision: bool,
    run_dir: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "config": config.to_dict(),
        "split": split,
        "normalizer": normalizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "checkpoint_metric": config.experiment.checkpoint_metric,
        "patience": patience,
        "history": history,
        "mixed_precision": mixed_precision,
        "run_dir": str(run_dir),
    }
    if lr_scheduler is not None:
        payload["lr_scheduler_state"] = lr_scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if ema is not None:
        payload["ema_model_state"] = ema.model.state_dict()
        payload["ema_decay"] = ema.decay
    if swa_model is not None:
        payload["swa_model_state"] = swa_model.state_dict()
    return payload


def _load_runtime_config(
    config_arg: str | Path,
    overrides: dict[str, object] | None = None,
) -> tuple[RunConfig, Path]:
    config, config_path = load_config(config_arg)
    if overrides:
        config = apply_overrides(config, overrides)
    config = resolve_config_paths(config, config_path)
    return config, config_path


def _evaluate_run_checkpoint(
    run_dir: str | Path,
    split_name: str,
    device_arg: str,
) -> tuple[Path, dict[str, float]]:
    checkpoint_path = Path(run_dir) / "best.pt"
    eval_dir = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        split_name=split_name,
        device_arg=device_arg,
    )
    metrics = _read_json(eval_dir / "metrics.json")
    return eval_dir, metrics


def _aggregate_split_metrics(metric_rows: list[dict[str, float]], split_name: str) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in METRIC_KEYS:
        values = [float(row[key]) for row in metric_rows if key in row]
        if not values:
            continue
        prefix = f"{split_name}_{key}"
        summary[f"{prefix}_mean"] = statistics.fmean(values)
        summary[f"{prefix}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary[f"{prefix}_min"] = min(values)
        summary[f"{prefix}_max"] = max(values)
    return summary


def _candidate_name(base_name: str, label: str) -> str:
    safe_label = label.replace("/", "_").replace(" ", "_")
    return f"{base_name}_{safe_label}"


def train_experiment(
    config_arg: str | Path,
    overrides: dict[str, object] | None = None,
    epoch_reporter=None,
    timing_callback=None,
    resume_checkpoint: str | Path | None = None,
) -> Path:
    if timing_callback is not None:
        timing_callback("train_experiment_start")
    config, config_path = _load_runtime_config(config_arg, overrides)
    if timing_callback is not None:
        timing_callback("train_config_loaded")
    set_seed(config.experiment.seed)
    device = resolve_device(config.experiment.device)
    print("Using device:", device)
    mixed_precision = _mixed_precision_enabled(device, config.training.mixed_precision)
    print("Mixed precision:", "enabled" if mixed_precision else "disabled")
    resume_state: dict[str, Any] | None = None
    resume_path: Path | None = None
    if resume_checkpoint is not None:
        resume_path = Path(resume_checkpoint).expanduser().resolve()
        if timing_callback is not None:
            timing_callback("checkpoint_restore_start", path=str(resume_path))
        resume_state = torch.load(resume_path, map_location=device, weights_only=False)
        _assert_resume_compatible(config, resume_state, resume_path)
    if timing_callback is not None:
        timing_callback("dataset_loading_start")
    split, all_sequences, normalizer = _prepare_split_and_normalizer(config)
    if timing_callback is not None:
        timing_callback("dataset_loading_done")
        timing_callback("dataloader_construction_start")
    train_dataset, val_dataset, train_loader, val_loader = _build_dataloaders(config, split, all_sequences, normalizer)
    if timing_callback is not None:
        timing_callback("dataloader_construction_done")
        timing_callback("model_construction_start")
    model = build_model(config)
    if timing_callback is not None:
        timing_callback("model_construction_done")
        timing_callback("first_cuda_operation_start")
    model = model.to(device)
    model = _apply_optimization_backend(model, config.model.optimization_backend)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if timing_callback is not None:
        timing_callback("first_cuda_operation_done")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    lr_scheduler = _build_lr_scheduler(config, optimizer)
    ema = ExponentialMovingAverage(model, config.training.ema_decay) if config.training.ema_decay > 0.0 else None
    swa_model = AveragedModel(model) if config.training.swa_start_pct > 0.0 else None
    swa_start_epoch = max(1, int(config.training.epochs * config.training.swa_start_pct)) if swa_model else 0
    scaler = _cuda_grad_scaler(mixed_precision) if mixed_precision else None
    best_metric = float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, float]] = []
    start_epoch = 1
    if resume_state is not None and resume_path is not None:
        model.load_state_dict(resume_state["model_state"])
        if "optimizer_state" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer_state"])
        if lr_scheduler is not None and "lr_scheduler_state" in resume_state:
            lr_scheduler.load_state_dict(resume_state["lr_scheduler_state"])
        if scaler is not None and "scaler_state" in resume_state:
            scaler.load_state_dict(resume_state["scaler_state"])
        if ema is not None and "ema_model_state" in resume_state:
            ema.model.load_state_dict(resume_state["ema_model_state"])
        if swa_model is not None and "swa_model_state" in resume_state:
            swa_model.load_state_dict(resume_state["swa_model_state"])
        best_metric = float(resume_state.get("best_metric", float("inf")))
        best_epoch = int(resume_state.get("best_epoch", 0))
        patience = int(resume_state.get("patience", 0))
        history = list(resume_state.get("history", []))
        start_epoch = int(resume_state.get("epoch", 0)) + 1
        if timing_callback is not None:
            timing_callback("checkpoint_restore_done", path=str(resume_path), start_epoch=start_epoch)
    initial_param_l2 = _model_parameter_l2(model)
    if timing_callback is not None:
        timing_callback("initial_model_state", model_param_l2=initial_param_l2)
    run_dir = _checkpoint_run_dir(resume_state) or _run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)
    if timing_callback is not None:
        timing_callback("run_dir_created", run_dir=str(run_dir))
    shutil.copy2(config_path, run_dir / "config_source.toml")
    _write_json(run_dir / "resolved_config.json", config.to_dict())
    _write_json(
        run_dir / "dataset_summary.json",
        {
            "train_sequences": len(split["train"]),
            "val_sequences": len(split["val"]),
            "holdout_sequences": len(split["holdout"]),
            "train_windows": len(train_dataset),
            "val_windows": len(val_dataset),
        },
    )

    epoch_iterator = _progress_bar(
        range(start_epoch, config.training.epochs + 1),
        total=max(0, config.training.epochs - start_epoch + 1),
        desc=f"epochs:{config.experiment.name}",
        leave=True,
    )
    for epoch in epoch_iterator:
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch - 1)
        if hasattr(val_loader.sampler, "set_epoch"):
            val_loader.sampler.set_epoch(epoch - 1)
        scheduled_sampling_prob = _scheduled_sampling_probability(config, epoch)
        train_loss = _train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=config.training.grad_clip,
            teacher_forcing_prob=config.training.teacher_forcing_prob,
            scheduled_sampling_prob=scheduled_sampling_prob,
            ema=ema,
            max_batches=_optional_batch_limit(config.training.max_train_batches),
            scaler=scaler,
            mixed_precision=mixed_precision,
            timing_callback=timing_callback if epoch == 1 else None,
        )
        if swa_model is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
        epoch_param_l2 = _model_parameter_l2(model)
        eval_model, eval_model_kind = _validation_model(
            model=model,
            ema=ema,
            swa_model=swa_model,
            swa_active=swa_model is not None and epoch >= swa_start_epoch,
        )
        val_metrics, _ = evaluate_model(
            model=eval_model,
            dataloader=val_loader,
            normalizer=normalizer,
            device=device,
            num_examples=0,
            max_batches=_optional_batch_limit(config.training.max_val_batches),
            mixed_precision=mixed_precision,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_ade": val_metrics["ade"],
            "val_rmse": val_metrics["rmse"],
            "scheduled_sampling_prob": scheduled_sampling_prob,
            "lr": optimizer.param_groups[0]["lr"],
            "model_param_l2": epoch_param_l2,
            "model_param_l2_delta": epoch_param_l2 - initial_param_l2,
        }
        history.append(row)
        _write_history_csv(run_dir / "history.csv", history)
        save_history_plot(history, run_dir / "history.png")
        if tqdm is not None and not _progress_disabled():
            epoch_iterator.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_metrics['ade']:.2f}",
                model=eval_model_kind,
            )
        current_metric = row[f"val_{config.experiment.checkpoint_metric}"]
        if current_metric < best_metric:
            best_metric = current_metric
            best_epoch = epoch
            patience = 0
            if timing_callback is not None:
                timing_callback("checkpoint_write_start", path=str(run_dir / "best.pt"), epoch=epoch)
            torch.save(
                {
                    "model_state": eval_model.state_dict(),
                    "config": config.to_dict(),
                    "split": split,
                    "normalizer": normalizer.state_dict(),
                    "best_metrics": val_metrics,
                    "epoch": epoch,
                    "eval_model_kind": eval_model_kind,
                },
                run_dir / "best.pt",
            )
            if timing_callback is not None:
                timing_callback("checkpoint_write_done", path=str(run_dir / "best.pt"), epoch=epoch)
        else:
            patience += 1
        if epoch_reporter is not None:
            report_payload = dict(row)
            report_payload["best_epoch"] = best_epoch
            report_payload[f"best_val_{config.experiment.checkpoint_metric}"] = best_metric
            report_payload["checkpoint_metric"] = current_metric
            report_payload["run_dir"] = str(run_dir)
            epoch_reporter(report_payload)
        if lr_scheduler is not None:
            lr_scheduler.step()
        torch.save(
            _training_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                scaler=scaler,
                ema=ema,
                swa_model=swa_model,
                config=config,
                split=split,
                normalizer=normalizer,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                patience=patience,
                history=history,
                mixed_precision=mixed_precision,
                run_dir=run_dir,
            ),
            run_dir / "last.pt",
        )
        if patience >= config.training.patience:
            break

    summary = {
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "checkpoint_metric": config.experiment.checkpoint_metric,
        "device": str(device),
        "history_rows": len(history),
        "run_dir": str(run_dir),
        "ema_decay": config.training.ema_decay,
        "swa_start_pct": config.training.swa_start_pct,
        "mixed_precision": mixed_precision,
    }
    _write_json(run_dir / "train_summary.json", summary)
    return run_dir


def _benchmark_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    mixed_precision: bool,
    grad_clip: float,
) -> dict[str, float]:
    timings: dict[str, float] = {}
    _synchronize_device(device)
    started = time.perf_counter()
    context = batch["context"].to(device)
    target = batch["future_xy"].to(device)
    _synchronize_device(device)
    timings["transfer_s"] = time.perf_counter() - started

    started = time.perf_counter()
    with _cuda_autocast(mixed_precision):
        prediction = model(context, future_xy=target)
        loss = normalized_mse_loss(prediction, target)
    _synchronize_device(device)
    timings["forward_s"] = time.perf_counter() - started

    started = time.perf_counter()
    optimizer.zero_grad()
    if scaler is not None and mixed_precision:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
    _synchronize_device(device)
    timings["backward_optimizer_s"] = time.perf_counter() - started
    timings["loss"] = float(loss.item())
    return timings


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def benchmark_experiment(
    config_arg: str | Path,
    overrides: dict[str, object] | None = None,
    warmup_steps: int = 5,
    steps: int = 20,
    profile: bool = False,
    output_dir: str | Path | None = None,
) -> Path:
    config, _ = _load_runtime_config(config_arg, overrides)
    set_seed(config.experiment.seed)
    device = resolve_device(config.experiment.device)
    mixed_precision = _mixed_precision_enabled(device, config.training.mixed_precision)
    split, all_sequences, normalizer = _prepare_split_and_normalizer(config)
    _, _, train_loader, _ = _build_dataloaders(config, split, all_sequences, normalizer)
    model = build_model(config).to(device)
    model = _apply_optimization_backend(model, config.model.optimization_backend)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    scaler = _cuda_grad_scaler(mixed_precision) if mixed_precision else None
    bench_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(config.experiment.output_root) / "benchmarks" / config.experiment.name / _timestamp()
    )
    bench_dir.mkdir(parents=True, exist_ok=True)

    def next_batch(iterator):
        try:
            return next(iterator), iterator
        except StopIteration:
            new_iterator = iter(train_loader)
            return next(new_iterator), new_iterator

    iterator = iter(train_loader)
    model.train()
    for _ in range(warmup_steps):
        batch, iterator = next_batch(iterator)
        _benchmark_step(model, optimizer, scaler, batch, device, mixed_precision, config.training.grad_clip)

    profiler = None
    if profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )

    rows: list[dict[str, float]] = []
    data_load_times: list[float] = []
    with profiler if profiler is not None else nullcontext() as active_profiler:
        for _ in range(steps):
            started = time.perf_counter()
            batch, iterator = next_batch(iterator)
            data_load_times.append(time.perf_counter() - started)
            row = _benchmark_step(model, optimizer, scaler, batch, device, mixed_precision, config.training.grad_clip)
            rows.append(row)
            if active_profiler is not None:
                active_profiler.step()

    if profiler is not None:
        profiler.export_chrome_trace(str(bench_dir / "torch_profiler_trace.json"))
        (bench_dir / "torch_profiler_key_averages.txt").write_text(
            profiler.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=40)
        )

    total_times = [
        data_load + row["transfer_s"] + row["forward_s"] + row["backward_optimizer_s"]
        for data_load, row in zip(data_load_times, rows, strict=True)
    ]
    summary = {
        "config": str(config_arg),
        "experiment": config.experiment.name,
        "model": config.experiment.model_name,
        "device": str(device),
        "mixed_precision": mixed_precision,
        "warmup_steps": warmup_steps,
        "steps": steps,
        "batch_size": config.training.batch_size,
        "num_workers": config.training.num_workers,
        "mean_data_load_s": _mean(data_load_times),
        "mean_transfer_s": _mean([row["transfer_s"] for row in rows]),
        "mean_forward_s": _mean([row["forward_s"] for row in rows]),
        "mean_backward_optimizer_s": _mean([row["backward_optimizer_s"] for row in rows]),
        "mean_step_s": _mean(total_times),
        "steps_per_second": 1.0 / _mean(total_times) if total_times else 0.0,
        "mean_loss": _mean([row["loss"] for row in rows]),
        "profile_trace": str(bench_dir / "torch_profiler_trace.json") if profile else None,
    }
    _write_json(bench_dir / "benchmark_summary.json", summary)
    _write_rows_csv(
        bench_dir / "benchmark_steps.csv",
        [
            {
                "step": idx,
                "data_load_s": data_load,
                **row,
                "step_s": data_load + row["transfer_s"] + row["forward_s"] + row["backward_optimizer_s"],
            }
            for idx, (data_load, row) in enumerate(zip(data_load_times, rows, strict=True), start=1)
        ],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return bench_dir.resolve()


def _load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[dict, RunConfig, Normalizer, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = RunConfig.from_dict(checkpoint["config"])
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return checkpoint, config, normalizer, model


def _dataset_for_split(
    config: RunConfig,
    checkpoint: dict,
    normalizer: Normalizer,
    split_name: str,
) -> SequenceWindowDataset:
    split = checkpoint["split"]
    if split_name not in split:
        raise ValueError(f"Unknown split {split_name!r}. Expected one of {sorted(split)}.")
    train_dir = Path(config.experiment.data_dir) / "train"
    all_sequences = load_sequences_from_dir_or_consolidated(train_dir)
    sequences_dict = {seq_id: all_sequences[seq_id] for seq_id in split[split_name]}
    return SequenceWindowDataset(
        sequence_paths=sequences_dict,
        normalizer=normalizer,
        context_size=config.experiment.context_size,
        horizon_size=config.experiment.horizon_size,
        use_velocity=config.features.use_velocity,
    )


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split_name: str,
    output_dir: str | Path | None = None,
    device_arg: str = "auto",
) -> Path:
    device = resolve_device(device_arg)
    print("Using device:", device)
    checkpoint, config, normalizer, model = _load_checkpoint(checkpoint_path, device)
    mixed_precision = _mixed_precision_enabled(device, config.training.mixed_precision)
    print("Mixed precision:", "enabled" if mixed_precision else "disabled")
    dataset = _dataset_for_split(config, checkpoint, normalizer, split_name)
    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
    )
    metrics, examples = evaluate_model(
        model=model,
        dataloader=dataloader,
        normalizer=normalizer,
        device=device,
        num_examples=config.evaluation.num_examples,
        mixed_precision=mixed_precision,
    )
    base_dir = Path(output_dir) if output_dir else Path(checkpoint_path).resolve().parent / f"eval_{split_name}"
    base_dir.mkdir(parents=True, exist_ok=True)
    _write_json(base_dir / "metrics.json", metrics)
    for label, records in examples.items():
        for rank, record in enumerate(records, start=1):
            save_prediction_plot(
                output_path=base_dir / f"{label}_{rank}_seq{record.sequence_id}_start{record.start_idx}.png",
                context_xy=record.context_xy,
                target_xy=record.target_xy,
                pred_xy=record.pred_xy,
                static=record.static,
                title=f"{label.title()} #{rank} | seq={record.sequence_id} start={record.start_idx} mse={record.error:.4f}",
            )
    return base_dir.resolve()


def _submission_columns(template_path: str | Path) -> list[str]:
    with Path(template_path).open(newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def submit_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    device_arg: str = "auto",
) -> Path:
    device = resolve_device(device_arg)
    print("Using device:", device)
    _, config, normalizer, model = _load_checkpoint(checkpoint_path, device)
    mixed_precision = _mixed_precision_enabled(device, config.training.mixed_precision)
    print("Mixed precision:", "enabled" if mixed_precision else "disabled")
    dataset = TestSequenceDataset(
        test_dir=Path(config.experiment.data_dir) / "test",
        normalizer=normalizer,
        use_velocity=config.features.use_velocity,
    )
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    rows: list[list[float]] = []
    with torch.no_grad():
        for batch in dataloader:
            context = batch["context"].to(device)
            with _cuda_autocast(mixed_precision):
                prediction = model(context)
            prediction_raw = _denormalize_predictions(prediction, normalizer).cpu()
            for idx in range(prediction_raw.shape[0]):
                seq_id = int(batch["sequence_id"][idx].item())
                rows.append([seq_id] + prediction_raw[idx].reshape(-1).tolist())
    rows.sort(key=lambda row: row[0])
    template_path = Path(config.experiment.data_dir) / "sample_submission.csv"
    columns = _submission_columns(template_path)
    output = Path(output_path) if output_path else Path(checkpoint_path).resolve().parent / "submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return output.resolve()


def train_many_experiments(
    configs: list[str],
    seeds: list[int] | None = None,
    overrides: dict[str, object] | None = None,
    evaluate_splits: list[str] | None = None,
    fail_fast: bool = False,
) -> Path:
    if not configs:
        raise ValueError("train-many requires at least one config.")
    seeds = seeds or [0]
    evaluate_splits = evaluate_splits or []
    output_root = str((overrides or {}).get("experiment.output_root", "runs"))
    batch_root = Path(output_root) / "batches" / _timestamp()
    batch_root.mkdir(parents=True, exist_ok=False)
    jobs: list[dict[str, object]] = []
    for config_arg in configs:
        for seed in seeds:
            job_overrides = dict(overrides or {})
            name_suffix = job_overrides.pop("experiment.name_suffix", None)
            job_overrides["experiment.seed"] = seed
            if name_suffix:
                job_overrides["experiment.name"] = f"{Path(config_arg).stem}_{name_suffix}_s{seed}"
            job: dict[str, object] = {
                "config": config_arg,
                "seed": seed,
                "overrides": job_overrides,
                "status": "started",
            }
            try:
                print(f"[train-many] starting {Path(config_arg).stem} seed={seed}")
                run_dir = train_experiment(config_arg, overrides=job_overrides)
                job["run_dir"] = str(run_dir)
                eval_dirs: dict[str, str] = {}
                checkpoint = Path(run_dir) / "best.pt"
                device_arg = str(job_overrides.get("experiment.device", "auto"))
                for split_name in evaluate_splits:
                    eval_dir = evaluate_checkpoint(
                        checkpoint_path=checkpoint,
                        split_name=split_name,
                        device_arg=device_arg,
                    )
                    eval_dirs[split_name] = str(eval_dir)
                if eval_dirs:
                    job["eval_dirs"] = eval_dirs
                job["status"] = "completed"
            except Exception as exc:
                job["status"] = "failed"
                job["error"] = f"{type(exc).__name__}: {exc}"
                job["traceback"] = traceback.format_exc()
                jobs.append(job)
                _write_json(batch_root / "batch_summary.json", {"jobs": jobs})
                if fail_fast:
                    raise
                continue
            jobs.append(job)
            _write_json(batch_root / "batch_summary.json", {"jobs": jobs})
    return batch_root.resolve()


def _required_eval_splits(evaluate_splits: list[str] | None = None) -> list[str]:
    ordered = ["val"]
    for split_name in evaluate_splits or []:
        if split_name not in ordered:
            ordered.append(split_name)
    return ordered


def repeat_trials_experiment(
    config_arg: str | Path,
    repeats: int,
    overrides: dict[str, object] | None = None,
    evaluate_splits: list[str] | None = None,
    fail_fast: bool = False,
    seed_step: int = 1,
) -> Path:
    if repeats < 1:
        raise ValueError("repeats must be at least 1.")

    config, config_path = _load_runtime_config(config_arg, overrides)
    base_name = config.experiment.name
    base_seed = config.experiment.seed
    repeat_root = Path(config.experiment.output_root) / "repeat_trials" / base_name / _timestamp()
    repeat_root.mkdir(parents=True, exist_ok=False)
    required_splits = _required_eval_splits(evaluate_splits)
    rows: list[dict[str, object]] = []

    for trial_idx in range(repeats):
        seed = base_seed + trial_idx * seed_step
        job_overrides = dict(overrides or {})
        job_overrides["experiment.seed"] = seed
        job_overrides["experiment.output_root"] = str(repeat_root / "runs")
        job_overrides["experiment.name"] = _candidate_name(base_name, f"trial{trial_idx:02d}_s{seed}")
        row: dict[str, object] = {
            "trial": trial_idx,
            "seed": seed,
            "status": "started",
        }
        try:
            print(f"[repeat-trials] starting {base_name} trial={trial_idx} seed={seed}")
            run_dir = train_experiment(config_arg, overrides=job_overrides)
            row["run_dir"] = str(run_dir)
            device_arg = str(job_overrides.get("experiment.device", config.experiment.device))
            split_metrics: dict[str, dict[str, float]] = {}
            eval_dirs: dict[str, str] = {}
            for split_name in required_splits:
                eval_dir, metrics = _evaluate_run_checkpoint(run_dir, split_name, device_arg)
                eval_dirs[split_name] = str(eval_dir)
                split_metrics[split_name] = metrics
                for metric_name, metric_value in metrics.items():
                    row[f"{split_name}_{metric_name}"] = metric_value
            row["eval_dirs"] = json.dumps(eval_dirs, sort_keys=True)
            row["status"] = "completed"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
            rows.append(row)
            _write_rows_csv(repeat_root / "trial_metrics.csv", rows)
            _write_json(repeat_root / "repeat_summary.json", {"trials": rows})
            if fail_fast:
                raise
            continue
        rows.append(row)
        _write_rows_csv(repeat_root / "trial_metrics.csv", rows)

    completed = [row for row in rows if row["status"] == "completed"]
    aggregate: dict[str, float] = {}
    for split_name in required_splits:
        split_rows = [
            {metric: float(row[f"{split_name}_{metric}"]) for metric in METRIC_KEYS if f"{split_name}_{metric}" in row} for row in completed
        ]
        aggregate.update(_aggregate_split_metrics(split_rows, split_name))
    ranking_key = f"val_{config.experiment.checkpoint_metric}_mean"
    summary = {
        "protocol": "repeat_trials",
        "config": str(config_path),
        "repeats": repeats,
        "seed_step": seed_step,
        "checkpoint_metric": config.experiment.checkpoint_metric,
        "ranking_metric": ranking_key,
        "best_score": aggregate.get(ranking_key),
        "aggregate": aggregate,
        "trials": rows,
    }
    _write_json(repeat_root / "repeat_summary.json", summary)
    return repeat_root.resolve()


def cross_validate_experiment(
    config_arg: str | Path,
    num_folds: int,
    overrides: dict[str, object] | None = None,
    evaluate_splits: list[str] | None = None,
    fail_fast: bool = False,
    include_holdout: bool = False,
) -> Path:
    config, config_path = _load_runtime_config(config_arg, overrides)
    base_name = config.experiment.name
    cv_root = Path(config.experiment.output_root) / "cross_validation" / base_name / _timestamp()
    cv_root.mkdir(parents=True, exist_ok=False)
    train_dir = Path(config.experiment.data_dir) / "train"
    fold_paths = freeze_kfold_sequence_splits(
        train_dir=train_dir,
        output_dir=cv_root / "splits",
        num_folds=num_folds,
        seed=config.experiment.seed,
        base_split_path=config.experiment.split_path,
        include_holdout=include_holdout,
    )
    required_splits = _required_eval_splits(evaluate_splits)
    rows: list[dict[str, object]] = []

    for fold_idx, fold_path in enumerate(fold_paths):
        job_overrides = dict(overrides or {})
        job_overrides["experiment.output_root"] = str(cv_root / "runs")
        job_overrides["experiment.name"] = _candidate_name(base_name, f"fold{fold_idx:02d}")
        job_overrides["experiment.split_path"] = str(fold_path)
        row: dict[str, object] = {
            "fold": fold_idx,
            "split_path": str(fold_path),
            "status": "started",
        }
        try:
            print(f"[cross-validate] starting {base_name} fold={fold_idx + 1}/{num_folds}")
            run_dir = train_experiment(config_arg, overrides=job_overrides)
            row["run_dir"] = str(run_dir)
            device_arg = str(job_overrides.get("experiment.device", config.experiment.device))
            eval_dirs: dict[str, str] = {}
            for split_name in required_splits:
                eval_dir, metrics = _evaluate_run_checkpoint(run_dir, split_name, device_arg)
                eval_dirs[split_name] = str(eval_dir)
                for metric_name, metric_value in metrics.items():
                    row[f"{split_name}_{metric_name}"] = metric_value
            row["eval_dirs"] = json.dumps(eval_dirs, sort_keys=True)
            row["status"] = "completed"
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc()
            rows.append(row)
            _write_rows_csv(cv_root / "fold_metrics.csv", rows)
            _write_json(cv_root / "cv_summary.json", {"folds": rows})
            if fail_fast:
                raise
            continue
        rows.append(row)
        _write_rows_csv(cv_root / "fold_metrics.csv", rows)

    completed = [row for row in rows if row["status"] == "completed"]
    aggregate: dict[str, float] = {}
    for split_name in required_splits:
        split_rows = [
            {metric: float(row[f"{split_name}_{metric}"]) for metric in METRIC_KEYS if f"{split_name}_{metric}" in row} for row in completed
        ]
        aggregate.update(_aggregate_split_metrics(split_rows, split_name))
    ranking_key = f"val_{config.experiment.checkpoint_metric}_mean"
    summary = {
        "protocol": "cross_validation",
        "config": str(config_path),
        "num_folds": num_folds,
        "include_holdout": include_holdout,
        "checkpoint_metric": config.experiment.checkpoint_metric,
        "ranking_metric": ranking_key,
        "best_score": aggregate.get(ranking_key),
        "aggregate": aggregate,
        "folds": rows,
    }
    _write_json(cv_root / "cv_summary.json", summary)
    return cv_root.resolve()
