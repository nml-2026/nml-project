from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import csv
import copy
import json
import os
import random
import shutil
import statistics
import sys
import traceback

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
    collect_sequence_paths,
    compute_normalizer,
    freeze_kfold_sequence_splits,
    freeze_sequence_split,
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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    if device_arg == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this environment.")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _prepare_split_and_normalizer(config: RunConfig) -> tuple[dict[str, list[int]], dict[int, Path], Normalizer]:
    train_dir = Path(config.experiment.data_dir) / "train"
    split = freeze_sequence_split(train_dir, config.experiment.split_path, seed=config.experiment.seed)
    all_paths = collect_sequence_paths(train_dir)
    normalizer = compute_normalizer(all_paths[seq_id] for seq_id in split["train"])
    return split, all_paths, normalizer


def _build_dataloaders(
    config: RunConfig,
    split: dict[str, list[int]],
    all_paths: dict[int, Path],
    normalizer: Normalizer,
) -> tuple[Dataset, Dataset, DataLoader, DataLoader]:
    def build_split_loader(
        sequence_ids: list[int],
        sampling_strategy: str,
        shuffle: bool,
    ) -> tuple[Dataset, DataLoader]:
        sequence_paths = [all_paths[seq_id] for seq_id in sequence_ids]
        if sampling_strategy == "all_windows":
            dataset = SequenceWindowDataset(
                sequence_paths=sequence_paths,
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
                sequence_paths=sequence_paths,
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
            f"Unknown window sampling strategy {sampling_strategy!r}. "
            "Expected one of {'all_windows', 'random_window_per_sequence'}."
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
            prediction = model(context)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Model produced NaN or Inf predictions during evaluation.")
            loss = normalized_mse_loss(prediction, target)
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
        context = batch["context"].to(device)
        target = batch["future_xy"].to(device)
        prediction = model(
            context,
            future_xy=target,
            teacher_forcing_prob=teacher_forcing_prob,
            scheduled_sampling_prob=scheduled_sampling_prob,
        )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Model produced NaN or Inf predictions during training.")
        loss = normalized_mse_loss(prediction, target)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
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
) -> Path:
    config, config_path = _load_runtime_config(config_arg, overrides)
    set_seed(config.experiment.seed)
    device = resolve_device(config.experiment.device)
    print("Using device:", device)
    split, all_paths, normalizer = _prepare_split_and_normalizer(config)
    train_dataset, val_dataset, train_loader, val_loader = _build_dataloaders(config, split, all_paths, normalizer)
    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    ema = ExponentialMovingAverage(model, config.training.ema_decay) if config.training.ema_decay > 0.0 else None
    swa_model = AveragedModel(model) if config.training.swa_start_pct > 0.0 else None
    swa_start_epoch = max(1, int(config.training.epochs * config.training.swa_start_pct)) if swa_model else 0
    run_dir = _run_dir(config)
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

    best_metric = float("inf")
    best_epoch = 0
    patience = 0
    history: list[dict[str, float]] = []
    epoch_iterator = _progress_bar(
        range(1, config.training.epochs + 1),
        total=config.training.epochs,
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
        )
        if swa_model is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
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
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_ade": val_metrics["ade"],
            "val_rmse": val_metrics["rmse"],
            "scheduled_sampling_prob": scheduled_sampling_prob,
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
        else:
            patience += 1
        if epoch_reporter is not None:
            report_payload = dict(row)
            report_payload["best_epoch"] = best_epoch
            report_payload[f"best_val_{config.experiment.checkpoint_metric}"] = best_metric
            report_payload["checkpoint_metric"] = current_metric
            report_payload["run_dir"] = str(run_dir)
            epoch_reporter(report_payload)
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
    }
    _write_json(run_dir / "train_summary.json", summary)
    return run_dir


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
    all_paths = collect_sequence_paths(Path(config.experiment.data_dir) / "train")
    return SequenceWindowDataset(
        sequence_paths=[all_paths[seq_id] for seq_id in split[split_name]],
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
            {metric: float(row[f"{split_name}_{metric}"]) for metric in METRIC_KEYS if f"{split_name}_{metric}" in row}
            for row in completed
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
            {metric: float(row[f"{split_name}_{metric}"]) for metric in METRIC_KEYS if f"{split_name}_{metric}" in row}
            for row in completed
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
