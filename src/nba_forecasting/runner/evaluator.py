from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import RunConfig
from ..data import (
    CourtReflectionTransform,
    Normalizer,
    _denormalize_predictions,
    build_eval_dataloader,
    load_sequences_from_dir_or_consolidated,
)
from ..metrics import (
    compute_raw_metrics,
    min_ade_k,
    min_fde_k,
    mixture_nll,
    normalized_mse_loss,
    per_example_mse,
    per_step_l2_error,
)
from ..models.builders import has_mixture_distribution, mixture_point_estimate_for_inference
from ..models.common import forecast_model, unwrap_model
from ..plots import save_per_step_error_plot, save_prediction_plot
from .io import (
    METRIC_KEYS,
    ExampleRecord,
    _cuda_autocast,
    _load_checkpoint,
    _mark_compiled_cuda_step_begin,
    _mixed_precision_enabled,
    _progress_bar,
    _progress_disabled,
    _read_json,
    _write_json,
    resolve_device,
    tqdm,
)


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
    tta_group: str = "none",
    use_velocity: bool = True,
    distribution_point_estimate: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[ExampleRecord]]]:
    model.eval()

    totals = {key: 0.0 for key in METRIC_KEYS}
    sample_totals = {"min_ade_k": 0.0, "min_fde_k": 0.0}
    saw_sample_modes = False
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
                _mark_compiled_cuda_step_begin(model, device)
                if tta_group == "none":
                    output = forecast_model(
                        model,
                        context,
                        future_xy=target,
                        mdn_point_estimate=distribution_point_estimate,
                    )
                    prediction = output.point
                    sample_modes = output.modes
                else:
                    prediction = _tta_predict(model, context, normalizer, tta_group, use_velocity, distribution_point_estimate)
                    sample_modes = None
                loss = normalized_mse_loss(prediction, target)
            if not torch.isfinite(prediction).all():
                finite = torch.isfinite(prediction)
                finite_values = prediction[finite]
                max_abs = float(finite_values.abs().max().item()) if finite_values.numel() else float("nan")
                raise FloatingPointError(
                    "Model produced NaN or Inf predictions during evaluation "
                    f"(batch_idx={batch_idx}, finite={int(finite.sum().item())}/{prediction.numel()}, max_abs_finite={max_abs:.6g})."
                )
            prediction_raw = _denormalize_predictions(prediction, normalizer)
            target_raw = batch["future_xy_raw"].to(device)
            metrics = compute_raw_metrics(prediction_raw, target_raw)
            totals["loss"] += float(loss.item())
            for key, value in metrics.items():
                totals[key] += value
            if sample_modes is not None:
                saw_sample_modes = True
                modes_raw = _denormalize_predictions(sample_modes, normalizer)
                sample_totals["min_ade_k"] += min_ade_k(modes_raw, target_raw)
                sample_totals["min_fde_k"] += min_fde_k(modes_raw, target_raw)

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
    if saw_sample_modes:
        averaged.update({key: value / max(batches, 1) for key, value in sample_totals.items()})
    return averaged, {"best": best_examples, "worst": worst_examples}


def _dataset_for_split(
    config: RunConfig,
    checkpoint: dict,
    normalizer: Normalizer,
    split_name: str,
) -> Dataset:
    from ..data import SequenceWindowDataset

    split = checkpoint["split"]
    if split_name not in split:
        raise ValueError(f"Split {split_name!r} not found in checkpoint.")
    train_dir = Path(config.experiment.data_dir) / "train"
    all_sequences = load_sequences_from_dir_or_consolidated(train_dir)
    sequence_data = {seq_id: all_sequences[seq_id] for seq_id in split[split_name]}
    return SequenceWindowDataset(
        sequence_paths=sequence_data,
        normalizer=normalizer,
        context_size=config.experiment.context_size,
        horizon_size=config.experiment.horizon_size,
        use_velocity=config.features.use_velocity,
    )


def _compute_per_step_error(
    model: torch.nn.Module,
    dataloader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
    tta_group: str,
    use_velocity: bool,
    distribution_point_estimate: str | None = None,
) -> list[float]:
    total: torch.Tensor | None = None
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            context = batch["context"].to(device)
            prediction = _tta_predict(model, context, normalizer, tta_group, use_velocity, distribution_point_estimate)
            prediction_raw = _denormalize_predictions(prediction, normalizer)
            target_raw = batch["future_xy_raw"].to(device)
            step_error = per_step_l2_error(prediction_raw, target_raw)
            total = step_error if total is None else total + step_error
            batches += 1
    if total is None:
        return []
    return (total / batches).cpu().tolist()


def _compute_mixture_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    normalizer: Normalizer,
    device: torch.device,
) -> dict[str, float]:
    nll_total = 0.0
    ade_total = 0.0
    fde_total = 0.0
    examples = 0
    with torch.no_grad():
        for batch in dataloader:
            context = batch["context"].to(device)
            target = batch["future_xy"].to(device)
            target_raw = batch["future_xy_raw"].to(device)
            output = forecast_model(model, context, future_xy=target, mdn_point_estimate="mean")
            if output.distribution is None:
                continue
            params = output.distribution
            nll = mixture_nll(params, target)
            nll_total += float(nll.item()) * context.shape[0]
            means_raw = _denormalize_predictions(params.means, normalizer)
            min_ade = min_ade_k(means_raw, target_raw)
            min_fde = min_fde_k(means_raw, target_raw)
            ade_total += float(min_ade) * context.shape[0]
            fde_total += float(min_fde) * context.shape[0]
            examples += context.shape[0]
    return {
        "mixture_nll": nll_total / max(examples, 1),
        "min_ade_k": ade_total / max(examples, 1),
        "min_fde_k": fde_total / max(examples, 1),
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    split_name: str,
    output_dir: str | Path | None = None,
    device_arg: str = "auto",
    tta_group: str | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    device = resolve_device(device_arg)
    print("Using device:", device)
    checkpoint, config, normalizer, model = _load_checkpoint(checkpoint_path, device)
    if data_dir is not None:
        config.experiment.data_dir = str(Path(data_dir).resolve())
    resolved_tta_group = tta_group if tta_group is not None else config.evaluation.tta_group
    dataset = _dataset_for_split(config, checkpoint, normalizer, split_name)
    dataloader = build_eval_dataloader(config, dataset)
    mixed_precision = _mixed_precision_enabled(device, config.runtime.mixed_precision)
    distribution_point_estimate = mixture_point_estimate_for_inference(config)
    metrics, examples = evaluate_model(
        model=model,
        dataloader=dataloader,
        normalizer=normalizer,
        device=device,
        num_examples=config.evaluation.num_examples,
        mixed_precision=mixed_precision,
        tta_group=resolved_tta_group,
        use_velocity=config.features.use_velocity,
        distribution_point_estimate=distribution_point_estimate,
    )
    metrics["tta_group"] = resolved_tta_group
    metrics["eval_model_kind"] = checkpoint.get("eval_model_kind", "raw")
    if has_mixture_distribution(config):
        mixture = _compute_mixture_metrics(model, dataloader, normalizer, device)
        metrics.update(mixture)
        inner = unwrap_model(model)
        if hasattr(inner, "mdn_components"):
            metrics["mdn_components"] = float(inner.mdn_components)

    suffix = f"_tta_{resolved_tta_group}" if resolved_tta_group != "none" else ""
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path(checkpoint_path).resolve().parent / f"eval_{split_name}{suffix}"
    base_dir.mkdir(parents=True, exist_ok=True)
    _write_json(base_dir / "metrics.json", metrics)

    for rank, key in (("best", "best"), ("worst", "worst")):
        for idx, record in enumerate(examples[key], start=1):
            save_prediction_plot(
                output_path=base_dir / f"{key}_{idx}.png",
                context_xy=record.context_xy,
                target_xy=record.target_xy,
                pred_xy=record.pred_xy,
                static=record.static,
                title=f"Sequence {record.sequence_id} (idx {record.start_idx}, rank {rank} {idx}, L2={record.error:.2f})",
            )
    per_step = _compute_per_step_error(
        model, dataloader, normalizer, device, resolved_tta_group, config.features.use_velocity, distribution_point_estimate
    )
    _write_json(base_dir / "per_step_error.json", {"ade_l2_per_step": per_step})
    save_per_step_error_plot(per_step, base_dir / "per_step_error.png")
    return base_dir.resolve()


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


def _normalize_predictions(prediction_raw: torch.Tensor, normalizer: Normalizer) -> torch.Tensor:
    mu = normalizer.mu.to(prediction_raw.device).view(1, 1, 1, 2)
    sigma = normalizer.sigma.to(prediction_raw.device).view(1, 1, 1, 2)
    return (prediction_raw - mu) / sigma


def _load_ensemble(
    checkpoint_paths: list[str | Path],
    device: torch.device,
) -> tuple[list[tuple[torch.nn.Module, Normalizer]], RunConfig, dict]:
    paths = list(checkpoint_paths)
    if not paths:
        raise ValueError("Ensemble requires at least one checkpoint path.")
    models: list[tuple[torch.nn.Module, Normalizer]] = []
    base_config: RunConfig | None = None
    base_checkpoint: dict | None = None
    base_shape: tuple[int, int] | None = None
    for path in paths:
        checkpoint, config, normalizer, model = _load_checkpoint(path, device)
        shape = (config.experiment.horizon_size, config.experiment.num_entities)
        if base_config is None:
            base_config, base_checkpoint, base_shape = config, checkpoint, shape
        elif shape != base_shape:
            raise ValueError(
                f"Ensemble member {str(path)!r} has (horizon, entities)={shape}, incompatible with the first member's {base_shape}."
            )
        models.append((model, normalizer))
    assert base_config is not None and base_checkpoint is not None
    return models, base_config, base_checkpoint


def _ensemble_predict_raw(
    context: torch.Tensor,
    models_with_normalizers: list[tuple[torch.nn.Module, Normalizer]],
    tta_group: str,
    use_velocity: bool,
) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    for model, normalizer in models_with_normalizers:
        prediction = _tta_predict(model, context, normalizer, tta_group, use_velocity)
        predictions.append(_denormalize_predictions(prediction, normalizer))
    averaged = torch.stack(predictions, dim=0).mean(dim=0)
    if not torch.isfinite(averaged).all():
        raise FloatingPointError("Ensemble produced NaN or Inf predictions.")
    return averaged


def _compute_ensemble_per_step_error(
    models_with_normalizers: list[tuple[torch.nn.Module, Normalizer]],
    dataloader: DataLoader,
    tta_group: str,
    use_velocity: bool,
    device: torch.device,
) -> list[float]:
    total: torch.Tensor | None = None
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            context = batch["context"].to(device)
            prediction_raw = _ensemble_predict_raw(context, models_with_normalizers, tta_group, use_velocity)
            target_raw = batch["future_xy_raw"].to(device)
            step_error = per_step_l2_error(prediction_raw, target_raw)
            total = step_error if total is None else total + step_error
            batches += 1
    if total is None:
        return []
    return (total / batches).cpu().tolist()


def evaluate_ensemble(
    checkpoints: list[str | Path],
    split_name: str,
    output_dir: str | Path | None = None,
    device_arg: str = "auto",
    tta_group: str = "xy",
    data_dir: str | Path | None = None,
) -> Path:
    device = resolve_device(device_arg)
    print("Using device:", device)
    models, config, checkpoint = _load_ensemble(checkpoints, device)
    if data_dir is not None:
        config.experiment.data_dir = str(Path(data_dir).resolve())
    use_velocity = config.features.use_velocity
    normalizer = models[0][1]
    dataset = _dataset_for_split(config, checkpoint, normalizer, split_name)
    dataloader = build_eval_dataloader(config, dataset)
    totals = {key: 0.0 for key in METRIC_KEYS}
    batches = 0
    with torch.no_grad():
        for batch in dataloader:
            context = batch["context"].to(device)
            prediction_raw = _ensemble_predict_raw(context, models, tta_group, use_velocity)
            target = batch["future_xy"].to(device)
            loss = normalized_mse_loss(_normalize_predictions(prediction_raw, normalizer), target)
            target_raw = batch["future_xy_raw"].to(device)
            metrics = compute_raw_metrics(prediction_raw, target_raw)
            totals["loss"] += float(loss.item())
            for key, value in metrics.items():
                totals[key] += value
            batches += 1
    averaged: dict[str, object] = {key: value / max(batches, 1) for key, value in totals.items()}
    averaged["tta_group"] = tta_group
    averaged["ensemble_n"] = len(models)
    averaged["checkpoints"] = [str(Path(p).resolve()) for p in checkpoints]
    suffix = f"_tta_{tta_group}" if tta_group != "none" else ""
    if output_dir:
        base_dir = Path(output_dir)
    else:
        base_dir = Path(checkpoints[0]).resolve().parent / f"eval_ensemble_{split_name}_n{len(models)}{suffix}"
    base_dir.mkdir(parents=True, exist_ok=True)
    _write_json(base_dir / "metrics.json", averaged)
    per_step_error = _compute_ensemble_per_step_error(models, dataloader, tta_group, use_velocity, device)
    _write_json(base_dir / "per_step_error.json", {"ade_l2_per_step": per_step_error})
    save_per_step_error_plot(per_step_error, base_dir / "per_step_error.png")
    return base_dir.resolve()


TTA_GROUPS: dict[str, list[tuple[bool, bool]]] = {
    "none": [(False, False)],
    "y": [(False, False), (False, True)],
    "x": [(False, False), (True, False)],
    "xy": [(False, False), (True, False), (False, True), (True, True)],
}


def _tta_predict(
    model: torch.nn.Module,
    context: torch.Tensor,
    normalizer: Normalizer,
    tta_group: str,
    use_velocity: bool,
    distribution_point_estimate: str | None = None,
) -> torch.Tensor:
    if tta_group not in TTA_GROUPS:
        raise ValueError(f"Unknown TTA group {tta_group!r}.")
    elements = TTA_GROUPS[tta_group]
    if elements == [(False, False)]:
        return forecast_model(model, context, mdn_point_estimate=distribution_point_estimate).point
    transform = CourtReflectionTransform.from_normalization(
        normalizer,
        use_velocity=use_velocity,
    )
    predictions: list[torch.Tensor] = []
    for flip_x, flip_y in elements:
        if flip_x or flip_y:
            reflected = transform.reflect_context(context, flip_x=flip_x, flip_y=flip_y)
            point = forecast_model(model, reflected, mdn_point_estimate=distribution_point_estimate).point
            output = transform.inverse_prediction(point, flip_x=flip_x, flip_y=flip_y)
        else:
            output = forecast_model(model, context, mdn_point_estimate=distribution_point_estimate).point
        predictions.append(output)
    return torch.stack(predictions, dim=0).mean(dim=0)
