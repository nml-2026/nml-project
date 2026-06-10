from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data import TestSequenceDataset, _denormalize_predictions
from ..models.builders import mixture_point_estimate_for_inference
from .evaluator import _ensemble_predict_raw, _load_ensemble, _tta_predict
from .io import _cuda_autocast, _load_checkpoint, resolve_device


def _submission_columns(template_path: str | Path) -> list[str]:
    with Path(template_path).open("r", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def submit_checkpoint(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
    device_arg: str = "auto",
    tta_group: str | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    device = resolve_device(device_arg)
    print("Using device:", device)
    checkpoint, config, normalizer, model = _load_checkpoint(checkpoint_path, device)

    mixed_precision = config.runtime.mixed_precision
    print("Mixed precision:", "enabled" if mixed_precision else "disabled")
    if data_dir is not None:
        config.experiment.data_dir = str(Path(data_dir).resolve())
    resolved_tta_group = tta_group if tta_group is not None else config.evaluation.tta_group
    use_velocity = config.features.use_velocity
    dataset = TestSequenceDataset(
        test_dir=Path(config.experiment.data_dir) / "test",
        normalizer=normalizer,
        use_velocity=use_velocity,
    )
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    rows: list[list[float]] = []
    with torch.no_grad():
        for batch in dataloader:
            context = batch["context"].to(device)
            with _cuda_autocast(mixed_precision):
                prediction = _tta_predict(
                    model,
                    context,
                    normalizer,
                    resolved_tta_group,
                    use_velocity,
                    mixture_point_estimate_for_inference(config),
                )
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


def submit_ensemble(
    checkpoints: list[str | Path],
    output_path: str | Path | None = None,
    device_arg: str = "auto",
    tta_group: str = "xy",
    data_dir: str | Path | None = None,
) -> Path:
    device = resolve_device(device_arg)
    print("Using device:", device)
    models, config, _ = _load_ensemble(checkpoints, device)
    if data_dir is not None:
        config.experiment.data_dir = str(Path(data_dir).resolve())
    use_velocity = config.features.use_velocity
    dataset = TestSequenceDataset(
        test_dir=Path(config.experiment.data_dir) / "test",
        normalizer=models[0][1],
        use_velocity=use_velocity,
    )
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False, num_workers=0)
    rows: list[list[float]] = []
    with torch.no_grad():
        for batch in dataloader:
            context = batch["context"].to(device)
            prediction_raw = _ensemble_predict_raw(context, models, tta_group, use_velocity).cpu()
            for idx in range(prediction_raw.shape[0]):
                seq_id = int(batch["sequence_id"][idx].item())
                rows.append([seq_id] + prediction_raw[idx].reshape(-1).tolist())
    rows.sort(key=lambda row: row[0])
    template_path = Path(config.experiment.data_dir) / "sample_submission.csv"
    columns = _submission_columns(template_path)
    output = Path(output_path) if output_path else Path(checkpoints[0]).resolve().parent / "ensemble_submission.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    return output.resolve()
