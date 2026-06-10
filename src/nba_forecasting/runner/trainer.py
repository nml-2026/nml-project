from __future__ import annotations

import json
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from torch.optim.swa_utils import SWALR, AveragedModel

from ..config import RunConfig
from ..data import (
    Normalizer,
    build_dataloaders,
    freeze_kfold_sequence_splits,
    prepare_split_and_normalizer,
)
from ..models import build_model
from ..plots import save_history_plot
from .evaluator import _aggregate_split_metrics, _evaluate_run_checkpoint, evaluate_checkpoint, evaluate_model
from .io import (
    METRIC_KEYS,
    ExponentialMovingAverage,
    _assert_resume_compatible,
    _checkpoint_model_state_dict,
    _checkpoint_run_dir,
    _cuda_grad_scaler,
    _load_runtime_config,
    _mixed_precision_enabled,
    _model_parameter_l2,
    _optional_batch_limit,
    _progress_bar,
    _progress_disabled,
    _run_dir,
    _scheduled_sampling_probability,
    _timestamp,
    _write_history_csv,
    _write_json,
    _write_rows_csv,
    log_epoch_progress,
    resolve_device,
    set_seed,
    tqdm,
)
from .recipe import ForecastTrainingRecipe


@runtime_checkable
class SetEpochSampler(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


def _candidate_name(base_name: str, label: str) -> str:
    safe_label = label.replace("/", "_").replace(" ", "_")
    return f"{base_name}_{safe_label}"


def _training_checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any | None,
    swa_scheduler: Any | None,
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
        "optimizer_backend": config.runtime.optimizer_backend,
        "run_dir": str(run_dir),
    }
    if lr_scheduler is not None:
        payload["lr_scheduler_state"] = lr_scheduler.state_dict()
    if swa_scheduler is not None:
        payload["swa_scheduler_state"] = swa_scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state"] = scaler.state_dict()
    if ema is not None:
        payload["ema_model_state"] = ema.model.state_dict()
        payload["ema_decay"] = ema.decay
    if swa_model is not None:
        payload["swa_model_state"] = swa_model.state_dict()
    return payload


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
    mixed_precision = _mixed_precision_enabled(device, config.runtime.mixed_precision)
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
    split, all_sequences, normalizer = prepare_split_and_normalizer(config)
    if timing_callback is not None:
        timing_callback("dataset_loading_done")
        timing_callback("dataloader_construction_start")
    from ..runtime_env import configure_training_cpu_runtime, format_training_cpu_runtime

    cpu_runtime = configure_training_cpu_runtime(config.training.num_workers)
    print(format_training_cpu_runtime(cpu_runtime))
    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        config,
        split,
        all_sequences,
        normalizer,
        num_workers=cpu_runtime.num_workers,
    )
    if timing_callback is not None:
        timing_callback("dataloader_construction_done")
        timing_callback("model_construction_start")
    model = build_model(config)
    if timing_callback is not None:
        timing_callback("model_construction_done")
        timing_callback("first_cuda_operation_start")
    model = model.to(device)
    prepare_static_model_caches(model, config, device)
    model = _apply_optimization_backend(model, config.runtime.optimization_backend, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if timing_callback is not None:
        timing_callback("first_cuda_operation_done")
    optimizer = build_optimizer(config, model, device)
    lr_scheduler = _build_lr_scheduler(config, optimizer)
    ema = ExponentialMovingAverage(model, config.training.ema_decay) if config.training.ema_decay > 0.0 else None
    swa_model = AveragedModel(model) if config.training.swa_start_pct > 0.0 else None
    swa_start_epoch = max(1, int(config.training.epochs * config.training.swa_start_pct)) if swa_model else 0
    swa_scheduler = (
        SWALR(optimizer, swa_lr=config.training.swa_lr) if swa_model is not None and config.training.swa_lr is not None else None
    )
    scaler = _cuda_grad_scaler(mixed_precision) if mixed_precision else None
    from ..models.builders import resolve_training_loss_policy

    training_recipe = ForecastTrainingRecipe(
        loss_config=config.loss,
        normalizer=normalizer,
        device=device,
        grad_clip=config.training.grad_clip,
        mixed_precision=mixed_precision,
        training_policy=resolve_training_loss_policy(config),
    )
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
        if swa_scheduler is not None and "swa_scheduler_state" in resume_state:
            swa_scheduler.load_state_dict(resume_state["swa_scheduler_state"])
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
    epoch_loop_started_at = time.monotonic()
    for epoch in epoch_iterator:
        train_sampler = train_loader.sampler
        if isinstance(train_sampler, SetEpochSampler):
            train_sampler.set_epoch(epoch - 1)
        val_sampler = val_loader.sampler
        if isinstance(val_sampler, SetEpochSampler):
            val_sampler.set_epoch(epoch - 1)
        scheduled_sampling_prob = _scheduled_sampling_probability(config, epoch)
        train_metrics = training_recipe.train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            teacher_forcing_prob=config.training.teacher_forcing_prob,
            scheduled_sampling_prob=scheduled_sampling_prob,
            ema=ema,
            max_batches=_optional_batch_limit(config.training.max_train_batches),
            scaler=scaler,
            timing_callback=timing_callback if epoch == 1 else None,
            epoch=epoch,
        )
        train_loss = train_metrics["loss"]
        epoch_lr = optimizer.param_groups[0]["lr"]
        in_swa_tail = swa_model is not None and epoch >= swa_start_epoch
        if in_swa_tail:
            swa_model.update_parameters(model)

        if in_swa_tail and swa_scheduler is not None:
            swa_scheduler.step()
        elif lr_scheduler is not None:
            lr_scheduler.step()

        epoch_param_l2 = _model_parameter_l2(model)

        eval_model, eval_model_kind = _validation_model(
            model=model,
            ema=ema,
            swa_model=swa_model,
            swa_active=in_swa_tail,
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
            "train_mse": train_metrics["mse"],
            "train_smooth": train_metrics["smooth"],
            "train_boundary": train_metrics["boundary"],
            "train_ball": train_metrics["ball"],
            "train_sample": train_metrics["sample"],
            "train_point": train_metrics["point"],
            "train_nll": train_metrics["nll"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_ade": val_metrics["ade"],
            "val_rmse": val_metrics["rmse"],
            "val_ade_l2": val_metrics["ade_l2"],
            "val_fde_l2": val_metrics["fde_l2"],
            "scheduled_sampling_prob": scheduled_sampling_prob,
            "lr": epoch_lr,
            "model_param_l2": epoch_param_l2,
            "model_param_l2_delta": epoch_param_l2 - initial_param_l2,
        }
        if "min_ade_k" in val_metrics:
            row["val_min_ade_k"] = val_metrics["min_ade_k"]
            row["val_min_fde_k"] = val_metrics["min_fde_k"]
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
                    "model_state": _checkpoint_model_state_dict(eval_model),
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
        log_epoch_progress(
            experiment_name=config.experiment.name,
            epoch=epoch,
            total_epochs=config.training.epochs,
            start_epoch=start_epoch,
            started_at=epoch_loop_started_at,
            train_loss=train_loss,
            val_metric=float(current_metric),
            best_epoch=best_epoch,
            best_metric=float(best_metric),
            checkpoint_metric=config.experiment.checkpoint_metric,
            patience=patience,
            patience_limit=config.training.patience,
        )
        if epoch_reporter is not None:
            report_payload = dict(row)
            report_payload["best_epoch"] = best_epoch
            report_payload[f"best_val_{config.experiment.checkpoint_metric}"] = best_metric
            report_payload["checkpoint_metric"] = current_metric
            report_payload["run_dir"] = str(run_dir)
            epoch_reporter(report_payload)
        torch.save(
            _training_checkpoint_payload(
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                swa_scheduler=swa_scheduler,
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
        "lr_scheduler": config.training.lr_scheduler,
        "swa_lr": config.training.swa_lr,
        "mixed_precision": mixed_precision,
        "optimizer_backend": config.runtime.optimizer_backend,
    }
    _write_json(run_dir / "train_summary.json", summary)
    return run_dir


def train_many_experiments(
    configs: list[str],
    seeds: list[int] | None = None,
    overrides: dict[str, object] | None = None,
    evaluate_splits: list[str] | None = None,
    fail_fast: bool = False,
) -> Path:
    if not configs:
        raise ValueError("batch training requires at least one config.")
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
                print(f"[train] starting {Path(config_arg).stem} seed={seed}")
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

    completed: list[dict[str, Any]] = [row for row in rows if row["status"] == "completed"]
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

    completed: list[dict[str, Any]] = [row for row in rows if row["status"] == "completed"]
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


def _build_lr_scheduler(
    config: RunConfig,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    kind = config.training.lr_scheduler
    if kind == "none":
        return None
    if kind == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.training.lr_milestones,
            gamma=config.training.lr_gamma,
        )
    epochs = max(1, config.training.epochs)
    eta_min = config.training.lr_min
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=eta_min)
    if kind == "cosine_warmup":
        warmup = min(max(1, config.training.lr_warmup_epochs), epochs - 1) if epochs > 1 else 1
        warmup_sched = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup), eta_min=eta_min)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])
    raise ValueError(f"Unknown training.lr_scheduler {kind!r}. Expected one of: none, multistep, cosine, cosine_warmup.")


def _configure_torch_dynamo_for_compile() -> None:
    import torch._dynamo

    if torch._dynamo.config.cache_size_limit < 64:
        torch._dynamo.config.cache_size_limit = 64
    accumulated = getattr(torch._dynamo.config, "accumulated_cache_size_limit", None)
    if isinstance(accumulated, int) and accumulated < 256:
        torch._dynamo.config.accumulated_cache_size_limit = 256


def _apply_optimization_backend(model: torch.nn.Module, backend: str, device: torch.device | None = None) -> torch.nn.Module:
    if backend == "none":
        return model
    if device is not None and device.type != "cuda":
        return model
    if backend == "compile":
        _configure_torch_dynamo_for_compile()
        compiled = torch.compile(model, mode="reduce-overhead")
        assert isinstance(compiled, torch.nn.Module)
        return compiled
    if backend == "compile_default":
        _configure_torch_dynamo_for_compile()
        compiled = torch.compile(model, mode="default")
        assert isinstance(compiled, torch.nn.Module)
        return compiled
    raise ValueError("optimization_backend must be one of: none, compile, compile_default.")


def prepare_static_model_caches(model: torch.nn.Module, config: RunConfig, device: torch.device) -> None:
    del model, config, device


def build_optimizer(config: RunConfig, model: torch.nn.Module, device: torch.device) -> torch.optim.Optimizer:
    backend = config.runtime.optimizer_backend
    if backend == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
    if backend == "adamw_fused":
        if device.type != "cuda":
            return torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
        try:
            return torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay, fused=True)
        except TypeError as exc:
            raise RuntimeError("optimizer_backend='adamw_fused' is not supported by this PyTorch build.") from exc
    raise ValueError("optimizer_backend must be one of: adamw, adamw_fused.")


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
