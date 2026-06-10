from __future__ import annotations

import csv
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from .config import load_config, parse_override_value, resolve_config_paths
from .models.builders import collect_search_profiles

SEARCH_PATTERN = re.compile(r"^(?P<kind>[a-z_]+)\((?P<body>.*)\)$")
FAILED_TRIAL_METRIC = 1e9
RUNTIME_SEARCH_PROFILES: dict[str, list[str]] = {
    "aux_losses": [
        "loss.lambda_smooth=loguniform(1e-4,2e-1)",
        "loss.lambda_boundary=loguniform(1e-4,2e-1)",
        "loss.lambda_ball=loguniform(1e-4,2e-1)",
    ],
    "lr_scheduler": [
        "training.lr=loguniform(1e-4,1e-3)",
        "training.swa_lr=loguniform(5e-5,3.5e-4)",
    ],
}
SEARCH_PROFILES: dict[str, list[str]] = {**collect_search_profiles(), **RUNTIME_SEARCH_PROFILES}


def resolve_search_specs(profile: str | None, search_specs: list[str] | None = None) -> list[str]:
    profile_specs: list[str] = []
    if profile and profile != "none":
        if profile not in SEARCH_PROFILES:
            available = ", ".join(["none", *sorted(SEARCH_PROFILES)])
            raise ValueError(f"Unknown search profile {profile!r}. Available profiles: {available}.")
        profile_specs = SEARCH_PROFILES[profile]
    resolved = [*profile_specs, *(search_specs or [])]
    if not resolved:
        raise ValueError("At least one --search specification or --search-profile is required.")
    return resolved


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _require_ray():
    try:
        import ray
        from ray import train as ray_train
        from ray import tune
        from ray.tune import Checkpoint
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search.optuna import OptunaSearch
    except ImportError as exc:  # pragma: no cover - exercised only when Ray is missing
        raise RuntimeError(
            "Ray Tune is required for hyperparameter tuning. Install it with "
            '`pip install "ray[tune]>=2.52.1" "optuna>=3.6"` or sync the project environment first.'
        ) from exc
    return ray, ray_train, tune, Checkpoint, ASHAScheduler, OptunaSearch


def _parse_list(raw_body: str) -> list[object]:
    values = [parse_override_value(chunk.strip()) for chunk in raw_body.split(",") if chunk.strip()]
    if not values:
        raise ValueError("Search spaces must contain at least one value.")
    return values


def _parse_search_expression(raw_expression: str, tune):
    expression = raw_expression.strip()
    match = SEARCH_PATTERN.match(expression)
    if match is None:
        return tune.choice([parse_override_value(expression)])

    kind = match.group("kind")
    body = match.group("body")
    if kind == "choice":
        return tune.choice(_parse_list(body))
    if kind == "grid":
        return tune.grid_search(_parse_list(body))
    if kind == "uniform":
        low, high = _parse_list(body)
        assert isinstance(low, (int, float, str))
        assert isinstance(high, (int, float, str))
        return tune.uniform(float(low), float(high))
    if kind == "loguniform":
        low, high = _parse_list(body)
        assert isinstance(low, (int, float, str))
        assert isinstance(high, (int, float, str))
        return tune.loguniform(float(low), float(high))
    if kind == "randint":
        low, high = _parse_list(body)
        assert isinstance(low, (int, float, str))
        assert isinstance(high, (int, float, str))
        return tune.randint(int(low), int(high))
    raise ValueError(
        f"Unsupported Ray Tune search expression {raw_expression!r}. "
        "Use choice(...), grid(...), uniform(...), loguniform(...), or randint(...)."
    )


def _build_search_space(search_specs: list[str], tune) -> tuple[dict[str, object], set[str]]:
    if not search_specs:
        raise ValueError("At least one --search specification is required.")
    space: dict[str, object] = {}
    kinds: set[str] = set()
    for raw_spec in search_specs:
        if "=" not in raw_spec:
            raise ValueError(f"Invalid search spec {raw_spec!r}. Expected PATH=EXPR.")
        path, raw_expression = raw_spec.split("=", 1)
        space[path] = _parse_search_expression(raw_expression, tune)
        match = SEARCH_PATTERN.match(raw_expression.strip())
        kinds.add(match.group("kind") if match is not None else "literal")
    return space, kinds


class TrainableTimingTrace:
    def __init__(self, root: str | Path, trial_name: str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", trial_name).strip("_") or "trial"
        self.path = self.root / f"{safe_name}_{os.getpid()}.jsonl"
        self.started_perf = time.perf_counter()
        self.first_tune_report_seen = False

    def mark(self, event: str, **extra: object) -> None:
        payload = {
            "event": event,
            "elapsed_s": time.perf_counter() - self.started_perf,
            "timestamp": datetime.now().isoformat(),
            **extra,
        }
        with self.path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


class TrialPrunedByMetric(RuntimeError):
    def __init__(self, report: dict[str, object], *, metric_name: str, threshold: float) -> None:
        super().__init__(f"{metric_name} exceeded prune threshold {threshold}.")
        self.report = report
        self.metric_name = metric_name
        self.threshold = threshold


def _runtime_context(ray_module) -> dict[str, object]:
    try:
        gpu_ids = ray_module.get_gpu_ids()
    except Exception as exc:  # pragma: no cover - depends on Ray runtime internals
        gpu_ids = f"unavailable: {exc}"
    try:
        import torch

        torch_context: dict[str, object] = {
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "torch_cuda_is_available": torch.cuda.is_available(),
            "torch_cuda_device_count": torch.cuda.device_count(),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        torch_context = {"torch_context_error": str(exc)}
    return {
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "virtual_env": os.environ.get("VIRTUAL_ENV", ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "ray_gpu_ids": gpu_ids,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", ""),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", ""),
        **torch_context,
    }


def tune_metric_name(checkpoint_metric: str) -> str:
    """Ray Tune objective: best validation score seen so far (matches best.pt selection)."""
    return f"best_val_{checkpoint_metric}"


def _read_history_rows(run_dir: str | Path) -> list[dict[str, str]]:
    history_path = Path(run_dir) / "history.csv"
    if not history_path.exists():
        return []
    with history_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _last_history_row(run_dir: str | Path) -> dict[str, object]:
    rows = _read_history_rows(run_dir)
    if not rows:
        return {}
    parsed: dict[str, object] = {}
    for key, value in rows[-1].items():
        if value is None:
            continue
        try:
            parsed[f"train_history_{key}"] = float(value)
        except ValueError:
            parsed[f"train_history_{key}"] = value
    return parsed


def _last_epoch_val_metric(run_dir: str | Path, checkpoint_metric: str) -> float | None:
    rows = _read_history_rows(run_dir)
    if not rows:
        return None
    key = f"val_{checkpoint_metric}"
    raw_value = rows[-1].get(key)
    if raw_value is None:
        return None
    try:
        return float(raw_value)
    except ValueError:
        return None


def _build_final_tune_report(
    *,
    run_dir: str | Path,
    eval_dir: str | Path,
    checkpoint_metric: str,
    metric_name: str,
    trial_config_arg: str,
    export_overrides: dict[str, object],
) -> dict[str, object]:
    final_metrics = json.loads((Path(eval_dir) / "metrics.json").read_text())
    train_summary_path = Path(run_dir) / "train_summary.json"
    train_summary = json.loads(train_summary_path.read_text()) if train_summary_path.exists() else {}
    best_checkpoint_score = final_metrics.get(checkpoint_metric)
    if not isinstance(best_checkpoint_score, int | float):
        raise ValueError(f"Missing numeric {checkpoint_metric!r} in {eval_dir}/metrics.json.")

    report: dict[str, object] = {f"val_{key}": value for key, value in final_metrics.items()}
    report[metric_name] = float(best_checkpoint_score)
    report[f"val_{checkpoint_metric}"] = float(best_checkpoint_score)
    report["best_epoch"] = train_summary.get("best_epoch")
    report["checkpoint_metric"] = checkpoint_metric
    last_epoch_score = _last_epoch_val_metric(run_dir, checkpoint_metric)
    if last_epoch_score is not None:
        report[f"val_{checkpoint_metric}_last_epoch"] = last_epoch_score
    report.update(_last_history_row(run_dir))
    report["config_arg"] = trial_config_arg
    report["applied_overrides"] = json.dumps(export_overrides, sort_keys=True, default=str)
    report["run_dir"] = str(run_dir)
    report["eval_dir"] = str(eval_dir)
    return report


def _local_torch_cuda_device_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:
        return 0


def run_ray_tune_search(
    config_arg: str | Path,
    search_specs: list[str],
    overrides: dict[str, object] | None = None,
    num_samples: int = 1,
    scheduler_name: str = "asha",
    search_algorithm: str = "optuna",
    grace_period: int = 1,
    reduction_factor: int = 2,
    cpus_per_trial: int = 1,
    gpus_per_trial: float = 0.0,
    max_concurrent_trials: int | None = 1,
    ray_address: str | None = None,
    ray_num_cpus: int | None = None,
    ray_num_gpus: int | None = None,
    ray_temp_dir: str | Path | None = None,
    tune_storage_path: str | Path | None = None,
    trace_trainable: bool = False,
    prune_metric_threshold: float | None = None,
    prune_min_epoch: int = 1,
) -> Path:
    # Ray 2.5x can auto-enable its uv runtime environment when it sees a
    # pyproject.toml. Workers should inherit the already-activated .venv
    # instead of creating a separate environment without the selected Torch extra.
    os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

    ray, ray_train, tune, Checkpoint, ASHAScheduler, OptunaSearch = _require_ray()

    started_at = datetime.now()
    started_perf = time.perf_counter()
    config, config_path = load_config(config_arg)
    resolved_config_arg = str(config_path)
    if overrides:
        from .config import apply_overrides

        config = apply_overrides(config, overrides)
    config = resolve_config_paths(config, config_path)

    base_name = config.experiment.name
    run_timestamp = _timestamp()
    output_root = Path(config.experiment.output_root) / "ray_tune" / base_name / run_timestamp
    output_root.mkdir(parents=True, exist_ok=False)
    tune_storage_root = Path(tune_storage_path).resolve() if tune_storage_path else output_root
    tune_run_name = base_name if tune_storage_path is None else f"{base_name}_{run_timestamp}"
    if tune_storage_path:
        tune_storage_root.mkdir(parents=True, exist_ok=True)
    search_space, search_kinds = _build_search_space(search_specs, tune)
    checkpoint_metric = config.experiment.checkpoint_metric
    metric_name = tune_metric_name(checkpoint_metric)
    if search_algorithm == "optuna" and "grid" in search_kinds:
        raise ValueError(
            "OptunaSearch does not support grid(...) search spaces. Use choice(...), randint(...), uniform(...), or loguniform(...)."
        )

    def trainable(sampled_config: dict[str, object]) -> None:
        try:
            trial_name = ray_train.get_context().get_trial_name()
        except Exception:
            trial_name = f"trial_{os.getpid()}"
        trace = TrainableTimingTrace(output_root / "trainable_timing", trial_name) if trace_trainable else None
        if trace is not None:
            trace.mark("trial_start", sampled_config=sampled_config)
            trace.mark("heavy_imports_start")
        from .runner import evaluate_checkpoint, train_experiment

        if trace is not None:
            trace.mark("heavy_imports_done")
            trace.mark("runtime_context", **_runtime_context(ray))
            trace.mark("checkpoint_restore_skipped")
        trial_overrides: dict[str, object] = {}
        trial_overrides.update(overrides or {})
        trial_overrides.update(sampled_config)
        trial_config_arg = resolved_config_arg
        export_overrides = dict(trial_overrides)
        trial_overrides["experiment.output_root"] = str(output_root / "trial_runs")

        def reporter(metrics: dict[str, object]) -> None:
            if trace is not None and not trace.first_tune_report_seen:
                trace.mark("first_tune_report_start", metrics=metrics)
            tune.report(metrics)
            if trace is not None and not trace.first_tune_report_seen:
                trace.mark("first_tune_report_done")
                trace.first_tune_report_seen = True
            raw_epoch = metrics.get("epoch", 0)
            epoch = int(float(raw_epoch)) if isinstance(raw_epoch, int | float | str) else 0
            metric_value = metrics.get(metric_name)
            if prune_metric_threshold is not None and epoch >= prune_min_epoch and isinstance(metric_value, int | float):
                if metric_value > prune_metric_threshold:
                    report = dict(metrics)
                    report[metric_name] = FAILED_TRIAL_METRIC
                    report["checkpoint_metric"] = FAILED_TRIAL_METRIC
                    report[metric_name] = FAILED_TRIAL_METRIC
                    report["pruned_by_metric"] = (
                        f"{metric_name}={float(metric_value):.6g} exceeded threshold {prune_metric_threshold:.6g} at epoch {epoch}"
                    )
                    report["config_arg"] = trial_config_arg
                    report["applied_overrides"] = json.dumps(export_overrides, sort_keys=True, default=str)
                    raise TrialPrunedByMetric(report, metric_name=metric_name, threshold=prune_metric_threshold)

        try:
            run_dir = train_experiment(
                trial_config_arg,
                overrides=trial_overrides,
                epoch_reporter=reporter,
                timing_callback=trace.mark if trace is not None else None,
            )
            device_arg = str(trial_overrides.get("experiment.device", config.experiment.device))
            if trace is not None:
                trace.mark("final_checkpoint_restore_start", path=str(Path(run_dir) / "best.pt"))
            eval_dir = evaluate_checkpoint(Path(run_dir) / "best.pt", "val", device_arg=device_arg)
            if trace is not None:
                trace.mark("final_checkpoint_restore_done", path=str(Path(run_dir) / "best.pt"), eval_dir=str(eval_dir))
        except TrialPrunedByMetric as exc:
            if trace is not None:
                trace.mark("metric_pruned_trial_terminated", error=str(exc), report=exc.report)
            tune.report(exc.report)
            return
        except FloatingPointError as exc:
            report = {
                metric_name: FAILED_TRIAL_METRIC,
                "checkpoint_metric": checkpoint_metric,
                "nonfinite_failure": str(exc),
                "config_arg": trial_config_arg,
                "applied_overrides": json.dumps(export_overrides, sort_keys=True, default=str),
            }
            if trace is not None:
                trace.mark("nonfinite_trial_terminated", error=str(exc), report=report)
            tune.report(report)
            return
        report = _build_final_tune_report(
            run_dir=run_dir,
            eval_dir=eval_dir,
            checkpoint_metric=checkpoint_metric,
            metric_name=metric_name,
            trial_config_arg=trial_config_arg,
            export_overrides=export_overrides,
        )
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_root = Path(checkpoint_dir)
            if trace is not None:
                trace.mark("ray_checkpoint_write_start", checkpoint_dir=str(checkpoint_root))
            shutil.copy2(Path(run_dir) / "best.pt", checkpoint_root / "best.pt")
            shutil.copy2(run_dir / "resolved_config.json", checkpoint_root / "resolved_config.json")
            tune.report(report, checkpoint=Checkpoint.from_directory(checkpoint_dir))
            if trace is not None:
                trace.mark("ray_checkpoint_write_done", checkpoint_dir=str(checkpoint_root))

    resources: dict[str, int | float] = {"cpu": cpus_per_trial}
    if gpus_per_trial > 0:
        resources["gpu"] = gpus_per_trial
        local_gpu_count = _local_torch_cuda_device_count()
        if local_gpu_count <= 0:
            raise RuntimeError("Ray Tune requested GPU resources, but PyTorch cannot see any CUDA devices in the driver process.")
        if ray_num_gpus is None:
            ray_num_gpus = local_gpu_count
        elif ray_num_gpus > local_gpu_count:
            print(
                f"warning: requested ray_num_gpus={ray_num_gpus}, but PyTorch sees only {local_gpu_count}; "
                f"using ray_num_gpus={local_gpu_count} to avoid invalid CUDA_VISIBLE_DEVICES assignments."
            )
            ray_num_gpus = local_gpu_count

    search_alg = None
    if search_algorithm == "optuna":
        search_alg = OptunaSearch(metric=metric_name, mode="min")

    tune_config = tune.TuneConfig(
        metric=metric_name,
        mode="min",
        num_samples=num_samples,
        search_alg=search_alg,
        max_concurrent_trials=max_concurrent_trials,
        scheduler=ASHAScheduler(
            max_t=config.training.epochs,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
        )
        if scheduler_name == "asha"
        else None,
    )

    if not ray.is_initialized():
        ray_init_kwargs: dict[str, object] = {
            "include_dashboard": False,
            "ignore_reinit_error": True,
        }
        if ray_num_cpus is not None:
            ray_init_kwargs["num_cpus"] = ray_num_cpus
        if ray_num_gpus is not None:
            ray_init_kwargs["num_gpus"] = ray_num_gpus
        if ray_temp_dir is not None:
            ray_init_kwargs["_temp_dir"] = str(Path(ray_temp_dir).resolve())
        if ray_address:
            ray.init(address=ray_address, **ray_init_kwargs)
        else:
            ray.init(**ray_init_kwargs)
    try:
        tuner = tune.Tuner(
            tune.with_resources(trainable, resources=resources),
            param_space=search_space,
            tune_config=tune_config,
            run_config=tune.RunConfig(
                name=tune_run_name,
                storage_path=str(tune_storage_root),
                verbose=1,
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result(metric=metric_name, mode="min")
        best_applied_overrides = None
        best_source_config = None
        if isinstance(best_result.metrics, dict):
            best_source_config = best_result.metrics.get("config_arg")
            raw_applied = best_result.metrics.get("applied_overrides")
            if isinstance(raw_applied, str):
                try:
                    parsed_applied = json.loads(raw_applied)
                    if isinstance(parsed_applied, dict):
                        best_applied_overrides = parsed_applied
                except json.JSONDecodeError:
                    best_applied_overrides = None
        rows = []
        for idx, result in enumerate(results):
            applied_overrides = None
            if isinstance(result.metrics, dict) and isinstance(result.metrics.get("applied_overrides"), str):
                try:
                    parsed = json.loads(str(result.metrics["applied_overrides"]))
                    if isinstance(parsed, dict):
                        applied_overrides = parsed
                except json.JSONDecodeError:
                    applied_overrides = None
            rows.append(
                {
                    "trial": idx,
                    "path": result.path,
                    "config": result.config,
                    "source_config": result.metrics.get("config_arg") if isinstance(result.metrics, dict) else None,
                    "applied_overrides": applied_overrides,
                    "metrics": result.metrics,
                    "error": None if result.error is None else str(result.error),
                }
            )
        summary = {
            "config": str(config_path),
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "elapsed_s": time.perf_counter() - started_perf,
            "search_specs": search_specs,
            "num_samples": num_samples,
            "search_algorithm": search_algorithm,
            "scheduler": scheduler_name,
            "metric": metric_name,
            "checkpoint_metric": checkpoint_metric,
            "training_epochs": config.training.epochs,
            "training_max_train_batches": config.training.max_train_batches,
            "training_max_val_batches": config.training.max_val_batches,
            "cpus_per_trial": cpus_per_trial,
            "gpus_per_trial": gpus_per_trial,
            "max_concurrent_trials": max_concurrent_trials,
            "ray_num_cpus": ray_num_cpus,
            "ray_num_gpus": ray_num_gpus,
            "ray_temp_dir": None if ray_temp_dir is None else str(ray_temp_dir),
            "tune_storage_path": str(tune_storage_root),
            "tune_run_name": tune_run_name,
            "trace_trainable": trace_trainable,
            "prune_metric_threshold": prune_metric_threshold,
            "prune_min_epoch": prune_min_epoch,
            "best_trial": {
                "path": best_result.path,
                "source_config": best_source_config or str(config_path),
                "sampled_config": best_result.config,
                "config": best_applied_overrides or best_result.config,
                "metrics": best_result.metrics,
            },
            "trials": rows,
        }
        if tune_storage_path:
            local_tune_dir = tune_storage_root / tune_run_name
            persistent_tune_dir = output_root / "ray_results"
            if local_tune_dir.exists():
                if persistent_tune_dir.exists():
                    shutil.rmtree(persistent_tune_dir)
                shutil.copytree(local_tune_dir, persistent_tune_dir)
            summary["copied_tune_results"] = str(persistent_tune_dir)
        (output_root / "ray_tune_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    finally:
        ray.shutdown()
    return output_root.resolve()
