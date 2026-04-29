from __future__ import annotations

import json
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from .config import load_config, parse_override_value, resolve_config_paths
from .runner import evaluate_checkpoint, train_experiment

SEARCH_PATTERN = re.compile(r"^(?P<kind>[a-z_]+)\((?P<body>.*)\)$")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _require_ray():
    try:
        import ray
        from ray import train as ray_train
        from ray import tune
        from ray.tune import Checkpoint
        from ray.tune.schedulers import ASHAScheduler
        from ray.tune.search import ConcurrencyLimiter
        from ray.tune.search.optuna import OptunaSearch
    except ImportError as exc:  # pragma: no cover - exercised only when Ray is missing
        raise RuntimeError(
            "Ray Tune is required for hyperparameter tuning. Install it with "
            '`pip install "ray[tune]>=2.52.1" "optuna>=3.6"` or sync the project environment first.'
        ) from exc
    return ray, ray_train, tune, Checkpoint, ASHAScheduler, ConcurrencyLimiter, OptunaSearch


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
        return tune.uniform(float(low), float(high))
    if kind == "loguniform":
        low, high = _parse_list(body)
        return tune.loguniform(float(low), float(high))
    if kind == "randint":
        low, high = _parse_list(body)
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
) -> Path:
    ray, ray_train, tune, Checkpoint, ASHAScheduler, ConcurrencyLimiter, OptunaSearch = _require_ray()
    config, config_path = load_config(config_arg)
    resolved_config_arg = str(config_path)
    if overrides:
        from .config import apply_overrides

        config = apply_overrides(config, overrides)
    config = resolve_config_paths(config, config_path)

    base_name = config.experiment.name
    output_root = Path(config.experiment.output_root) / "ray_tune" / base_name / _timestamp()
    output_root.mkdir(parents=True, exist_ok=False)
    search_space, search_kinds = _build_search_space(search_specs, tune)
    metric_name = f"val_{config.experiment.checkpoint_metric}"
    if search_algorithm == "optuna" and "grid" in search_kinds:
        raise ValueError(
            "OptunaSearch does not support grid(...) search spaces. Use choice(...), randint(...), uniform(...), or loguniform(...)."
        )

    def trainable(sampled_config: dict[str, object]) -> None:
        trial_overrides = {}
        trial_overrides.update(overrides or {})
        trial_overrides.update(sampled_config)
        trial_overrides["experiment.output_root"] = str(output_root / "trial_runs")

        def reporter(metrics: dict[str, object]) -> None:
            tune.report(metrics)

        run_dir = train_experiment(resolved_config_arg, overrides=trial_overrides, epoch_reporter=reporter)
        device_arg = str(trial_overrides.get("experiment.device", config.experiment.device))
        eval_dir = evaluate_checkpoint(Path(run_dir) / "best.pt", "val", device_arg=device_arg)
        final_metrics = json.loads((eval_dir / "metrics.json").read_text())
        report = {f"val_{key}": value for key, value in final_metrics.items()}
        report["run_dir"] = str(run_dir)
        report["eval_dir"] = str(eval_dir)
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            checkpoint_root = Path(checkpoint_dir)
            shutil.copy2(Path(run_dir) / "best.pt", checkpoint_root / "best.pt")
            shutil.copy2(run_dir / "resolved_config.json", checkpoint_root / "resolved_config.json")
            tune.report(report, checkpoint=Checkpoint.from_directory(checkpoint_dir))

    resources = {"cpu": cpus_per_trial}
    if gpus_per_trial > 0:
        resources["gpu"] = gpus_per_trial

    search_alg = None
    if search_algorithm == "optuna":
        search_alg = OptunaSearch(metric=metric_name, mode="min")
        if max_concurrent_trials is not None:
            search_alg = ConcurrencyLimiter(search_alg, max_concurrent=max_concurrent_trials)

    tune_config = tune.TuneConfig(
        metric=metric_name,
        mode="min",
        num_samples=num_samples,
        search_alg=search_alg,
        max_concurrent_trials=None if search_alg is not None else max_concurrent_trials,
        scheduler=ASHAScheduler(
            max_t=config.training.epochs,
            grace_period=grace_period,
            reduction_factor=reduction_factor,
        )
        if scheduler_name == "asha"
        else None,
    )

    if not ray.is_initialized():
        if ray_address:
            ray.init(address=ray_address, ignore_reinit_error=True)
        else:
            ray.init(include_dashboard=False, ignore_reinit_error=True)
    try:
        tuner = tune.Tuner(
            tune.with_resources(trainable, resources=resources),
            param_space=search_space,
            tune_config=tune_config,
            run_config=tune.RunConfig(
                name=base_name,
                storage_path=str(output_root),
                verbose=1,
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result(metric=metric_name, mode="min")
        rows = []
        for idx, result in enumerate(results):
            rows.append(
                {
                    "trial": idx,
                    "path": result.path,
                    "config": result.config,
                    "metrics": result.metrics,
                    "error": None if result.error is None else str(result.error),
                }
            )
        summary = {
            "config": str(config_path),
            "search_specs": search_specs,
            "num_samples": num_samples,
            "search_algorithm": search_algorithm,
            "scheduler": scheduler_name,
            "metric": metric_name,
            "best_trial": {
                "path": best_result.path,
                "config": best_result.config,
                "metrics": best_result.metrics,
            },
            "trials": rows,
        }
        (output_root / "ray_tune_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    finally:
        ray.shutdown()
    return output_root.resolve()
