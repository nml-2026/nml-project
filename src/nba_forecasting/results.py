from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from .config import RunConfig, apply_overrides, load_config, resolve_config_paths

JOB_ID_PATTERN = re.compile(r"\b\d{5,}\b")


def default_cluster_results_root() -> Path:
    user = os.environ.get("USER")
    if user:
        return Path("/scratch/izar") / user / "nba_forecasting"
    return Path("runs")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _job_id_from_path(path: Path) -> str:
    for part in path.parts:
        if JOB_ID_PATTERN.fullmatch(part):
            return part
    match = JOB_ID_PATTERN.search(str(path))
    return match.group(0) if match else ""


def _metric_value(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _checkpoint_from_metrics(metrics: dict[str, Any]) -> str:
    run_dir = metrics.get("run_dir")
    if isinstance(run_dir, str) and run_dir:
        return str(Path(run_dir) / "best.pt")
    return ""


def _compact_params(params: dict[str, Any]) -> str:
    if not params:
        return ""
    return json.dumps(params, sort_keys=True, separators=(",", ":"))


def _ray_row(summary_path: Path) -> dict[str, Any]:
    summary = _read_json(summary_path)
    best_trial = summary.get("best_trial", {})
    metrics = best_trial.get("metrics", {}) if isinstance(best_trial, dict) else {}
    config_path = str(summary.get("config", ""))
    config = Path(config_path).stem if config_path else ""
    trial_errors = [trial.get("error") for trial in summary.get("trials", []) if isinstance(trial, dict) and trial.get("error")]
    return {
        "source": "ray_tune",
        "config": config,
        "val_ade": _metric_value(metrics, "val_ade"),
        "tuned_params": _compact_params(best_trial.get("config", {}) if isinstance(best_trial, dict) else {}),
        "checkpoint": _checkpoint_from_metrics(metrics),
        "run_dir": str(metrics.get("run_dir", "")),
        "summary_path": str(summary_path),
        "status": "completed_with_errors" if trial_errors else "completed",
        "error": "; ".join(str(error) for error in trial_errors[:3]),
        "job_id": _job_id_from_path(summary_path),
    }


def _metrics_from_eval_dirs(eval_dirs: dict[str, Any]) -> dict[str, Any]:
    val_dir = eval_dirs.get("val") if isinstance(eval_dirs, dict) else None
    if not isinstance(val_dir, str):
        return {}
    metrics_path = Path(val_dir) / "metrics.json"
    if not metrics_path.exists():
        return {}
    return _read_json(metrics_path)


def _batch_rows(summary_path: Path) -> list[dict[str, Any]]:
    summary = _read_json(summary_path)
    rows = []
    for job in summary.get("jobs", []):
        if not isinstance(job, dict):
            continue
        metrics = _metrics_from_eval_dirs(job.get("eval_dirs", {}))
        run_dir = str(job.get("run_dir", ""))
        rows.append(
            {
                "source": "batch",
                "config": str(job.get("config", "")),
                "val_ade": _metric_value(metrics, "ade"),
                "tuned_params": "",
                "checkpoint": str(Path(run_dir) / "best.pt") if run_dir else "",
                "run_dir": run_dir,
                "summary_path": str(summary_path),
                "status": str(job.get("status", "")),
                "error": str(job.get("error", "")),
                "job_id": _job_id_from_path(summary_path),
            }
        )
    return rows


def collect_result_rows(root: str | Path | None = None) -> list[dict[str, Any]]:
    search_root = Path(root) if root is not None else default_cluster_results_root()
    if not search_root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for summary_path in search_root.rglob("ray_tune_summary.json"):
        rows.append(_ray_row(summary_path))
    for summary_path in search_root.rglob("batch_summary.json"):
        rows.extend(_batch_rows(summary_path))
    rows.sort(key=lambda row: (row["val_ade"] is None, row["val_ade"] if row["val_ade"] is not None else float("inf"), row["config"]))
    return rows


RESULT_COLUMNS = ["source", "config", "val_ade", "tuned_params", "checkpoint", "run_dir", "status", "error", "job_id", "summary_path"]


def format_results_csv(rows: list[dict[str, Any]]) -> str:
    from io import StringIO

    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=RESULT_COLUMNS)
    writer.writeheader()
    for row in rows:
        writer.writerow({column: "" if row.get(column) is None else row.get(column, "") for column in RESULT_COLUMNS})
    return output.getvalue()


def format_results_markdown(rows: list[dict[str, Any]]) -> str:
    columns = ["rank", "source", "config", "val_ade", "status", "job_id", "checkpoint"]
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for rank, row in enumerate(rows, start=1):
        val_ade = "" if row.get("val_ade") is None else f"{row['val_ade']:.4f}"
        values = [
            str(rank),
            str(row.get("source", "")),
            str(row.get("config", "")),
            val_ade,
            str(row.get("status", "")),
            str(row.get("job_id", "")),
            str(row.get("checkpoint", "")),
        ]
        table.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(table) + "\n"


def _summary_elapsed_s(summary: dict[str, Any]) -> float | None:
    value = summary.get("elapsed_s")
    if isinstance(value, int | float) and value > 0:
        return float(value)
    trial_times = [
        metrics.get("time_total_s")
        for trial in summary.get("trials", [])
        if isinstance(trial, dict)
        for metrics in [trial.get("metrics", {})]
        if isinstance(metrics, dict) and isinstance(metrics.get("time_total_s"), int | float)
    ]
    if trial_times:
        return max(float(value) for value in trial_times)
    return None


def collect_runtime_rows(root: str | Path) -> list[dict[str, Any]]:
    search_root = Path(root)
    rows: list[dict[str, Any]] = []
    if not search_root.exists():
        return rows
    for summary_path in sorted(search_root.rglob("ray_tune_summary.json")):
        summary = _read_json(summary_path)
        config_path = str(summary.get("config", ""))
        trials = [trial for trial in summary.get("trials", []) if isinstance(trial, dict)]
        errors = [trial for trial in trials if trial.get("error")]
        rows.append(
            {
                "config": Path(config_path).stem if config_path else "",
                "summary_path": str(summary_path),
                "job_id": _job_id_from_path(summary_path),
                "elapsed_s": _summary_elapsed_s(summary),
                "num_samples": int(summary.get("num_samples", len(trials)) or len(trials) or 0),
                "completed_trials": len(trials) - len(errors),
                "errored_trials": len(errors),
                "training_epochs": int(summary.get("training_epochs") or 0),
                "max_train_batches": int(summary.get("training_max_train_batches") or 0),
                "max_val_batches": int(summary.get("training_max_val_batches") or 0),
                "max_concurrent_trials": int(summary.get("max_concurrent_trials") or 1),
            }
        )
    return rows


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    minutes = int(round(seconds / 60))
    hours, rem_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {rem_minutes:02d}m"
    return f"{rem_minutes}m"


def _project_runtime_s(row: dict[str, Any], target_samples: int, target_epochs: int) -> float | None:
    elapsed_s = row.get("elapsed_s")
    observed_samples = int(row.get("num_samples") or 0)
    observed_epochs = int(row.get("training_epochs") or 0)
    max_concurrent = max(1, int(row.get("max_concurrent_trials") or 1))
    if not isinstance(elapsed_s, int | float) or elapsed_s <= 0 or observed_samples <= 0 or observed_epochs <= 0:
        return None
    observed_waves = math.ceil(observed_samples / max_concurrent)
    target_waves = math.ceil(target_samples / max_concurrent)
    return float(elapsed_s) * (target_waves / observed_waves) * (target_epochs / observed_epochs)


def format_runtime_estimate_markdown(rows: list[dict[str, Any]], target_samples: int, target_epochs: int) -> str:
    columns = [
        "config",
        "observed",
        "samples",
        "epochs",
        "concurrency",
        "projection",
        "notes",
    ]
    table = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    total_s = 0.0
    complete_estimates = True
    for row in rows:
        projected_s = _project_runtime_s(row, target_samples, target_epochs)
        if projected_s is None:
            complete_estimates = False
        else:
            total_s += projected_s
        notes = []
        if row.get("errored_trials"):
            notes.append(f"{row['errored_trials']} errored")
        if row.get("max_train_batches") or row.get("max_val_batches"):
            notes.append("batch-limited")
        values = [
            str(row.get("config", "")),
            _format_duration(row.get("elapsed_s")),
            str(row.get("num_samples", "")),
            str(row.get("training_epochs", "")),
            str(row.get("max_concurrent_trials", "")),
            _format_duration(projected_s),
            ", ".join(notes),
        ]
        table.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    if rows and complete_estimates:
        table.append("")
        table.append(f"Projected sequential total for observed configs: **{_format_duration(total_s)}**.")
    elif rows:
        table.append("")
        table.append("Projected total unavailable because at least one summary lacks timing metadata.")
    else:
        table.append("")
        table.append("No Ray Tune summaries found.")
    return "\n".join(table) + "\n"


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    raise TypeError(f"Cannot write TOML value {value!r}.")


def _toml_from_config(config: RunConfig) -> str:
    data = asdict(config) if is_dataclass(config) else config.to_dict()
    lines: list[str] = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if value is None:
                continue
            lines.append(f"{key} = {_toml_value(value)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_best_config(summary_path: str | Path, output_path: str | Path | None = None) -> Path:
    summary_file = Path(summary_path)
    summary = _read_json(summary_file)
    config_arg = summary.get("config")
    if not isinstance(config_arg, str) or not config_arg:
        raise ValueError(f"{summary_file} does not contain a source config path.")
    best_trial = summary.get("best_trial", {})
    best_overrides = best_trial.get("config", {}) if isinstance(best_trial, dict) else {}
    if not isinstance(best_overrides, dict):
        raise ValueError(f"{summary_file} does not contain best-trial config overrides.")
    config, config_path = load_config(config_arg)
    resolved = apply_overrides(config, best_overrides)
    output = Path(output_path) if output_path else summary_file.parent / "best_config.toml"
    if output.parent.name != "configs":
        resolved = resolve_config_paths(resolved, config_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_toml_from_config(resolved))
    return output.resolve()
