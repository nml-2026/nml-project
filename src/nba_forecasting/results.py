from __future__ import annotations

import csv
import json
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


def _ray_tune_val_ade(metrics: dict[str, Any], summary: dict[str, Any]) -> float | None:
    checkpoint_metric = str(summary.get("checkpoint_metric", "ade"))
    best_metric_key = f"best_val_{checkpoint_metric}"
    return _metric_value(metrics, best_metric_key) or _metric_value(metrics, f"val_{checkpoint_metric}")


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
        "val_ade": _ray_tune_val_ade(metrics, summary),
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


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"Cannot write TOML value {value!r}.")


def _toml_from_config(config: RunConfig) -> str:
    data = asdict(config) if is_dataclass(config) else config.to_dict()
    lines: list[str] = []

    def emit_section(section: str, values: dict[str, Any]) -> None:
        scalars: dict[str, Any] = {}
        nested: dict[str, dict[str, Any]] = {}
        for key, value in values.items():
            if value is None:
                continue
            if isinstance(value, dict):
                nested[key] = value
            else:
                scalars[key] = value
        if scalars:
            lines.append(f"[{section}]")
            for key, value in scalars.items():
                lines.append(f"{key} = {_toml_value(value)}")
            lines.append("")
        for key, value in nested.items():
            emit_section(f"{section}.{key}", value)

    for section, values in data.items():
        if isinstance(values, dict):
            emit_section(section, values)
        else:
            if values is None:
                continue
            lines.append(f"{section} = {_toml_value(values)}")
    return "\n".join(lines).rstrip() + "\n"


def export_best_config(summary_path: str | Path, output_path: str | Path | None = None) -> Path:
    summary_file = Path(summary_path)
    summary = _read_json(summary_file)
    best_trial = summary.get("best_trial", {})
    config_arg = best_trial.get("source_config") if isinstance(best_trial, dict) else None
    if not isinstance(config_arg, str) or not config_arg:
        config_arg = summary.get("config")
    if not isinstance(config_arg, str) or not config_arg:
        raise ValueError(f"{summary_file} does not contain a source config path.")
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
