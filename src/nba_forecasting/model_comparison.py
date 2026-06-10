"""Model-comparison job registry and batch runner for the report table."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from .runner import evaluate_checkpoint, train_experiment

DEFAULT_COMPARISON_ROOT = Path("runs/model_comparison")
DEFAULT_CONNECTIVITY_ROOT = Path("runs/connectivity_ablations")
DEFAULT_SMOKE_ROOT = Path("runs/model_comparison_smoke")
DEFAULT_LOG_FILE = Path("logs/model_comparison.log")

SMOKE_MAX_TRAIN_BATCHES = 4
SMOKE_MAX_VAL_BATCHES = 2


@dataclass(frozen=True)
class ComparisonJob:
    name: str
    report_label: str
    config: str
    epochs: int = 80
    patience: int | None = None
    output_category: str = "comparison"  # comparison | connectivity
    name_suffix: str | None = None
    extra_overrides: Mapping[str, object] = field(default_factory=dict)

    def resolved_patience(self) -> int:
        return self.epochs if self.patience is None else self.patience

    def run_folder_name(self) -> str:
        """experiment.name directory under output_root (unique per job)."""
        if self.name_suffix:
            return f"{self.config}_{self.name_suffix}"
        return self.config

    def output_root(self, *, comparison_root: Path, connectivity_root: Path, smoke: bool, smoke_root: Path) -> Path:
        if self.output_category == "connectivity":
            return connectivity_root
        if smoke:
            return smoke_root
        return comparison_root

    def with_name_suffix(self, suffix: str | None) -> ComparisonJob:
        if self.output_category == "connectivity" or not suffix:
            return self
        return replace(self, name_suffix=suffix)


def _job(
    name: str,
    report_label: str,
    config: str,
    *,
    epochs: int = 80,
    output_category: str = "comparison",
    name_suffix: str | None = None,
    extra_overrides: Mapping[str, object] | None = None,
) -> ComparisonJob:
    return ComparisonJob(
        name=name,
        report_label=report_label,
        config=config,
        epochs=epochs,
        output_category=output_category,
        name_suffix=name_suffix,
        extra_overrides=dict(extra_overrides or {}),
    )


ALL_JOBS: tuple[ComparisonJob, ...] = (
    _job("baseline", "Per-entity GRU", "baseline"),
    _job("joint_gru", "Joint GRU", "joint_gru"),
    _job("static_graph", "Static graph MPNN", "static_graph"),
    _job("dynamic_graph_autoregressive", "Dynamic MPNN (autoreg)", "dynamic_graph_autoregressive"),
    _job("dynamic_graph_direct", "Dynamic MPNN (direct)", "dynamic_graph_direct"),
    _job("graph_transformer", "Graph Transformer (delta)", "graph_transformer"),
    _job("graph_transformer_mdn", "Graph Transformer (MDN)", "graph_transformer_mdn"),
    _job("mart", "MART", "mart", epochs=100),
    _job(
        "connectivity_no_graph",
        "GT no_graph",
        "graph_transformer",
        output_category="connectivity",
        name_suffix="graph_ablation_no_graph",
        extra_overrides={"model.graph_transformer.graph_ablation_mode": "no_graph"},
    ),
    _job(
        "connectivity_shuffled_graph",
        "GT shuffled_graph",
        "graph_transformer",
        output_category="connectivity",
        name_suffix="graph_ablation_shuffled_graph",
        extra_overrides={"model.graph_transformer.graph_ablation_mode": "shuffled_graph"},
    ),
)

MART_AUG_JOBS: tuple[ComparisonJob, ...] = (
    _job("mart_aug_none", "MART (aug none)", "mart_aug_none", epochs=100),
    _job("mart_aug_reflect", "MART (aug reflect)", "mart_aug_reflect", epochs=100),
    _job("mart_aug_perm", "MART (aug perm)", "mart_aug_perm", epochs=100),
)

JOB_REGISTRY: tuple[ComparisonJob, ...] = ALL_JOBS + MART_AUG_JOBS
ALL_JOB_NAMES: frozenset[str] = frozenset(job.name for job in JOB_REGISTRY)


@dataclass(frozen=True)
class ModelComparisonSettings:
    device: str = "cuda"
    comparison_root: Path = DEFAULT_COMPARISON_ROOT
    connectivity_root: Path = DEFAULT_CONNECTIVITY_ROOT
    smoke_root: Path = DEFAULT_SMOKE_ROOT
    log_file: Path = DEFAULT_LOG_FILE
    name_suffix: str | None = None
    smoke: bool = False


def resolve_jobs(
    *,
    job: str | None = None,
    jobs: list[str] | None = None,
    skip_jobs: list[str] | None = None,
) -> list[ComparisonJob]:
    if job is not None:
        if jobs:
            raise ValueError("Use either --job or --jobs, not both.")
        if job not in ALL_JOB_NAMES:
            raise ValueError(f"Unknown job {job!r}. Expected one of: {', '.join(sorted(ALL_JOB_NAMES))}")
        return [_job_by_name(job)]

    selected = list(ALL_JOBS)
    if jobs:
        unknown = [name for name in jobs if name not in ALL_JOB_NAMES]
        if unknown:
            raise ValueError(f"Unknown job(s): {', '.join(unknown)}")
        selected = [_job_by_name(name) for name in jobs]

    skip = set(skip_jobs or [])
    if skip:
        unknown_skip = sorted(skip - ALL_JOB_NAMES)
        if unknown_skip:
            raise ValueError(f"Unknown skip job(s): {', '.join(unknown_skip)}")
        selected = [item for item in selected if item.name not in skip]
    return selected


def latest_run_dir(root: Path, folder_name: str) -> Path | None:
    """Return the newest run under ``root/folder_name/`` that has a checkpoint."""
    search = root / folder_name
    if not search.is_dir():
        return None
    runs = [history.parent for history in search.rglob("history.csv")]
    if not runs:
        return None
    complete = [run_dir for run_dir in runs if (run_dir / "best.pt").is_file()]
    candidates = complete or runs
    return max(candidates, key=lambda path: path.stat().st_mtime)


def report_table_specs(
    *,
    comparison_root: Path = DEFAULT_COMPARISON_ROOT,
    connectivity_root: Path = DEFAULT_CONNECTIVITY_ROOT,
) -> list[tuple[str, Path, str]]:
    specs: list[tuple[str, Path, str]] = []
    for job in ALL_JOBS:
        root = job.output_root(
            comparison_root=comparison_root,
            connectivity_root=connectivity_root,
            smoke=False,
            smoke_root=DEFAULT_SMOKE_ROOT,
        )
        specs.append((job.report_label, root, job.run_folder_name()))
    return specs


REPORT_TABLE_SPECS: list[tuple[str, Path, str]] = report_table_specs()[:8]
FULL_SPECS: list[tuple[str, Path, str]] = report_table_specs()


def _job_by_name(name: str) -> ComparisonJob:
    for job in JOB_REGISTRY:
        if job.name == name:
            return job
    raise KeyError(name)


def _log(message: str, *, log_file: Path) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def latest_checkpoint(output_root: Path, folder_name: str) -> Path | None:
    run_dir = latest_run_dir(output_root, folder_name)
    if run_dir is None:
        return None
    checkpoint = run_dir / "best.pt"
    return checkpoint if checkpoint.is_file() else None


def _train_overrides(job: ComparisonJob, settings: ModelComparisonSettings) -> dict[str, object]:
    overrides: dict[str, object] = {
        "experiment.device": settings.device,
        "experiment.output_root": str(
            job.output_root(
                comparison_root=settings.comparison_root,
                connectivity_root=settings.connectivity_root,
                smoke=settings.smoke,
                smoke_root=settings.smoke_root,
            )
        ),
        "evaluation.tta_group": "none",
    }
    overrides.update(job.extra_overrides)
    if settings.smoke:
        overrides.update(
            {
                "training.epochs": 1,
                "training.patience": 1,
                "training.max_train_batches": SMOKE_MAX_TRAIN_BATCHES,
                "training.max_val_batches": SMOKE_MAX_VAL_BATCHES,
            }
        )
    else:
        overrides.update(
            {
                "training.epochs": job.epochs,
                "training.patience": job.resolved_patience(),
            }
        )
    return overrides


def _eval_checkpoint_both_tta(checkpoint: Path, device: str) -> None:
    for split_name in ("val", "holdout"):
        for tta_group in ("none", "xy"):
            evaluate_checkpoint(
                checkpoint,
                split_name,
                device_arg=device,
                tta_group=tta_group,
            )


def run_job(job: ComparisonJob, settings: ModelComparisonSettings) -> Path:
    job = job.with_name_suffix(settings.name_suffix)
    output_root = job.output_root(
        comparison_root=settings.comparison_root,
        connectivity_root=settings.connectivity_root,
        smoke=settings.smoke,
        smoke_root=settings.smoke_root,
    )
    _log(
        "train "
        f"job={job.name} config={job.config} epochs={job.epochs if not settings.smoke else 1} "
        f"output_root={output_root} name_suffix={job.name_suffix or '<none>'}",
        log_file=settings.log_file,
    )

    overrides = _train_overrides(job, settings)
    if job.name_suffix:
        overrides["experiment.name"] = f"{job.config}_{job.name_suffix}"
    run_dir = train_experiment(job.config, overrides=overrides)

    checkpoint = run_dir / "best.pt"
    if not checkpoint.is_file():
        folder_name = job.run_folder_name()
        fallback = latest_checkpoint(output_root, folder_name)
        if fallback is not None:
            checkpoint = fallback
        else:
            raise FileNotFoundError(f"checkpoint missing for job={job.name} run_dir={run_dir} folder={folder_name} under {output_root}")

    if settings.smoke:
        _log(f"status=OK smoke job={job.name} checkpoint={checkpoint} (skipped TTA eval)", log_file=settings.log_file)
        return checkpoint

    _log(f"eval checkpoint={checkpoint}", log_file=settings.log_file)
    _eval_checkpoint_both_tta(checkpoint, settings.device)
    _log(f"status=OK job={job.name} checkpoint={checkpoint}", log_file=settings.log_file)
    return checkpoint


def _parse_gpu_list(raw: str | None) -> list[int]:
    if not raw:
        return [0]
    gpus: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        gpus.append(int(part))
    return gpus or [0]


def _subprocess_env(*, gpu_id: int, training_slots: int, settings: ModelComparisonSettings) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["NBA_TRAINING_SLOTS"] = str(training_slots)
    env["NBA_FORECASTING_DISABLE_TQDM"] = "1" if training_slots > 1 else env.get("NBA_FORECASTING_DISABLE_TQDM", "")
    return env


def _subprocess_command(job_name: str, settings: ModelComparisonSettings) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "nba_forecasting",
        "model-comparison",
        "--job",
        job_name,
        "--device",
        settings.device,
        "--comparison-root",
        str(settings.comparison_root),
        "--connectivity-root",
        str(settings.connectivity_root),
    ]
    if settings.smoke:
        command.append("--smoke")
        command.extend(["--smoke-root", str(settings.smoke_root)])
    if settings.name_suffix:
        command.extend(["--name-suffix", settings.name_suffix])
    if settings.log_file != DEFAULT_LOG_FILE:
        command.extend(["--log-file", str(settings.log_file)])
    return command


def run_batch(
    jobs: list[ComparisonJob],
    *,
    settings: ModelComparisonSettings,
    gpus: list[int],
) -> None:
    if not jobs:
        return

    settings.log_file.parent.mkdir(parents=True, exist_ok=True)

    if len(gpus) == 1:
        os.environ["NBA_TRAINING_SLOTS"] = "1"
        _log(f"sequential mode gpu={gpus[0]} jobs={' '.join(job.name for job in jobs)}", log_file=settings.log_file)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[0])
        failures: list[str] = []
        for job in jobs:
            _log(f"start job={job.name}", log_file=settings.log_file)
            try:
                run_job(job, settings)
                _log(f"OK job={job.name}", log_file=settings.log_file)
            except Exception as exc:
                _log(f"FAILED job={job.name} error={type(exc).__name__}: {exc}", log_file=settings.log_file)
                failures.append(job.name)
        if failures:
            raise RuntimeError(f"model comparison failed for: {', '.join(failures)}")
        return

    training_slots = min(len(jobs), len(gpus))
    os.environ["NBA_TRAINING_SLOTS"] = str(training_slots)
    _log(
        f"parallel mode gpus={gpus} NBA_TRAINING_SLOTS={training_slots} jobs={' '.join(job.name for job in jobs)}",
        log_file=settings.log_file,
    )

    processes: list[tuple[subprocess.Popen[bytes], str, int]] = []
    for index, job in enumerate(jobs):
        gpu_id = gpus[index % len(gpus)]
        env = _subprocess_env(gpu_id=gpu_id, training_slots=training_slots, settings=settings)
        command = _subprocess_command(job.name, settings)
        _log(f"launch gpu={gpu_id} job={job.name}", log_file=settings.log_file)
        processes.append(
            (
                subprocess.Popen(command, env=env),
                job.name,
                gpu_id,
            )
        )

    failures: list[str] = []
    for process, job_name, gpu_id in processes:
        return_code = process.wait()
        if return_code != 0:
            _log(f"FAILED gpu={gpu_id} job={job_name} exit={return_code}", log_file=settings.log_file)
            failures.append(job_name)
        else:
            _log(f"OK gpu={gpu_id} job={job_name}", log_file=settings.log_file)
    if failures:
        raise RuntimeError(f"model comparison failed for: {', '.join(failures)}")


def register_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--smoke", action="store_true", help="1-epoch capped-batch pre-flight.")
    parser.add_argument("--job", default=None, help="Run a single named job.")
    parser.add_argument("--jobs", default=None, help="Comma-separated job subset (default: all 10).")
    parser.add_argument("--skip-jobs", default=None, help="Comma-separated jobs to skip.")
    parser.add_argument("--gpus", default=None, help="GPU ids for batch mode (default: 0).")
    parser.add_argument("--device", default="cuda", help="Training/eval device.")
    parser.add_argument("--comparison-root", type=Path, default=DEFAULT_COMPARISON_ROOT)
    parser.add_argument("--connectivity-root", type=Path, default=DEFAULT_CONNECTIVITY_ROOT)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE)
    parser.add_argument("--name-suffix", default=None, help="Optional suffix for comparison-root runs.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate model-comparison report-table jobs.")
    register_arguments(parser)
    return parser


def _split_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def run_from_args(args: argparse.Namespace) -> int:
    settings = ModelComparisonSettings(
        device=args.device,
        comparison_root=args.comparison_root,
        connectivity_root=args.connectivity_root,
        smoke_root=args.smoke_root,
        log_file=args.log_file,
        name_suffix=args.name_suffix,
        smoke=args.smoke,
    )

    jobs = resolve_jobs(job=args.job, jobs=_split_csv(args.jobs), skip_jobs=_split_csv(args.skip_jobs))
    if not jobs:
        print("No jobs selected.", file=sys.stderr)
        return 1

    if args.job is not None:
        run_job(jobs[0], settings)
        return 0

    gpus = _parse_gpu_list(args.gpus)
    _log(
        f"command=model-comparison smoke={int(settings.smoke)} jobs={' '.join(job.name for job in jobs)}",
        log_file=settings.log_file,
    )
    run_batch(jobs, settings=settings, gpus=gpus)
    _log("status=OK", log_file=settings.log_file)
    _log("collect: python3 scripts/collect_results.py", log_file=settings.log_file)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_from_args(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
