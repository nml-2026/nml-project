from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import parse_override_value
from .data import prepare_remote_dataset
from .doctor import run_doctor
from .model_comparison import register_arguments as register_model_comparison_arguments
from .model_comparison import run_from_args as run_model_comparison
from .ray_tune import SEARCH_PROFILES, resolve_search_specs, run_ray_tune_search
from .results import (
    collect_result_rows,
    default_cluster_results_root,
    export_best_config,
    format_results_csv,
    format_results_markdown,
)
from .runner import (
    cross_validate_experiment,
    evaluate_checkpoint,
    evaluate_ensemble,
    repeat_trials_experiment,
    submit_checkpoint,
    submit_ensemble,
    train_experiment,
    train_many_experiments,
)


def _add_runtime_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=None, help="Override experiment.device, e.g. cpu, mps, cuda, auto.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override training.num_workers (-1=auto, 0=disable, N=explicit).",
    )
    parser.add_argument("--output-root", default=None, help="Override experiment.output_root.")
    parser.add_argument("--name-suffix", default=None, help="Append a suffix to experiment.name.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Apply an extra dotted-path override such as model.common.hidden_dim=256.",
    )


def _build_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if getattr(args, "device", None) is not None:
        overrides["experiment.device"] = args.device
    if getattr(args, "seed", None) is not None:
        overrides["experiment.seed"] = args.seed
    if getattr(args, "epochs", None) is not None:
        overrides["training.epochs"] = args.epochs
    if getattr(args, "batch_size", None) is not None:
        overrides["training.batch_size"] = args.batch_size
    if getattr(args, "num_workers", None) is not None:
        overrides["training.num_workers"] = args.num_workers
    if getattr(args, "output_root", None) is not None:
        overrides["experiment.output_root"] = args.output_root
    for raw in getattr(args, "set", []) or []:
        if "=" not in raw:
            raise ValueError(f"Invalid override {raw!r}. Expected PATH=VALUE.")
        path, value = raw.split("=", 1)
        overrides[path] = parse_override_value(value)
    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NBA trajectory forecasting CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train one or more configs sequentially.")
    train_parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        help="One or more config names under configs/ or direct TOML paths.",
    )
    train_parser.add_argument("--resume-checkpoint", default=None, help="Resume training from a last.pt checkpoint (single-config only).")
    train_parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seed list for batch training.")
    train_parser.add_argument(
        "--evaluate-splits",
        nargs="*",
        choices=["train", "val", "holdout"],
        default=None,
        help="Optionally evaluate each completed run on these splits.",
    )
    train_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch immediately if one run fails.",
    )
    _add_runtime_override_args(train_parser)

    repeat_parser = subparsers.add_parser("repeat-trials", help="Train the same config multiple times and aggregate metrics.")
    repeat_parser.add_argument("--config", required=True, help="Config name under configs/ or a direct TOML path.")
    repeat_parser.add_argument("--repeats", type=int, required=True, help="Number of repeated trials.")
    repeat_parser.add_argument("--seed-step", type=int, default=1, help="Increment between repeated seeds.")
    repeat_parser.add_argument(
        "--evaluate-splits",
        nargs="*",
        choices=["train", "val", "holdout"],
        default=None,
        help="Optionally evaluate each completed trial on these splits in addition to val.",
    )
    repeat_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the repeat run immediately if one trial fails.",
    )
    _add_runtime_override_args(repeat_parser)

    cv_parser = subparsers.add_parser("cross-validate", help="Run deterministic k-fold cross-validation for one config.")
    cv_parser.add_argument("--config", required=True, help="Config name under configs/ or a direct TOML path.")
    cv_parser.add_argument("--folds", type=int, required=True, help="Number of CV folds.")
    cv_parser.add_argument(
        "--include-holdout",
        action="store_true",
        help="Include the frozen holdout sequences in the CV pool instead of keeping them fixed.",
    )
    cv_parser.add_argument(
        "--evaluate-splits",
        nargs="*",
        choices=["train", "val", "holdout"],
        default=None,
        help="Optionally evaluate each completed fold on these splits in addition to val.",
    )
    cv_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop cross-validation immediately if one fold fails.",
    )
    _add_runtime_override_args(cv_parser)

    tune_parser = subparsers.add_parser("tune", help="Run Ray Tune + OptunaSearch hyperparameter tuning for one config.")
    tune_parser.add_argument("--config", required=True, help="Config name under configs/ or a direct TOML path.")
    tune_parser.add_argument(
        "--search",
        action="append",
        default=[],
        metavar="PATH=EXPR",
        help="Ray Tune search expression such as model.common.hidden_dim=choice(128,192,256) or training.lr=loguniform(1e-4,5e-3).",
    )
    tune_parser.add_argument(
        "--search-profile",
        choices=["none", *sorted(SEARCH_PROFILES)],
        default="none",
        help="Named Ray Tune search profile. Explicit --search specs are appended and can override profile paths.",
    )
    tune_parser.add_argument("--num-samples", type=int, default=8, help="Number of Ray Tune samples to draw.")
    tune_parser.add_argument(
        "--search-alg",
        choices=["optuna", "none"],
        default="optuna",
        help="Search algorithm to use with Ray Tune. Use none for exact grid search.",
    )
    tune_parser.add_argument("--scheduler", choices=["asha", "none"], default="asha", help="Early stopping scheduler.")
    tune_parser.add_argument("--grace-period", type=int, default=1, help="ASHA grace period in epochs.")
    tune_parser.add_argument("--reduction-factor", type=int, default=2, help="ASHA reduction factor.")
    tune_parser.add_argument("--cpus-per-trial", type=int, default=1, help="CPU resources reserved per Ray Tune trial.")
    tune_parser.add_argument("--gpus-per-trial", type=float, default=0.0, help="GPU resources reserved per Ray Tune trial.")
    tune_parser.add_argument(
        "--ray-address",
        default=None,
        help="Optional Ray cluster address such as auto or 10.0.0.1:6379. Leave unset for local single-node Ray.",
    )
    tune_parser.add_argument("--ray-num-cpus", type=int, default=None, help="Explicit CPU count passed to ray.init.")
    tune_parser.add_argument("--ray-num-gpus", type=int, default=None, help="Explicit GPU count passed to ray.init.")
    tune_parser.add_argument("--ray-temp-dir", default=None, help="Optional Ray temp directory, such as $TMPDIR/ray on Slurm.")
    tune_parser.add_argument(
        "--tune-storage-path",
        default=None,
        help="Optional Ray Tune storage root. Job-local paths are copied back under the persistent Tune output directory.",
    )
    tune_parser.add_argument(
        "--trace-trainable", action="store_true", help="Write per-trial JSONL timing traces for Tune startup diagnostics."
    )
    tune_parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=1,
        help="Maximum number of concurrently running Ray Tune trials. Keep this at 1 for local MPS runs.",
    )
    tune_parser.add_argument(
        "--prune-metric-threshold",
        type=float,
        default=None,
        help="Terminate a Tune trial as a bad sample when the checkpoint metric exceeds this value after --prune-min-epoch.",
    )
    tune_parser.add_argument(
        "--prune-min-epoch",
        type=int,
        default=1,
        help="First epoch where --prune-metric-threshold is allowed to terminate a trial.",
    )
    _add_runtime_override_args(tune_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint on a split.")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    eval_parser.add_argument("--split", required=True, choices=["train", "val", "holdout"])
    eval_parser.add_argument("--output-dir", default=None, help="Optional evaluation output directory.")
    eval_parser.add_argument("--device", default="auto", help="Device for evaluation: auto, cpu, mps, cuda.")
    eval_parser.add_argument(
        "--tta",
        choices=["none", "y", "x", "xy"],
        default=None,
        help="Test-time reflection averaging group. Defaults to checkpoint evaluation.tta_group.",
    )
    eval_parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the dataset directory baked into the checkpoint (e.g. when the checkpoint was trained on another machine).",
    )

    submit_parser = subparsers.add_parser("submit", help="Generate a Kaggle submission from a checkpoint.")
    submit_parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    submit_parser.add_argument("--output", default=None, help="Optional CSV output path.")
    submit_parser.add_argument("--device", default="auto", help="Device for submission: auto, cpu, mps, cuda.")
    submit_parser.add_argument(
        "--tta",
        choices=["none", "y", "x", "xy"],
        default=None,
        help="Test-time reflection averaging group. Defaults to checkpoint evaluation.tta_group.",
    )
    submit_parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the dataset directory baked into the checkpoint (e.g. when the checkpoint was trained on another machine).",
    )

    ens_eval_parser = subparsers.add_parser(
        "ensemble-evaluate", help="Evaluate an averaged N-checkpoint ensemble (each TTA-averaged) on a split."
    )
    ens_eval_parser.add_argument("--checkpoints", nargs="+", required=True, help="Two or more best.pt paths to average.")
    ens_eval_parser.add_argument("--split", required=True, choices=["train", "val", "holdout"])
    ens_eval_parser.add_argument("--output-dir", default=None, help="Optional evaluation output directory.")
    ens_eval_parser.add_argument("--device", default="auto", help="Device for evaluation: auto, cpu, mps, cuda.")
    ens_eval_parser.add_argument(
        "--tta",
        choices=["none", "y", "x", "xy"],
        default="xy",
        help="Per-model test-time dihedral reflection group (default xy = the LB-winning recipe).",
    )
    ens_eval_parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the dataset directory baked into the checkpoints (all legs share the frozen split).",
    )

    ens_submit_parser = subparsers.add_parser(
        "ensemble-submit", help="Generate a Kaggle submission from an averaged N-checkpoint ensemble."
    )
    ens_submit_parser.add_argument("--checkpoints", nargs="+", required=True, help="Two or more best.pt paths to average.")
    ens_submit_parser.add_argument("--output", default=None, help="Optional CSV output path.")
    ens_submit_parser.add_argument("--device", default="auto", help="Device for submission: auto, cpu, mps, cuda.")
    ens_submit_parser.add_argument(
        "--tta",
        choices=["none", "y", "x", "xy"],
        default="xy",
        help="Per-model test-time dihedral reflection group (default xy = the LB-winning recipe).",
    )
    ens_submit_parser.add_argument(
        "--data-dir",
        default=None,
        help="Override the dataset directory baked into the checkpoints (all legs share the frozen split).",
    )

    results_parser = subparsers.add_parser("summarize-results", help="Rank Ray Tune and batch results under a run root.")
    results_parser.add_argument(
        "--root",
        default=None,
        help=f"Run root to scan. Defaults to {default_cluster_results_root()}.",
    )
    results_parser.add_argument("--format", choices=["csv", "markdown"], default="csv", help="Output table format.")
    results_parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows.")
    results_parser.add_argument("--output", default=None, help="Optional output file path. Defaults to stdout.")

    export_parser = subparsers.add_parser("export-best-config", help="Export the best Ray Tune trial config as reusable TOML.")
    export_parser.add_argument("--summary", required=True, help="Path to ray_tune_summary.json.")
    export_parser.add_argument("--output", default=None, help="Output TOML path. Defaults to best_config.toml next to the summary.")

    doctor_parser = subparsers.add_parser("doctor", help="Print Python, PyTorch, CUDA, GPU, and key dependency diagnostics.")
    doctor_parser.add_argument(
        "--expect-cuda",
        action="store_true",
        help="Exit non-zero if PyTorch cannot see CUDA.",
    )
    doctor_parser.add_argument(
        "--check-inductor",
        action="store_true",
        help="Run a tiny torch.compile/Inductor execution probe and exit non-zero if it fails.",
    )
    doctor_parser.add_argument(
        "--inductor-device",
        default="cuda",
        help="Device used for --check-inductor. Defaults to cuda.",
    )

    prepare_data_parser = subparsers.add_parser(
        "prepare-data",
        help="Consolidate train/test .pt files and freeze the default train/val/holdout split for remote sync.",
    )
    prepare_data_parser.add_argument("--data-dir", default="data", help="Dataset root containing train/ and test/ directories.")
    prepare_data_parser.add_argument(
        "--split-path",
        default="artifacts/splits/default_split.json",
        help="Path for the frozen train/val/holdout split JSON.",
    )
    prepare_data_parser.add_argument("--seed", type=int, default=0, help="Split seed used only when the split file does not exist.")
    prepare_data_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate consolidated train/test files even if they already exist.",
    )

    model_comparison_parser = subparsers.add_parser(
        "model-comparison",
        help="Train and evaluate report-table model-comparison jobs.",
    )
    register_model_comparison_arguments(model_comparison_parser)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        overrides = _build_overrides(args)
        configs = args.config
        batch_mode = len(configs) > 1 or args.seeds or args.evaluate_splits or args.fail_fast
        if batch_mode:
            if args.resume_checkpoint is not None:
                raise ValueError("--resume-checkpoint is only supported for single-config training.")
            if args.name_suffix:
                overrides["experiment.name_suffix"] = args.name_suffix
            batch_dir = train_many_experiments(
                configs=configs,
                seeds=args.seeds,
                overrides=overrides,
                evaluate_splits=args.evaluate_splits,
                fail_fast=args.fail_fast,
            )
            print(f"Batch training complete. Outputs written to {batch_dir}")
            return 0
        if args.name_suffix:
            overrides["experiment.name"] = f"{Path(configs[0]).stem}_{args.name_suffix}"
        run_dir = train_experiment(configs[0], overrides=overrides, resume_checkpoint=args.resume_checkpoint)
        print(f"Training complete. Outputs written to {run_dir}")
        return 0
    if args.command == "repeat-trials":
        overrides = _build_overrides(args)
        if args.name_suffix:
            overrides["experiment.name"] = f"{Path(args.config).stem}_{args.name_suffix}"
        repeat_dir = repeat_trials_experiment(
            config_arg=args.config,
            repeats=args.repeats,
            overrides=overrides,
            evaluate_splits=args.evaluate_splits,
            fail_fast=args.fail_fast,
            seed_step=args.seed_step,
        )
        print(f"Repeat trials complete. Outputs written to {repeat_dir}")
        return 0
    if args.command == "cross-validate":
        overrides = _build_overrides(args)
        if args.name_suffix:
            overrides["experiment.name"] = f"{Path(args.config).stem}_{args.name_suffix}"
        cv_dir = cross_validate_experiment(
            config_arg=args.config,
            num_folds=args.folds,
            overrides=overrides,
            evaluate_splits=args.evaluate_splits,
            fail_fast=args.fail_fast,
            include_holdout=args.include_holdout,
        )
        print(f"Cross-validation complete. Outputs written to {cv_dir}")
        return 0
    if args.command == "tune":
        overrides = _build_overrides(args)
        if args.name_suffix:
            overrides["experiment.name"] = f"{Path(args.config).stem}_{args.name_suffix}"
        tune_dir = run_ray_tune_search(
            config_arg=args.config,
            search_specs=resolve_search_specs(args.search_profile, args.search),
            overrides=overrides,
            num_samples=args.num_samples,
            scheduler_name=args.scheduler,
            search_algorithm=args.search_alg,
            grace_period=args.grace_period,
            reduction_factor=args.reduction_factor,
            cpus_per_trial=args.cpus_per_trial,
            gpus_per_trial=args.gpus_per_trial,
            max_concurrent_trials=args.max_concurrent_trials,
            ray_address=args.ray_address,
            ray_num_cpus=args.ray_num_cpus,
            ray_num_gpus=args.ray_num_gpus,
            ray_temp_dir=args.ray_temp_dir,
            tune_storage_path=args.tune_storage_path,
            trace_trainable=args.trace_trainable,
            prune_metric_threshold=args.prune_metric_threshold,
            prune_min_epoch=args.prune_min_epoch,
        )
        print(f"Ray Tune search complete. Outputs written to {tune_dir}")
        return 0
    if args.command == "evaluate":
        eval_dir = evaluate_checkpoint(
            args.checkpoint,
            args.split,
            args.output_dir,
            device_arg=args.device,
            tta_group=args.tta,
            data_dir=args.data_dir,
        )
        print(f"Evaluation complete. Outputs written to {eval_dir}")
        return 0
    if args.command == "submit":
        submission_path = submit_checkpoint(
            args.checkpoint, args.output, device_arg=args.device, tta_group=args.tta, data_dir=args.data_dir
        )
        print(f"Submission written to {submission_path}")
        return 0
    if args.command == "ensemble-evaluate":
        eval_dir = evaluate_ensemble(
            args.checkpoints,
            args.split,
            args.output_dir,
            device_arg=args.device,
            tta_group=args.tta,
            data_dir=args.data_dir,
        )
        print(f"Ensemble evaluation complete. Outputs written to {eval_dir}")
        return 0
    if args.command == "ensemble-submit":
        submission_path = submit_ensemble(args.checkpoints, args.output, device_arg=args.device, tta_group=args.tta, data_dir=args.data_dir)
        print(f"Ensemble submission written to {submission_path}")
        return 0
    if args.command == "summarize-results":
        rows = collect_result_rows(args.root)
        if args.limit is not None:
            rows = rows[: args.limit]
        output = format_results_csv(rows) if args.format == "csv" else format_results_markdown(rows)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(output)
            print(f"Result summary written to {args.output}")
        else:
            print(output, end="")
        return 0
    if args.command == "export-best-config":
        output_path = export_best_config(args.summary, args.output)
        print(f"Best config written to {output_path}")
        return 0
    if args.command == "doctor":
        return run_doctor(expect_cuda=args.expect_cuda, check_inductor=args.check_inductor, inductor_device=args.inductor_device)
    if args.command == "prepare-data":
        summary = prepare_remote_dataset(
            data_dir=args.data_dir,
            split_path=args.split_path,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        for key, value in summary.items():
            print(f"{key}: {value}")
        return 0
    if args.command == "model-comparison":
        try:
            return run_model_comparison(args)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
    parser.error(f"Unknown command {args.command!r}")
    return 2
