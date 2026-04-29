from __future__ import annotations

import argparse
from pathlib import Path

from .config import parse_override_value
from .ray_tune import run_ray_tune_search
from .runner import (
    cross_validate_experiment,
    evaluate_checkpoint,
    repeat_trials_experiment,
    submit_checkpoint,
    train_experiment,
    train_many_experiments,
)


def _add_runtime_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default=None, help="Override experiment.device, e.g. cpu, mps, cuda, auto.")
    parser.add_argument("--seed", type=int, default=None, help="Override experiment.seed.")
    parser.add_argument("--epochs", type=int, default=None, help="Override training.epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers.")
    parser.add_argument("--output-root", default=None, help="Override experiment.output_root.")
    parser.add_argument("--name-suffix", default=None, help="Append a suffix to experiment.name.")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="PATH=VALUE",
        help="Apply an extra dotted-path override such as model.hidden_dim=256.",
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

    train_parser = subparsers.add_parser("train", help="Train one experiment from a config.")
    train_parser.add_argument("--config", required=True, help="Config name under configs/ or a direct TOML path.")
    _add_runtime_override_args(train_parser)

    train_many_parser = subparsers.add_parser("train-many", help="Train multiple configs sequentially.")
    train_many_parser.add_argument("--configs", nargs="+", required=True, help="One or more config names or TOML paths.")
    train_many_parser.add_argument("--seeds", nargs="*", type=int, default=None, help="Optional seed list.")
    train_many_parser.add_argument(
        "--evaluate-splits",
        nargs="*",
        choices=["train", "val", "holdout"],
        default=None,
        help="Optionally evaluate each completed run on these splits.",
    )
    train_many_parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch immediately if one run fails.",
    )
    _add_runtime_override_args(train_many_parser)

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
        required=True,
        metavar="PATH=EXPR",
        help="Ray Tune search expression such as model.hidden_dim=choice(128,192,256) or training.lr=loguniform(1e-4,5e-3).",
    )
    tune_parser.add_argument("--num-samples", type=int, default=8, help="Number of Ray Tune samples to draw.")
    tune_parser.add_argument("--search-alg", choices=["optuna"], default="optuna", help="Search algorithm to use with Ray Tune.")
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
    tune_parser.add_argument(
        "--max-concurrent-trials",
        type=int,
        default=1,
        help="Maximum number of concurrently running Ray Tune trials. Keep this at 1 for local MPS runs.",
    )
    _add_runtime_override_args(tune_parser)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a saved checkpoint on a split.")
    eval_parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    eval_parser.add_argument("--split", required=True, choices=["train", "val", "holdout"])
    eval_parser.add_argument("--output-dir", default=None, help="Optional evaluation output directory.")
    eval_parser.add_argument("--device", default="auto", help="Device for evaluation: auto, cpu, mps, cuda.")

    submit_parser = subparsers.add_parser("submit", help="Generate a Kaggle submission from a checkpoint.")
    submit_parser.add_argument("--checkpoint", required=True, help="Path to best.pt")
    submit_parser.add_argument("--output", default=None, help="Optional CSV output path.")
    submit_parser.add_argument("--device", default="auto", help="Device for submission: auto, cpu, mps, cuda.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "train":
        overrides = _build_overrides(args)
        if args.name_suffix:
            overrides["experiment.name"] = f"{Path(args.config).stem}_{args.name_suffix}"
        run_dir = train_experiment(args.config, overrides=overrides)
        print(f"Training complete. Outputs written to {run_dir}")
        return 0
    if args.command == "train-many":
        overrides = _build_overrides(args)
        if args.name_suffix:
            overrides["experiment.name_suffix"] = args.name_suffix
        batch_dir = train_many_experiments(
            configs=args.configs,
            seeds=args.seeds,
            overrides=overrides,
            evaluate_splits=args.evaluate_splits,
            fail_fast=args.fail_fast,
        )
        print(f"Batch training complete. Outputs written to {batch_dir}")
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
            search_specs=args.search,
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
        )
        print(f"Ray Tune search complete. Outputs written to {tune_dir}")
        return 0
    if args.command == "evaluate":
        eval_dir = evaluate_checkpoint(args.checkpoint, args.split, args.output_dir, device_arg=args.device)
        print(f"Evaluation complete. Outputs written to {eval_dir}")
        return 0
    if args.command == "submit":
        submission_path = submit_checkpoint(args.checkpoint, args.output, device_arg=args.device)
        print(f"Submission written to {submission_path}")
        return 0
    parser.error(f"Unknown command {args.command!r}")
    return 2
