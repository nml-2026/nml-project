from __future__ import annotations

import csv
import json
from pathlib import Path

from nba_forecasting.ray_tune import (
    _build_final_tune_report,
    _last_epoch_val_metric,
    tune_metric_name,
)


def test_tune_metric_name_uses_best_validation_score() -> None:
    assert tune_metric_name("ade") == "best_val_ade"


def test_build_final_tune_report_prefers_best_checkpoint_eval(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    eval_dir = tmp_path / "eval"
    run_dir.mkdir()
    eval_dir.mkdir()
    (eval_dir / "metrics.json").write_text(json.dumps({"ade": 5.612, "ade_l2": 1.41}))
    (run_dir / "train_summary.json").write_text(json.dumps({"best_epoch": 84, "best_metric": 5.612}))
    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val_ade", "train_loss"])
        writer.writeheader()
        writer.writerow({"epoch": "84", "val_ade": "5.612", "train_loss": "0.2"})
        writer.writerow({"epoch": "100", "val_ade": "5.737", "train_loss": "0.19"})

    report = _build_final_tune_report(
        run_dir=run_dir,
        eval_dir=eval_dir,
        checkpoint_metric="ade",
        metric_name="best_val_ade",
        trial_config_arg="configs/tune_mart.toml",
        export_overrides={"training.lr": 0.0006},
    )

    assert report["best_val_ade"] == 5.612
    assert report["val_ade"] == 5.612
    assert report["val_ade_last_epoch"] == 5.737
    assert report["best_epoch"] == 84


def test_last_epoch_val_metric_reads_history_tail(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with (run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "val_ade"])
        writer.writeheader()
        writer.writerow({"epoch": "1", "val_ade": "8.0"})
        writer.writerow({"epoch": "2", "val_ade": "6.5"})

    assert _last_epoch_val_metric(run_dir, "ade") == 6.5
