from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from nba_forecasting.cli import main
from nba_forecasting.ray_tune import SEARCH_PROFILES
from tests.helpers import write_synthetic_dataset


def write_cli_config(root: Path, data_dir: Path, name: str = "cli_model") -> Path:
    config_path = root / f"{name}.toml"
    config_path.write_text(
        f"""
[experiment]
name = "{name}"
seed = 0
data_dir = "{data_dir}"
output_root = "{root / "runs"}"
split_path = "{root / "artifacts" / "splits" / "default.json"}"
device = "cpu"
context_size = 8
horizon_size = 12
num_entities = 11
checkpoint_metric = "ade"

[features]
use_velocity = false

[model]
family = "baseline"

[model.common]
hidden_dim = 32
num_layers = 1
dropout = 0.0

[training]
epochs = 1
batch_size = 4
lr = 0.01
weight_decay = 0.0
grad_clip = 1.0
num_workers = 0
patience = 1
log_every = 10
teacher_forcing_prob = 0.0
scheduled_sampling_final_prob = 0.0
ema_decay = 0.0
swa_start_pct = 0.0
max_train_batches = 1
max_val_batches = 1

[evaluation]
num_examples = 1
"""
    )
    return config_path


def write_mdn_cli_config(root: Path, data_dir: Path, name: str = "cli_mdn") -> Path:
    config_path = root / f"{name}.toml"
    config_path.write_text(
        f"""
[experiment]
name = "{name}"
seed = 0
data_dir = "{data_dir}"
output_root = "{root / "runs"}"
split_path = "{root / "artifacts" / "splits" / "default.json"}"
device = "cpu"
context_size = 8
horizon_size = 12
num_entities = 11
checkpoint_metric = "ade"

[features]
use_velocity = true

[model]
family = "graph_transformer"

[model.common]
hidden_dim = 32
dropout = 0.0

[model.graph]
k_neighbors = 2

[model.graph_transformer]
transformer_heads = 4
spatial_layers = 1
temporal_layers = 1
decoder_layers = 1

[model.head]
head_type = "mdn"
mdn_components = 3
mdn_point_estimate = "mean"

[training]
epochs = 1
batch_size = 4
lr = 0.01
weight_decay = 0.0
grad_clip = 1.0
num_workers = 0
patience = 1
log_every = 10

[evaluation]
num_examples = 1
"""
    )
    return config_path


def test_mdn_cli_train_evaluate_and_submit(tmp_path: Path) -> None:
    root = tmp_path
    data_dir = write_synthetic_dataset(root, train_lengths=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38], test_count=2)
    config_path = write_mdn_cli_config(root, data_dir)

    train_output = io.StringIO()
    with contextlib.redirect_stdout(train_output):
        code = main(["train", "--config", str(config_path), "--device", "cpu", "--epochs", "1"])
    assert code == 0
    prefix = "Training complete. Outputs written to "
    run_dir = Path(next(line.removeprefix(prefix) for line in train_output.getvalue().splitlines() if line.startswith(prefix)))
    checkpoint = run_dir / "best.pt"
    assert checkpoint.exists()

    eval_dir = root / "eval_mdn"
    with contextlib.redirect_stdout(io.StringIO()):
        code = main(["evaluate", "--checkpoint", str(checkpoint), "--split", "val", "--output-dir", str(eval_dir), "--device", "cpu"])
    assert code == 0
    metrics = json.loads((eval_dir / "metrics.json").read_text())
    for key in ("ade", "ade_l2", "fde_l2", "min_ade_k", "min_fde_k", "mdn_components"):
        assert key in metrics
    assert metrics["mdn_components"] == 3.0
    assert metrics["min_ade_k"] >= 0.0
    assert metrics["min_fde_k"] >= 0.0
    assert len(json.loads((eval_dir / "per_step_error.json").read_text())["ade_l2_per_step"]) == 12

    submission = root / "mdn_submission.csv"
    with contextlib.redirect_stdout(io.StringIO()):
        code = main(["submit", "--checkpoint", str(checkpoint), "--output", str(submission), "--device", "cpu", "--tta", "xy"])
    assert code == 0
    assert submission.exists()


def test_module_cli_train_evaluate_and_submit(tmp_path: Path) -> None:
    root = tmp_path
    data_dir = write_synthetic_dataset(root, train_lengths=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38], test_count=2)
    config_path = write_cli_config(root, data_dir)

    train_output = io.StringIO()
    with contextlib.redirect_stdout(train_output):
        code = main(
            [
                "train",
                "--config",
                str(config_path),
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--set",
                "training.max_train_batches=1",
                "--set",
                "training.max_val_batches=1",
            ]
        )
    assert code == 0
    prefix = "Training complete. Outputs written to "
    run_dir = Path(next(line.removeprefix(prefix) for line in train_output.getvalue().splitlines() if line.startswith(prefix)))
    checkpoint = run_dir / "best.pt"
    assert checkpoint.exists()

    eval_dir = root / "eval_cli"
    code = main(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--split",
            "val",
            "--output-dir",
            str(eval_dir),
            "--device",
            "cpu",
        ]
    )
    assert code == 0
    assert (eval_dir / "metrics.json").exists()
    eval_metrics = json.loads((eval_dir / "metrics.json").read_text())
    assert "ade_l2" in eval_metrics
    assert "fde_l2" in eval_metrics
    assert (eval_dir / "per_step_error.json").exists()
    per_step = json.loads((eval_dir / "per_step_error.json").read_text())
    assert len(per_step["ade_l2_per_step"]) == 12

    submission = root / "submission.csv"
    code = main(
        [
            "submit",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(submission),
            "--device",
            "cpu",
        ],
    )
    assert code == 0
    assert submission.exists()

    tta_eval_dir = root / "eval_cli_tta"
    code = main(
        [
            "evaluate",
            "--checkpoint",
            str(checkpoint),
            "--split",
            "val",
            "--output-dir",
            str(tta_eval_dir),
            "--device",
            "cpu",
            "--tta",
            "xy",
        ]
    )
    assert code == 0
    tta_metrics = json.loads((tta_eval_dir / "metrics.json").read_text())
    assert tta_metrics["tta_group"] == "xy"
    tta_per_step = json.loads((tta_eval_dir / "per_step_error.json").read_text())
    assert len(tta_per_step["ade_l2_per_step"]) == 12

    tta_submission = root / "submission_tta.csv"
    code = main(
        [
            "submit",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tta_submission),
            "--device",
            "cpu",
            "--tta",
            "xy",
        ],
    )
    assert code == 0
    assert tta_submission.exists()

    ens_eval_dir = root / "eval_cli_ensemble"
    code = main(
        [
            "ensemble-evaluate",
            "--checkpoints",
            str(checkpoint),
            "--split",
            "val",
            "--output-dir",
            str(ens_eval_dir),
            "--device",
            "cpu",
            "--tta",
            "xy",
        ]
    )
    assert code == 0
    ens_metrics = json.loads((ens_eval_dir / "metrics.json").read_text())
    assert ens_metrics["tta_group"] == "xy"
    assert ens_metrics["ensemble_n"] == 1
    assert len(json.loads((ens_eval_dir / "per_step_error.json").read_text())["ade_l2_per_step"]) == 12

    ens_submission = root / "ensemble_submission.csv"
    code = main(
        [
            "ensemble-submit",
            "--checkpoints",
            str(checkpoint),
            "--output",
            str(ens_submission),
            "--device",
            "cpu",
            "--tta",
            "xy",
        ]
    )
    assert code == 0
    assert ens_submission.exists()


def test_batch_train_rejects_resume_checkpoint() -> None:
    with pytest.raises(ValueError, match="--resume-checkpoint is only supported for single-config training"):
        main(
            [
                "train",
                "--config",
                "baseline",
                "dynamic_graph_direct",
                "--resume-checkpoint",
                "/tmp/last.pt",
            ]
        )


def test_train_batch_cli_dispatches_cluster_arguments() -> None:
    with patch("nba_forecasting.cli.train_many_experiments") as train_many:
        train_many.return_value = Path("/tmp/batch")
        code = main(
            [
                "train",
                "--config",
                "baseline",
                "dynamic_graph_direct",
                "--seeds",
                "0",
                "1",
                "--device",
                "cpu",
                "--epochs",
                "2",
                "--batch-size",
                "8",
                "--output-root",
                "/tmp/runs",
                "--evaluate-splits",
                "val",
                "holdout",
            ]
        )
        assert code == 0
        train_many.assert_called_once()
        _, kwargs = train_many.call_args
        assert kwargs["configs"] == ["baseline", "dynamic_graph_direct"]
        assert kwargs["seeds"] == [0, 1]
        assert kwargs["evaluate_splits"] == ["val", "holdout"]
        assert kwargs["overrides"]["experiment.device"] == "cpu"
        assert kwargs["overrides"]["training.epochs"] == 2
        assert kwargs["overrides"]["training.batch_size"] == 8
        assert kwargs["overrides"]["experiment.output_root"] == "/tmp/runs"


def test_tune_cli_dispatches_cluster_arguments() -> None:
    with patch("nba_forecasting.cli.run_ray_tune_search") as tune:
        tune.return_value = Path("/tmp/tune")
        code = main(
            [
                "tune",
                "--config",
                "dynamic_graph_direct",
                "--device",
                "cpu",
                "--output-root",
                "/tmp/runs",
                "--epochs",
                "1",
                "--set",
                "training.max_train_batches=2",
                "--set",
                "training.max_val_batches=1",
                "--search",
                "model.common.hidden_dim=choice(128,192)",
                "--search",
                "training.lr=loguniform(1e-4,5e-3)",
                "--num-samples",
                "2",
                "--search-alg",
                "optuna",
                "--scheduler",
                "asha",
                "--cpus-per-trial",
                "8",
                "--gpus-per-trial",
                "1",
                "--max-concurrent-trials",
                "2",
            ]
        )
        assert code == 0
        tune.assert_called_once()
        _, kwargs = tune.call_args
        assert kwargs["config_arg"] == "dynamic_graph_direct"
        assert kwargs["search_specs"] == ["model.common.hidden_dim=choice(128,192)", "training.lr=loguniform(1e-4,5e-3)"]
        assert kwargs["num_samples"] == 2
        assert kwargs["scheduler_name"] == "asha"
        assert kwargs["search_algorithm"] == "optuna"
        assert kwargs["cpus_per_trial"] == 8
        assert kwargs["gpus_per_trial"] == 1.0
        assert kwargs["max_concurrent_trials"] == 2
        assert kwargs["overrides"]["experiment.device"] == "cpu"
        assert kwargs["overrides"]["experiment.output_root"] == "/tmp/runs"
        assert kwargs["overrides"]["training.epochs"] == 1
        assert kwargs["overrides"]["training.max_train_batches"] == 2
        assert kwargs["overrides"]["training.max_val_batches"] == 1


def test_tune_cli_expands_search_profile() -> None:
    with patch("nba_forecasting.cli.run_ray_tune_search") as tune:
        tune.return_value = Path("/tmp/tune")
        code = main(
            [
                "tune",
                "--config",
                "graph_transformer",
                "--search-profile",
                "graph_transformer",
                "--search",
                "training.lr=loguniform(1e-5,1e-3)",
                "--num-samples",
                "2",
            ]
        )
        assert code == 0
        _, kwargs = tune.call_args
        assert kwargs["search_specs"][:-1] == SEARCH_PROFILES["graph_transformer"]
        assert kwargs["search_specs"][-1] == "training.lr=loguniform(1e-5,1e-3)"


def test_tune_cli_expands_graph_transformer_profile() -> None:
    with patch("nba_forecasting.cli.run_ray_tune_search") as tune:
        tune.return_value = Path("/tmp/tune")
        code = main(
            [
                "tune",
                "--config",
                "graph_transformer",
                "--search-profile",
                "graph_transformer",
                "--num-samples",
                "2",
            ]
        )
        assert code == 0
        _, kwargs = tune.call_args
        assert kwargs["search_specs"] == SEARCH_PROFILES["graph_transformer"]


def test_tune_cli_expands_mart_profile() -> None:
    with patch("nba_forecasting.cli.run_ray_tune_search") as tune:
        tune.return_value = Path("/tmp/tune")
        code = main(
            [
                "tune",
                "--config",
                "mart",
                "--search-profile",
                "mart",
                "--num-samples",
                "2",
            ]
        )
        assert code == 0
        _, kwargs = tune.call_args
        assert kwargs["search_specs"] == SEARCH_PROFILES["mart"]


def test_doctor_cli_dispatches_expect_cuda() -> None:
    with patch("nba_forecasting.cli.run_doctor") as doctor:
        doctor.return_value = 0
        code = main(["doctor", "--expect-cuda", "--check-inductor", "--inductor-device", "cuda"])
        assert code == 0
        doctor.assert_called_once_with(expect_cuda=True, check_inductor=True, inductor_device="cuda")


def test_prepare_data_cli_dispatches_arguments() -> None:
    with patch("nba_forecasting.cli.prepare_remote_dataset") as prepare:
        prepare.return_value = {
            "data_dir": "/tmp/data",
            "train_consolidated": "/tmp/data/train_consolidated.pt",
            "test_consolidated": "/tmp/data/test_consolidated.pt",
            "split_path": "/tmp/artifacts/splits/default_split.json",
            "train_sequences": 8,
            "val_sequences": 1,
            "holdout_sequences": 1,
        }
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "prepare-data",
                    "--data-dir",
                    "/tmp/data",
                    "--split-path",
                    "/tmp/artifacts/splits/default_split.json",
                    "--seed",
                    "7",
                    "--overwrite",
                ]
            )
        assert code == 0
        prepare.assert_called_once_with(
            data_dir="/tmp/data",
            split_path="/tmp/artifacts/splits/default_split.json",
            seed=7,
            overwrite=True,
        )
        assert "train_consolidated: /tmp/data/train_consolidated.pt" in output.getvalue()


def test_result_summary_and_best_config_export_cli(tmp_path: Path) -> None:
    root = tmp_path
    config_path = write_cli_config(root, root / "data", name="tuned")
    run_dir = root / "123456" / "ray_tune" / "tuned" / "20260101" / "trial_runs" / "trial-a"
    run_dir.mkdir(parents=True)
    (run_dir / "best.pt").write_text("checkpoint")
    summary_path = root / "123456" / "ray_tune" / "tuned" / "20260101" / "ray_tune_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "config": str(config_path),
                "elapsed_s": 120.0,
                "num_samples": 2,
                "training_epochs": 2,
                "max_concurrent_trials": 2,
                "best_trial": {
                    "config": {"model.common.hidden_dim": 192, "training.lr": 0.0005},
                    "metrics": {"val_ade": 1.25, "run_dir": str(run_dir)},
                },
                "trials": [],
            }
        )
    )

    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = main(["summarize-results", "--root", str(root), "--format", "markdown"])
    assert code == 0
    assert "tuned" in output.getvalue()
    assert "1.2500" in output.getvalue()
    assert "123456" in output.getvalue()

    exported = root / "exported.toml"
    code = main(["export-best-config", "--summary", str(summary_path), "--output", str(exported)])
    assert code == 0
    exported_text = exported.read_text()
    assert 'name = "tuned"' in exported_text
    assert "hidden_dim = 192" in exported_text
    assert "lr = 0.0005" in exported_text
