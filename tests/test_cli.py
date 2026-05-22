from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nba_forecasting.cli import main
from nba_forecasting.ray_tune import SEARCH_PROFILES
from tests.helpers import write_synthetic_dataset


def write_cli_config(root: Path, data_dir: Path, name: str = "cli_model") -> Path:
    config_path = root / f"{name}.toml"
    config_path.write_text(
        f"""
[experiment]
name = "{name}"
model_name = "e1_joint_gru"
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
hidden_dim = 32
message_dim = 32
num_layers = 1
dropout = 0.0
k_neighbors = 2
use_semantic_flags = true
dynamic_edges = true
use_relation_types = false
use_entity_embeddings = false
entity_embedding_dim = 8
decoder_type = "autoregressive"
direct_decoder_heads = 2

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


class CliTests(unittest.TestCase):
    def test_module_cli_train_evaluate_and_submit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = write_synthetic_dataset(root, train_lengths=[20, 22, 24, 26, 28, 30], test_count=2)
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
            self.assertEqual(code, 0)
            prefix = "Training complete. Outputs written to "
            run_dir = Path(next(line.removeprefix(prefix) for line in train_output.getvalue().splitlines() if line.startswith(prefix)))
            checkpoint = run_dir / "best.pt"
            self.assertTrue(checkpoint.exists())

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
            self.assertEqual(code, 0)
            self.assertTrue((eval_dir / "metrics.json").exists())

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
            self.assertEqual(code, 0)
            self.assertTrue(submission.exists())

    def test_train_many_cli_dispatches_cluster_arguments(self) -> None:
        with patch("nba_forecasting.cli.train_many_experiments") as train_many:
            train_many.return_value = Path("/tmp/batch")
            code = main(
                [
                    "train-many",
                    "--configs",
                    "e0_reference",
                    "e1_joint_gru",
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
            self.assertEqual(code, 0)
            train_many.assert_called_once()
            _, kwargs = train_many.call_args
            self.assertEqual(kwargs["configs"], ["e0_reference", "e1_joint_gru"])
            self.assertEqual(kwargs["seeds"], [0, 1])
            self.assertEqual(kwargs["evaluate_splits"], ["val", "holdout"])
            self.assertEqual(kwargs["overrides"]["experiment.device"], "cpu")
            self.assertEqual(kwargs["overrides"]["training.epochs"], 2)
            self.assertEqual(kwargs["overrides"]["training.batch_size"], 8)
            self.assertEqual(kwargs["overrides"]["experiment.output_root"], "/tmp/runs")

    def test_tune_cli_dispatches_cluster_arguments(self) -> None:
        with patch("nba_forecasting.cli.run_ray_tune_search") as tune:
            tune.return_value = Path("/tmp/tune")
            code = main(
                [
                    "tune",
                    "--config",
                    "e3_dynamic_graph_direct",
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
                    "model.hidden_dim=choice(128,192)",
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
            self.assertEqual(code, 0)
            tune.assert_called_once()
            _, kwargs = tune.call_args
            self.assertEqual(kwargs["config_arg"], "e3_dynamic_graph_direct")
            self.assertEqual(kwargs["search_specs"], ["model.hidden_dim=choice(128,192)", "training.lr=loguniform(1e-4,5e-3)"])
            self.assertEqual(kwargs["num_samples"], 2)
            self.assertEqual(kwargs["scheduler_name"], "asha")
            self.assertEqual(kwargs["search_algorithm"], "optuna")
            self.assertEqual(kwargs["cpus_per_trial"], 8)
            self.assertEqual(kwargs["gpus_per_trial"], 1.0)
            self.assertEqual(kwargs["max_concurrent_trials"], 2)
            self.assertEqual(kwargs["overrides"]["experiment.device"], "cpu")
            self.assertEqual(kwargs["overrides"]["experiment.output_root"], "/tmp/runs")
            self.assertEqual(kwargs["overrides"]["training.epochs"], 1)
            self.assertEqual(kwargs["overrides"]["training.max_train_batches"], 2)
            self.assertEqual(kwargs["overrides"]["training.max_val_batches"], 1)

    def test_tune_cli_expands_search_profile(self) -> None:
        with patch("nba_forecasting.cli.run_ray_tune_search") as tune:
            tune.return_value = Path("/tmp/tune")
            code = main(
                [
                    "tune",
                    "--config",
                    "e3_graph_ssm",
                    "--search-profile",
                    "graph_ssm",
                    "--search",
                    "training.lr=loguniform(1e-5,1e-3)",
                    "--num-samples",
                    "2",
                ]
            )
            self.assertEqual(code, 0)
            _, kwargs = tune.call_args
            self.assertEqual(kwargs["search_specs"][:-1], SEARCH_PROFILES["graph_ssm"])
            self.assertEqual(kwargs["search_specs"][-1], "training.lr=loguniform(1e-5,1e-3)")

    def test_tune_cli_expands_graph_transformer_profile(self) -> None:
        with patch("nba_forecasting.cli.run_ray_tune_search") as tune:
            tune.return_value = Path("/tmp/tune")
            code = main(
                [
                    "tune",
                    "--config",
                    "e5_graph_transformer",
                    "--search-profile",
                    "graph_transformer",
                    "--num-samples",
                    "2",
                ]
            )
            self.assertEqual(code, 0)
            _, kwargs = tune.call_args
            self.assertEqual(kwargs["search_specs"], SEARCH_PROFILES["graph_transformer"])

    def test_fixed_message_graph_profile_leaves_message_type_to_config(self) -> None:
        specs = SEARCH_PROFILES["graph_fixed_message"]
        self.assertTrue(any(spec.startswith("model.message_dim=") for spec in specs))
        self.assertTrue(any(spec.startswith("model.k_neighbors=") for spec in specs))
        self.assertFalse(any(spec.startswith("model.message_passing_type=") for spec in specs))

    def test_result_summary_and_best_config_export_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
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
                            "config": {"model.hidden_dim": 192, "training.lr": 0.0005},
                            "metrics": {"val_ade": 1.25, "run_dir": str(run_dir)},
                        },
                        "trials": [],
                    }
                )
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["summarize-results", "--root", str(root), "--format", "markdown"])
            self.assertEqual(code, 0)
            self.assertIn("tuned", output.getvalue())
            self.assertIn("1.2500", output.getvalue())
            self.assertIn("123456", output.getvalue())

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = main(["estimate-runtime", "--root", str(root), "--target-samples", "4", "--target-epochs", "35"])
            self.assertEqual(code, 0)
            self.assertIn("tuned", output.getvalue())
            self.assertIn("1h 10m", output.getvalue())

            exported = root / "exported.toml"
            code = main(["export-best-config", "--summary", str(summary_path), "--output", str(exported)])
            self.assertEqual(code, 0)
            exported_text = exported.read_text()
            self.assertIn('name = "tuned"', exported_text)
            self.assertIn("hidden_dim = 192", exported_text)
            self.assertIn("lr = 0.0005", exported_text)


if __name__ == "__main__":
    unittest.main()
