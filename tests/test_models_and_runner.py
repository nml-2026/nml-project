from __future__ import annotations

from pathlib import Path
import json
import os
import tempfile
import unittest

import torch
import pandas as pd
from torch.utils.data import DataLoader

from nba_forecasting.config import RunConfig, apply_overrides, parse_override_value
from nba_forecasting.data import (
    NotebookRandomWindowSampler,
    NotebookSequenceDataset,
    SequenceWindowDataset,
    collect_sequence_paths,
    compute_normalizer,
)
from nba_forecasting.metrics import normalized_mse_loss
from nba_forecasting.models import build_model
from nba_forecasting.runner import cross_validate_experiment, repeat_trials_experiment, submit_checkpoint, train_experiment, train_many_experiments

from tests.helpers import TEAM_VALUES, IS_PLAYER, write_synthetic_dataset


def make_config(
    model_name: str,
    data_dir: str = "data",
    use_velocity: bool = True,
    dynamic_edges: bool = True,
    use_semantic_flags: bool = True,
    decoder_type: str = "autoregressive",
    use_relation_types: bool = False,
    use_entity_embeddings: bool = False,
) -> RunConfig:
    return RunConfig.from_dict(
        {
            "experiment": {
                "name": model_name,
                "model_name": model_name,
                "data_dir": data_dir,
                "output_root": "runs",
                "split_path": "artifacts/splits/default_split.json",
                "device": "cpu",
                "context_size": 8,
                "horizon_size": 12,
                "num_entities": 11,
                "checkpoint_metric": "ade",
                "seed": 0,
            },
            "features": {"use_velocity": use_velocity},
            "model": {
                "hidden_dim": 64,
                "message_dim": 64,
                "num_layers": 2,
                "dropout": 0.0,
                "k_neighbors": 3,
                "use_semantic_flags": use_semantic_flags,
                "dynamic_edges": dynamic_edges,
                "use_relation_types": use_relation_types,
                "use_entity_embeddings": use_entity_embeddings,
                "entity_embedding_dim": 8,
                "decoder_type": decoder_type,
                "direct_decoder_heads": 2,
                "temporal_ssm_layers": 2,
                "social_ssm_layers": 1,
                "decoder_ssm_layers": 2,
                "ssm_conv_kernel": 3,
                "separate_ball_head": False,
            },
                "training": {
                    "epochs": 2,
                    "batch_size": 4,
                    "lr": 0.01,
                    "weight_decay": 0.0,
                "grad_clip": 1.0,
                "num_workers": 0,
                "patience": 2,
                "log_every": 10,
                "teacher_forcing_prob": 0.0,
                    "scheduled_sampling_final_prob": 0.0,
                    "ema_decay": 0.0,
                    "swa_start_pct": 0.0,
                    "train_window_sampling": "all_windows",
                    "val_window_sampling": "all_windows",
                },
                "evaluation": {"num_examples": 2},
            }
        )


def make_context(batch_size: int, use_velocity: bool) -> torch.Tensor:
    time = torch.arange(8, dtype=torch.float32).view(1, 8, 1, 1)
    entity = torch.arange(11, dtype=torch.float32).view(1, 1, 11, 1)
    x = 0.2 * time + entity
    y = -0.1 * time + 0.05 * entity
    positions = torch.cat([x, y], dim=-1).expand(batch_size, -1, -1, -1)
    static = torch.stack([IS_PLAYER, TEAM_VALUES], dim=-1).view(1, 1, 11, 2).expand(batch_size, 8, -1, -1)
    if not use_velocity:
        return torch.cat([positions, static], dim=-1)
    velocity = torch.zeros_like(positions)
    velocity[:, 1:] = positions[:, 1:] - positions[:, :-1]
    return torch.cat([positions, velocity, static], dim=-1)


class ModelAndRunnerTests(unittest.TestCase):
    def test_override_helpers(self) -> None:
        config = make_config("e1_joint_gru", use_velocity=True)
        overrides = {
            "experiment.device": "cpu",
            "training.epochs": 7,
            "features.use_velocity": False,
        }
        updated = apply_overrides(config, overrides)
        self.assertEqual(updated.experiment.device, "cpu")
        self.assertEqual(updated.training.epochs, 7)
        self.assertFalse(updated.features.use_velocity)
        self.assertEqual(parse_override_value("true"), True)
        self.assertEqual(parse_override_value("13"), 13)
        self.assertEqual(parse_override_value("1.25"), 1.25)
        self.assertEqual(parse_override_value("mps"), "mps")

    def test_all_models_return_expected_shapes(self) -> None:
        for (
            model_name,
            use_velocity,
            dynamic_edges,
            use_semantic_flags,
            decoder_type,
            use_relation_types,
            use_entity_embeddings,
        ) in [
            ("e0_reference", False, True, True, "autoregressive", False, False),
            ("e0_notebook_exact", False, True, True, "autoregressive", False, False),
            ("e1_joint_gru", True, True, True, "autoregressive", False, False),
            ("e1_joint_autoreg_gru", True, True, True, "autoregressive", False, False),
            ("e1_social_ssm", True, True, True, "autoregressive", False, True),
            ("e2_static_graph", True, False, True, "autoregressive", True, True),
            ("e3_dynamic_graph", True, True, True, "autoregressive", False, False),
            ("e3_dynamic_graph", True, False, True, "autoregressive", True, True),
            ("e3_dynamic_graph", True, True, False, "autoregressive", False, False),
            ("e3_dynamic_graph", True, True, True, "direct", True, True),
            ("e3_graph_ssm", True, True, True, "autoregressive", True, True),
        ]:
            config = make_config(
                model_name=model_name,
                use_velocity=use_velocity,
                dynamic_edges=dynamic_edges,
                use_semantic_flags=use_semantic_flags,
                decoder_type=decoder_type,
                use_relation_types=use_relation_types,
                use_entity_embeddings=use_entity_embeddings,
            )
            if model_name in {"e3_dynamic_graph", "e3_graph_ssm"} and use_semantic_flags:
                config.model.message_passing_type = "gat"
                config.model.separate_ball_head = True
            model = build_model(config)
            context = make_context(batch_size=2, use_velocity=use_velocity)
            future = torch.randn(2, 12, 11, 2)
            output = model(
                context,
                future_xy=future,
                teacher_forcing_prob=0.5,
                scheduled_sampling_prob=0.25,
            )
            self.assertEqual(tuple(output.shape), (2, 12, 11, 2))
            self.assertTrue(torch.isfinite(output).all())

    def test_joint_gru_can_overfit_tiny_subset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = write_synthetic_dataset(Path(tmp_dir), train_lengths=[20, 20, 20])
            config = make_config("e1_joint_gru", data_dir=str(data_dir), use_velocity=True)
            config.model.hidden_dim = 128
            paths = collect_sequence_paths(data_dir / "train")
            normalizer = compute_normalizer(paths.values())
            dataset = SequenceWindowDataset(
                sequence_paths=[paths[0]],
                normalizer=normalizer,
                context_size=8,
                horizon_size=12,
                use_velocity=True,
            )
            batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
            model = build_model(config)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            final_loss = None
            for _ in range(300):
                prediction = model(batch["context"])
                loss = normalized_mse_loss(prediction, batch["future_xy"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                final_loss = float(loss.item())
            self.assertIsNotNone(final_loss)
            self.assertLess(final_loss, 1e-2)

    def test_notebook_sampler_yields_one_window_per_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = write_synthetic_dataset(Path(tmp_dir), train_lengths=[20, 22, 24])
            paths = collect_sequence_paths(data_dir / "train")
            normalizer = compute_normalizer(paths.values())
            dataset = NotebookSequenceDataset(
                sequence_paths=paths.values(),
                normalizer=normalizer,
                context_size=8,
                horizon_size=12,
                use_velocity=False,
            )
            sampler = NotebookRandomWindowSampler(dataset.max_start, seed=3, shuffle=True)
            indices = list(iter(sampler))
            self.assertEqual(len(indices), len(dataset))
            self.assertEqual(len({sample_idx for sample_idx, _ in indices}), len(dataset))
            batch = dataset[indices[0]]
            self.assertEqual(tuple(batch["context"].shape), (8, 11, 4))
            self.assertEqual(tuple(batch["future_xy"].shape), (12, 11, 2))

    def test_non_notebook_model_can_use_notebook_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = write_synthetic_dataset(root, train_lengths=[20, 22, 24, 26, 28, 30], test_count=2)
            config_path = root / "joint_gru_notebook_sampling.toml"
            config_path.write_text(
                f"""
[experiment]
name = "joint_gru_notebook_sampling"
model_name = "e1_joint_gru"
seed = 0
data_dir = "{data_dir}"
output_root = "{root / 'runs'}"
split_path = "{root / 'artifacts' / 'splits' / 'default.json'}"
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
temporal_ssm_layers = 2
social_ssm_layers = 1
decoder_ssm_layers = 2
ssm_conv_kernel = 3
separate_ball_head = false

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
train_window_sampling = "random_window_per_sequence"
val_window_sampling = "random_window_per_sequence"

[evaluation]
num_examples = 1
"""
            )
            run_dir = train_experiment(str(config_path))
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config"]["training"]["train_window_sampling"], "random_window_per_sequence")

    def test_submit_checkpoint_matches_template_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = write_synthetic_dataset(root, train_lengths=[20, 22, 24], test_count=2)
            config = make_config("e1_joint_gru", data_dir=str(data_dir), use_velocity=True)
            model = build_model(config)
            checkpoint_path = root / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config.to_dict(),
                    "split": {"train": [0, 1], "val": [2], "holdout": []},
                    "normalizer": {"mu": [0.0, 0.0], "sigma": [1.0, 1.0]},
                    "best_metrics": {"ade": 0.0, "mse": 0.0, "rmse": 0.0},
                    "epoch": 0,
                },
                checkpoint_path,
            )
            output_path = submit_checkpoint(checkpoint_path, root / "submission.csv")
            frame = pd.read_csv(output_path)
            self.assertEqual(frame.shape, (2, 265))
            self.assertEqual(frame.columns[0], "id")
            self.assertEqual(frame["id"].tolist(), [0, 1])

    def test_train_many_runs_multiple_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = write_synthetic_dataset(root, train_lengths=[20, 22, 24, 26, 28, 30], test_count=2)
            config_a = root / "a.toml"
            config_b = root / "b.toml"
            base = f"""
[experiment]
seed = 0
data_dir = "{data_dir}"
output_root = "{root / 'runs'}"
split_path = "{root / 'artifacts' / 'splits' / 'default.json'}"
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

[evaluation]
num_examples = 1
"""
            config_a.write_text(
                base.replace('[experiment]', '[experiment]\nname = "job_a"\nmodel_name = "e0_reference"', 1)
            )
            config_b.write_text(
                base.replace('[experiment]', '[experiment]\nname = "job_b"\nmodel_name = "e1_joint_gru"', 1)
            )
            batch_dir = train_many_experiments(
                configs=[str(config_a), str(config_b)],
                seeds=[0, 1],
                overrides={"experiment.device": "cpu"},
                evaluate_splits=["val"],
            )
            summary = json.loads((batch_dir / "batch_summary.json").read_text())
            self.assertEqual(len(summary["jobs"]), 4)
            self.assertTrue(all(job["status"] == "completed" for job in summary["jobs"]))

    def test_config_relative_paths_do_not_depend_on_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            write_synthetic_dataset(root, train_lengths=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38], test_count=2)
            config_dir = root / "configs"
            config_dir.mkdir()
            config_path = config_dir / "relative.toml"
            config_path.write_text(
                """
[experiment]
name = "relative_paths"
model_name = "e1_joint_gru"
seed = 0
data_dir = "data"
output_root = "runs"
split_path = "artifacts/splits/default.json"
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
            other_dir = root / "elsewhere"
            other_dir.mkdir()
            previous_cwd = Path.cwd()
            try:
                os.chdir(other_dir)
                run_dir = train_experiment(str(config_path))
            finally:
                os.chdir(previous_cwd)
            self.assertTrue(str(run_dir).startswith(str((root / "runs").resolve())))
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["config"]["experiment"]["data_dir"], str((root / "data").resolve()))
            self.assertEqual(
                checkpoint["config"]["experiment"]["split_path"],
                str((root / "artifacts" / "splits" / "default.json").resolve()),
            )

    def test_train_experiment_supports_ema_and_swa_checkpointing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = write_synthetic_dataset(root, train_lengths=[20, 22, 24, 26, 28, 30], test_count=2)
            config_path = root / "direct.toml"
            config_path.write_text(
                f"""
[experiment]
name = "direct"
model_name = "e3_dynamic_graph"
seed = 0
data_dir = "{data_dir}"
output_root = "{root / 'runs'}"
split_path = "{root / 'artifacts' / 'splits' / 'default.json'}"
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
use_relation_types = true
use_entity_embeddings = true
entity_embedding_dim = 8
decoder_type = "direct"
direct_decoder_heads = 2
message_passing_type = "mpnn"
separate_ball_head = true

[training]
epochs = 2
batch_size = 4
lr = 0.01
weight_decay = 0.0
grad_clip = 1.0
num_workers = 0
patience = 2
log_every = 10
teacher_forcing_prob = 0.0
scheduled_sampling_final_prob = 0.0
ema_decay = 0.9
swa_start_pct = 0.5

[evaluation]
num_examples = 1
"""
            )
            run_dir = train_experiment(str(config_path))
            checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
            self.assertIn(checkpoint["eval_model_kind"], {"ema", "swa"})
            self.assertEqual(checkpoint["config"]["model"]["decoder_type"], "direct")

    def test_repeat_trials_writes_aggregate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = write_synthetic_dataset(root, train_lengths=[20, 22, 24, 26, 28, 30], test_count=2)
            config_path = root / "repeat.toml"
            config_path.write_text(
                f"""
[experiment]
name = "repeatable"
model_name = "e1_joint_gru"
seed = 3
data_dir = "{data_dir}"
output_root = "{root / 'runs'}"
split_path = "{root / 'artifacts' / 'splits' / 'default.json'}"
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
max_train_batches = 2
max_val_batches = 1

[evaluation]
num_examples = 1
"""
            )
            repeat_dir = repeat_trials_experiment(str(config_path), repeats=2, seed_step=5)
            summary = json.loads((repeat_dir / "repeat_summary.json").read_text())
            self.assertEqual(summary["protocol"], "repeat_trials")
            self.assertEqual(len(summary["trials"]), 2)
            self.assertIn("val_ade_mean", summary["aggregate"])

    def test_cross_validate_writes_fold_aggregate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            data_dir = write_synthetic_dataset(root, train_lengths=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38], test_count=2)
            config_path = root / "cv.toml"
            config_path.write_text(
                f"""
[experiment]
name = "cv_model"
model_name = "e1_joint_gru"
seed = 0
data_dir = "{data_dir}"
output_root = "{root / 'runs'}"
split_path = "{root / 'artifacts' / 'splits' / 'default.json'}"
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
max_train_batches = 2
max_val_batches = 1

[evaluation]
num_examples = 1
"""
            )
            cv_dir = cross_validate_experiment(str(config_path), num_folds=3)
            summary = json.loads((cv_dir / "cv_summary.json").read_text())
            self.assertEqual(summary["protocol"], "cross_validation")
            self.assertEqual(len(summary["folds"]), 3)
            self.assertIn("val_ade_mean", summary["aggregate"])


if __name__ == "__main__":
    unittest.main()
