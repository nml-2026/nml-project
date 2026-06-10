from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from nba_forecasting.config import load_config
from nba_forecasting.data import (
    Normalizer,
    NotebookRandomWindowSampler,
    NotebookSequenceDataset,
    SequenceWindowDataset,
    build_dataloaders,
    collect_sequence_paths,
    compute_normalizer,
)
from nba_forecasting.metrics import normalized_mse_loss
from nba_forecasting.models import build_model, resolve_training_loss_policy
from nba_forecasting.models.common import ForecastOutput, MixtureParams
from nba_forecasting.runner import train_experiment, train_many_experiments
from nba_forecasting.runner.recipe import ForecastTrainingRecipe
from nba_forecasting.runner.trainer import _build_lr_scheduler, build_optimizer
from tests.conftest import make_config
from tests.helpers import write_synthetic_dataset


def _loss_batch(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {"static": torch.zeros(batch_size, 11, 2)}


def test_multistep_lr_scheduler_decays_at_configured_milestones() -> None:
    config = make_config("baseline", use_velocity=False)
    config.training.lr = 0.01
    config.training.lr_scheduler = "multistep"
    config.training.lr_milestones = [1, 3]
    config.training.lr_gamma = 0.5
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    scheduler = _build_lr_scheduler(config, optimizer)
    assert scheduler is not None

    observed = []
    for _ in range(4):
        observed.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert observed == [0.01, 0.005, 0.005, 0.0025]


def test_build_optimizer_uses_adamw_backend() -> None:
    config = make_config("baseline", use_velocity=False)
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(config, model, torch.device("cpu"))
    assert isinstance(optimizer, torch.optim.AdamW)


def test_fused_adamw_falls_back_to_adamw_without_cuda() -> None:
    config = make_config("baseline", use_velocity=False)
    config.runtime.optimizer_backend = "adamw_fused"
    model = torch.nn.Linear(2, 2)
    optimizer = build_optimizer(config, model, torch.device("cpu"))
    assert isinstance(optimizer, torch.optim.AdamW)


def test_reference_model_can_overfit_tiny_subset(tmp_path: Path) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 20, 20])
    config = make_config("baseline", data_dir=str(data_dir), use_velocity=False)
    config.model.common.hidden_dim = 128
    paths = collect_sequence_paths(data_dir / "train")
    normalizer = compute_normalizer(paths.values())
    dataset = SequenceWindowDataset(
        sequence_paths=[paths[0]],
        normalizer=normalizer,
        context_size=8,
        horizon_size=12,
        use_velocity=False,
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
    assert final_loss is not None
    assert final_loss < 1e-2


def test_notebook_sampler_yields_one_window_per_sequence(tmp_path: Path) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24])
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
    assert len(indices) == len(dataset)
    assert len({sample_idx for sample_idx, _ in indices}) == len(dataset)
    batch = dataset[indices[0]]
    assert tuple(batch["context"].shape) == (8, 11, 4)
    assert tuple(batch["future_xy"].shape) == (12, 11, 2)


def test_sequence_dataset_can_expand_reflection_variants() -> None:
    normalizer = Normalizer(mu=torch.tensor([1.0, 2.0]), sigma=torch.tensor([2.0, 4.0]))
    seq = torch.tensor(
        [
            [[3.0, 6.0, 1.0, 0.0]],
            [[5.0, 10.0, 1.0, 0.0]],
            [[7.0, 14.0, 1.0, 0.0]],
        ]
    )
    dataset = SequenceWindowDataset(
        sequence_paths={0: seq},
        normalizer=normalizer,
        context_size=2,
        horizon_size=1,
        use_velocity=True,
        reflect_prob=1.0,
        reflect_x_prob=1.0,
        reflection_mode="expand",
    )

    identity = dataset[0]
    flip_y = dataset[1]
    flip_x = dataset[2]
    flip_xy = dataset[3]

    assert len(dataset) == 4
    torch.testing.assert_close(identity["context"][..., :2], torch.tensor([[[1.0, 1.0]], [[2.0, 2.0]]]))
    torch.testing.assert_close(identity["context"][1, :, 2:4], torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(flip_y["future_xy"], torch.tensor([[[3.0, -4.0]]]))
    torch.testing.assert_close(flip_x["future_xy"], torch.tensor([[[-4.0, 3.0]]]))
    torch.testing.assert_close(flip_xy["future_xy"], torch.tensor([[[-4.0, -4.0]]]))
    torch.testing.assert_close(flip_xy["future_xy_raw"], torch.tensor([[[-7.0, -14.0]]]))


def test_sample_loss_modes_point_components() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.loss.sample_loss_type = "softmin_ade"
    config.loss.sample_loss_space = "raw"
    config.loss.softmin_temperature = 2.0
    config.loss.point_loss_weight = 0.5
    recipe = ForecastTrainingRecipe(
        loss_config=config.loss,
        normalizer=Normalizer(mu=torch.tensor([10.0, 20.0]), sigma=torch.tensor([2.0, 4.0])),
        device=torch.device("cpu"),
        grad_clip=0.0,
        mixed_precision=False,
    )
    target = torch.zeros(1, 2, 1, 2)
    modes = torch.tensor([[[[[0.0, 0.0], [1.0, 0.0]]], [[[0.0, 0.0], [1.0, 0.0]]]]], requires_grad=True)
    point = modes.mean(dim=3)
    output = ForecastOutput(point=point, modes=modes)

    loss, components = recipe.compute_sample_loss(output, target, _loss_batch())
    loss.backward()

    assert torch.isfinite(loss)
    assert components["sample"] > 0.0
    assert components["point"] > 0.0
    assert modes.grad is not None


def test_raw_sample_loss_is_scaled_by_normalizer() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.loss.sample_loss_type = "hard_min_ade"
    config.loss.sample_loss_space = "raw"
    config.loss.raw_loss_scale_ft = 5.0
    recipe = ForecastTrainingRecipe(
        loss_config=config.loss,
        normalizer=Normalizer(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([3.0, 4.0])),
        device=torch.device("cpu"),
        grad_clip=0.0,
        mixed_precision=False,
    )
    target = torch.zeros(1, 1, 1, 2)
    modes = torch.tensor([[[[[1.0, 0.0]]]]], requires_grad=True)
    output = ForecastOutput(point=modes.squeeze(3), modes=modes)

    loss, components = recipe.compute_sample_loss(output, target, _loss_batch(), epoch=1)

    torch.testing.assert_close(loss, torch.tensor(0.6))
    torch.testing.assert_close(torch.tensor(components["sample"]), torch.tensor(0.6))


def test_mixture_nll_sample_loss_uses_distribution_params() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.loss.sample_loss_type = "mixture_nll"
    recipe = ForecastTrainingRecipe(
        loss_config=config.loss,
        normalizer=Normalizer(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([1.0, 1.0])),
        device=torch.device("cpu"),
        grad_clip=0.0,
        mixed_precision=False,
    )
    target = torch.zeros(1, 2, 1, 2)
    means = torch.zeros(1, 2, 1, 2, 2, requires_grad=True)
    log_sigma = torch.zeros_like(means)
    rho = torch.zeros(1, 2, 1, 2)
    log_pi = torch.log_softmax(torch.tensor([[[0.5, -0.5]]], requires_grad=True), dim=-1)
    output = ForecastOutput(
        point=means.mean(dim=3),
        distribution=MixtureParams(means=means, log_sigma=log_sigma, rho=rho, log_pi=log_pi),
        modes=means,
        mode_logits=log_pi,
    )

    loss, components = recipe.compute_sample_loss(output, target, _loss_batch(), epoch=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert components["mse"] == 0.0
    assert means.grad is not None


def test_mixture_nll_auxiliary_can_train_distribution_with_hard_min() -> None:
    config = make_config("mart_hypergraph", use_velocity=True)
    config.loss.sample_loss_type = "hard_min_ade"
    config.loss.mixture_nll_weight = 0.01
    config.loss.mixture_nll_detach_means = True
    recipe = ForecastTrainingRecipe(
        loss_config=config.loss,
        normalizer=Normalizer(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([1.0, 1.0])),
        device=torch.device("cpu"),
        grad_clip=0.0,
        mixed_precision=False,
    )
    target = torch.zeros(1, 2, 1, 2)
    means = torch.zeros(1, 2, 1, 2, 2, requires_grad=True)
    log_sigma = torch.zeros_like(means, requires_grad=True)
    rho = torch.zeros(1, 2, 1, 2, requires_grad=True)
    log_pi = torch.log_softmax(torch.tensor([[[0.5, -0.5]]], requires_grad=True), dim=-1)
    output = ForecastOutput(
        point=means.mean(dim=3),
        distribution=MixtureParams(means=means, log_sigma=log_sigma, rho=rho, log_pi=log_pi),
        modes=means,
    )

    loss, components = recipe.compute_sample_loss(output, target, _loss_batch(), epoch=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert components["sample"] == 0.0
    assert components["nll"] > 0.0
    if means.grad is not None:
        assert torch.count_nonzero(means.grad) == 0
    assert log_sigma.grad is not None


def test_primary_mixture_nll_honors_detach_means() -> None:
    config = make_config("graph_transformer", use_velocity=True)
    config.model.head.head_type = "mdn"
    config.model.head.mdn_components = 2
    config.loss.mixture_nll_detach_means = True
    recipe = ForecastTrainingRecipe(
        loss_config=config.loss,
        normalizer=Normalizer(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([1.0, 1.0])),
        device=torch.device("cpu"),
        grad_clip=0.0,
        mixed_precision=False,
        training_policy=resolve_training_loss_policy(config),
    )
    target = torch.zeros(1, 2, 1, 2)
    means = torch.zeros(1, 2, 1, 2, 2, requires_grad=True)
    log_sigma = torch.zeros_like(means, requires_grad=True)
    rho = torch.zeros(1, 2, 1, 2, requires_grad=True)
    log_pi = torch.log_softmax(torch.tensor([[[0.5, -0.5]]], requires_grad=True), dim=-1)
    output = ForecastOutput(
        point=means.mean(dim=3),
        distribution=MixtureParams(means=means, log_sigma=log_sigma, rho=rho, log_pi=log_pi),
    )

    loss, components = recipe.compute_distribution_loss(output, target, _loss_batch(), epoch=1)
    loss.backward()

    assert torch.isfinite(loss)
    assert components["nll"] > 0.0
    if means.grad is not None:
        assert torch.count_nonzero(means.grad) == 0
    assert log_sigma.grad is not None


def test_assert_mixture_finite_reports_tensor_name() -> None:
    from nba_forecasting.runner.recipe import _assert_mixture_finite

    params = MixtureParams(
        means=torch.tensor([[[[[float("nan"), 0.0]]]]]),
        log_sigma=torch.zeros(1, 2, 1, 2, 2),
        rho=torch.zeros(1, 2, 1, 2),
        log_pi=torch.log_softmax(torch.zeros(1, 1, 2), dim=-1),
    )
    with pytest.raises(FloatingPointError, match="distribution.means"):
        _assert_mixture_finite(params, epoch=31, batch_idx=4)


def test_oracle_student_loss_uses_detached_best_mode_after_warmup() -> None:
    config = make_config("graph_transformer", use_velocity=True)
    config.loss.oracle_student_loss_weight = 1.0
    config.loss.oracle_student_start_epoch = 3
    recipe = ForecastTrainingRecipe(
        loss_config=config.loss,
        normalizer=Normalizer(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([1.0, 1.0])),
        device=torch.device("cpu"),
        grad_clip=0.0,
        mixed_precision=False,
    )
    target = torch.zeros(1, 1, 1, 2)
    means = torch.tensor([[[[[0.0, 0.0], [2.0, 0.0]]]]], requires_grad=True)
    log_sigma = torch.zeros_like(means)
    rho = torch.zeros(1, 1, 1, 2)
    log_pi = torch.log_softmax(torch.tensor([[[0.0, 0.0]]]), dim=-1)
    student = torch.tensor([[[[1.0, 0.0]]]], requires_grad=True)
    output = ForecastOutput(
        point=student,
        distribution=MixtureParams(means=means, log_sigma=log_sigma, rho=rho, log_pi=log_pi),
        modes=means,
        oracle_student=student,
    )

    early_loss, early_components = recipe.compute_distribution_loss(output, target, _loss_batch(), epoch=1)
    early_loss.backward(retain_graph=True)
    assert early_components["oracle_student"] == 0.0
    assert student.grad is None

    active_loss, active_components = recipe.compute_distribution_loss(output, target, _loss_batch(), epoch=3)
    active_loss.backward()
    assert active_components["oracle_student"] > 0.0
    assert student.grad is not None
    assert torch.count_nonzero(student.grad) > 0


def test_context_noise_leaves_targets_unchanged_and_recomputes_velocity() -> None:
    normalizer = Normalizer(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([2.0, 4.0]))
    seq = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 0.0]],
            [[2.0, 4.0, 1.0, 0.0]],
            [[4.0, 8.0, 1.0, 0.0]],
        ]
    )
    base = SequenceWindowDataset(
        sequence_paths={0: seq},
        normalizer=normalizer,
        context_size=2,
        horizon_size=1,
        use_velocity=True,
    )[0]
    config = make_config("mart_hypergraph", use_velocity=True)
    config.experiment.context_size = 2
    config.experiment.horizon_size = 1
    config.features.augmentation.context_noise_std_ft = 1.0
    noisy_dataset, _, _, _ = build_dataloaders(
        config,
        {"train": [0], "val": [0], "holdout": []},
        {0: seq},
        normalizer,
    )
    noisy = noisy_dataset[0]

    torch.testing.assert_close(noisy["future_xy"], base["future_xy"])
    torch.testing.assert_close(noisy["future_xy_raw"], base["future_xy_raw"])
    torch.testing.assert_close(noisy["context"][1, :, 2:4], noisy["context"][1, :, :2] - noisy["context"][0, :, :2])


def test_context_frame_dropout_zero_velocity_keeps_final_frame_velocity() -> None:
    normalizer = Normalizer(mu=torch.tensor([0.0, 0.0]), sigma=torch.tensor([1.0, 1.0]))
    seq = torch.tensor(
        [
            [[0.0, 0.0, 1.0, 0.0]],
            [[1.0, 1.0, 1.0, 0.0]],
            [[3.0, 3.0, 1.0, 0.0]],
            [[6.0, 6.0, 1.0, 0.0]],
        ]
    )
    config = make_config("mart_hypergraph", use_velocity=True)
    config.experiment.context_size = 3
    config.experiment.horizon_size = 1
    config.features.augmentation.context_frame_dropout_prob = 1.0
    config.features.augmentation.context_frame_dropout_mode = "zero_velocity"
    dataset, _, _, _ = build_dataloaders(
        config,
        {"train": [0], "val": [0], "holdout": []},
        {0: seq},
        normalizer,
    )

    window = dataset[0]

    torch.testing.assert_close(window["context"][:-1, :, 2:4], torch.zeros(2, 1, 2))
    torch.testing.assert_close(window["context"][-1, :, 2:4], torch.tensor([[2.0, 2.0]]))
    torch.testing.assert_close(window["future_xy_raw"], torch.tensor([[[6.0, 6.0]]]))


def test_non_notebook_model_can_use_notebook_sampling(tmp_path: Path) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24, 26, 28, 30], test_count=2)
    config_path = tmp_path / "dynamic_graph_notebook_sampling.toml"
    config_path.write_text(
        f"""
[experiment]
name = "dynamic_graph_notebook_sampling"
seed = 0
data_dir = "{data_dir}"
output_root = "{tmp_path / "runs"}"
split_path = "{tmp_path / "artifacts" / "splits" / "default.json"}"
device = "cpu"
context_size = 8
horizon_size = 12
num_entities = 11
checkpoint_metric = "ade"

[features]
use_velocity = true

[model]
family = "dynamic_graph_direct"

[model.common]
hidden_dim = 32
message_dim = 32
num_layers = 1
dropout = 0.0

[model.graph]
k_neighbors = 2
use_semantic_flags = true
dynamic_edges = true
use_relation_types = false
use_entity_embeddings = false
entity_embedding_dim = 8
decoder_type = "direct"
direct_decoder_heads = 2
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
    assert checkpoint["config"]["training"]["train_window_sampling"] == "random_window_per_sequence"


def test_batch_train_runs_multiple_jobs(tmp_path: Path) -> None:
    data_dir = write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24, 26, 28, 30], test_count=2)
    config_a = tmp_path / "a.toml"
    config_b = tmp_path / "b.toml"
    base = f"""
[experiment]
seed = 0
data_dir = "{data_dir}"
output_root = "{tmp_path / "runs"}"
split_path = "{tmp_path / "artifacts" / "splits" / "default.json"}"
device = "cpu"
context_size = 8
horizon_size = 12
num_entities = 11
checkpoint_metric = "ade"

[features]
use_velocity = true

[model]
family = "dynamic_graph_direct"

[model.common]
hidden_dim = 32
message_dim = 32
num_layers = 1
dropout = 0.0

[model.graph]
k_neighbors = 2
use_semantic_flags = true
dynamic_edges = true
use_relation_types = false
use_entity_embeddings = false
entity_embedding_dim = 8
decoder_type = "direct"
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
        base.replace('[model]\nfamily = "dynamic_graph_direct"', '[model]\nfamily = "graph_transformer"', 1).replace(
            "[experiment]", '[experiment]\nname = "job_a"', 1
        )
    )
    config_b.write_text(base.replace("[experiment]", '[experiment]\nname = "job_b"', 1))
    batch_dir = train_many_experiments(
        configs=[str(config_a), str(config_b)],
        seeds=[0, 1],
        overrides={"experiment.device": "cpu"},
        evaluate_splits=["val"],
    )
    summary = json.loads((batch_dir / "batch_summary.json").read_text())
    assert len(summary["jobs"]) == 4
    assert all(job["status"] == "completed" for job in summary["jobs"])


def test_config_relative_paths_do_not_depend_on_cwd(tmp_path: Path) -> None:
    write_synthetic_dataset(tmp_path, train_lengths=[20, 22, 24, 26, 28, 30, 32, 34, 36, 38], test_count=2)
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "relative.toml"
    config_path.write_text(
        f"""
[experiment]
name = "relative_paths"
seed = 0
data_dir = "{tmp_path}/data"
output_root = "{tmp_path}/runs"
split_path = "{tmp_path}/artifacts/splits/default.json"
device = "cpu"
context_size = 8
horizon_size = 12
num_entities = 11
checkpoint_metric = "ade"

[features]
use_velocity = true

[model]
family = "dynamic_graph_direct"

[model.common]
hidden_dim = 32
message_dim = 32
num_layers = 1
dropout = 0.0

[model.graph]
k_neighbors = 2
use_semantic_flags = true
dynamic_edges = true
decoder_type = "direct"
direct_decoder_heads = 2
"""
    )
    config, path = load_config(config_path)
    assert path.resolve() == config_path.resolve()
