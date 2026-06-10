from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from torch import nn

from ..config import (
    GraphTransformerModelConfig,
    MartModelConfig,
    RunConfig,
)
from .baselines import build_baseline_model, build_joint_gru
from .dynamic_graph import (
    build_dynamic_graph_autoregressive,
    build_dynamic_graph_direct,
    build_static_graph,
)
from .graph_transformer import build_graph_transformer
from .mart import build_mart_hypergraph

PrimaryTrainingLoss = Literal["point_mse", "sample_modes", "mixture_nll"]


@dataclass(frozen=True, slots=True)
class TrainingLossPolicy:
    primary: PrimaryTrainingLoss = "point_mse"
    mixture_point_estimate: str | None = None


@dataclass(frozen=True, slots=True)
class FamilySpec:
    builder: Callable[[RunConfig], nn.Module]
    loss_modes: frozenset[PrimaryTrainingLoss]
    search_profiles: dict[str, list[str]] | None = None


MODEL_FAMILIES: dict[str, FamilySpec] = {
    "baseline": FamilySpec(
        builder=build_baseline_model,
        loss_modes=frozenset({"point_mse"}),
    ),
    "joint_gru": FamilySpec(
        builder=build_joint_gru,
        loss_modes=frozenset({"point_mse"}),
        search_profiles={
            "joint_gru": [
                "model.common.hidden_dim=choice(128,192,256)",
                "training.lr=loguniform(1e-4,5e-3)",
                "model.common.dropout=uniform(0.0,0.2)",
            ],
        },
    ),
    "static_graph": FamilySpec(
        builder=build_static_graph,
        loss_modes=frozenset({"point_mse"}),
        search_profiles={
            "static_graph": [
                "model.common.hidden_dim=choice(128,192,256)",
                "training.lr=loguniform(1e-4,5e-3)",
                "model.common.dropout=uniform(0.0,0.2)",
                "model.common.message_dim=choice(128,192,256)",
            ],
        },
    ),
    "dynamic_graph_autoregressive": FamilySpec(
        builder=build_dynamic_graph_autoregressive,
        loss_modes=frozenset({"point_mse"}),
        search_profiles={
            "dynamic_graph_autoregressive": [
                "model.common.hidden_dim=choice(128,192,256)",
                "training.lr=loguniform(1e-4,5e-3)",
                "model.common.dropout=uniform(0.0,0.2)",
                "model.common.message_dim=choice(128,192,256)",
                "model.graph.k_neighbors=choice(3,5,7)",
            ],
        },
    ),
    "dynamic_graph_direct": FamilySpec(
        builder=build_dynamic_graph_direct,
        loss_modes=frozenset({"point_mse"}),
        search_profiles={
            "graph": [
                "model.common.hidden_dim=choice(128,192,256)",
                "training.lr=loguniform(1e-4,5e-3)",
                "model.common.dropout=uniform(0.0,0.2)",
                "model.common.message_dim=choice(128,192,256)",
                "model.graph.k_neighbors=choice(3,5,7)",
            ],
            "graph_fixed_message": [
                "model.common.hidden_dim=choice(128,192,256)",
                "training.lr=loguniform(1e-4,5e-3)",
                "model.common.dropout=uniform(0.0,0.2)",
                "model.common.message_dim=choice(128,192,256)",
                "model.graph.k_neighbors=choice(3,5,7)",
            ],
        },
    ),
    "graph_transformer": FamilySpec(
        builder=build_graph_transformer,
        loss_modes=frozenset({"point_mse", "sample_modes", "mixture_nll"}),
        search_profiles={
            "graph_transformer": [
                "model.common.hidden_dim=choice(128,192,256)",
                "training.lr=loguniform(1e-4,5e-3)",
                "model.common.dropout=uniform(0.0,0.2)",
                "model.graph.k_neighbors=choice(3,5,7)",
                "model.graph_transformer.transformer_heads=choice(4,8)",
                "model.graph_transformer.spatial_layers=choice(1,2,3)",
                "model.graph_transformer.temporal_layers=choice(1,2,3)",
                "model.graph_transformer.decoder_layers=choice(1,2)",
                "model.graph_transformer.transformer_ffn_multiplier=choice(2.0,3.0)",
            ],
        },
    ),
    "mart_hypergraph": FamilySpec(
        builder=build_mart_hypergraph,
        loss_modes=frozenset({"sample_modes", "mixture_nll"}),
        search_profiles={
            "mart": [
                "model.common.hidden_dim=choice(64,128,192)",
                "model.common.message_dim=choice(128,192,256)",
                "model.common.num_layers=choice(2,3,4)",
                "model.graph_transformer.transformer_heads=choice(4,8)",
                "model.common.dropout=uniform(0.0,0.2)",
                "model.mart.mart_decoder_hidden_dim=choice(128,256,384)",
                "model.mart.mart_function_type=choice(1,2,3)",
                "training.lr=loguniform(1e-4,2e-3)",
            ],
        },
    ),
}


def list_model_family_names() -> tuple[str, ...]:
    return tuple(sorted(MODEL_FAMILIES))


def get_family_spec(name: str) -> FamilySpec:
    try:
        return MODEL_FAMILIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown model family {name!r}. Known families: {list_model_family_names()}") from exc


def build_model(config: RunConfig) -> nn.Module:
    return get_family_spec(config.model.family).builder(config)


def collect_search_profiles() -> dict[str, list[str]]:
    profiles: dict[str, list[str]] = {}
    for spec in MODEL_FAMILIES.values():
        if spec.search_profiles:
            profiles.update(spec.search_profiles)
    return profiles


def _forecast_traits_from_config(config: RunConfig) -> tuple[bool, bool, str | None]:
    model = config.model
    if model.family == "graph_transformer":
        assert isinstance(model, GraphTransformerModelConfig)
        mixture = model.head.head_type == "mdn"
        return mixture, mixture, model.head.mdn_point_estimate if mixture else None
    if model.family == "mart_hypergraph":
        assert isinstance(model, MartModelConfig)
        mixture = model.mart.mart_mode_distribution == "gaussian"
        return True, mixture, None
    return False, False, None


def resolve_training_loss_policy(config: RunConfig) -> TrainingLossPolicy:
    multi_mode, mixture_distribution, mixture_point_estimate = _forecast_traits_from_config(config)
    loss_config = config.loss
    if loss_config.sample_loss_type == "mixture_nll":
        if not mixture_distribution:
            raise ValueError("loss.sample_loss_type='mixture_nll' requires a model with mixture_distribution capability.")
        return TrainingLossPolicy(primary="mixture_nll", mixture_point_estimate=mixture_point_estimate)
    if mixture_distribution and mixture_point_estimate is not None:
        return TrainingLossPolicy(primary="mixture_nll", mixture_point_estimate=mixture_point_estimate)
    if multi_mode:
        return TrainingLossPolicy(primary="sample_modes")
    return TrainingLossPolicy(primary="point_mse")


def mixture_point_estimate_for_inference(config: RunConfig) -> str | None:
    _, mixture_distribution, mixture_point_estimate = _forecast_traits_from_config(config)
    return mixture_point_estimate if mixture_distribution else None


def has_mixture_distribution(config: RunConfig) -> bool:
    return _forecast_traits_from_config(config)[1]


def validate_loss_config(config: RunConfig) -> None:
    policy = resolve_training_loss_policy(config)
    allowed = get_family_spec(config.model.family).loss_modes
    if policy.primary not in allowed:
        raise ValueError(
            f"Model family {config.model.family!r} does not support primary loss {policy.primary!r}. Allowed: {sorted(allowed)}."
        )
