from __future__ import annotations

from pathlib import Path

from nba_forecasting.config import (
    BaselineModelConfig,
    DynamicGraphModelConfig,
    GraphTransformerModelConfig,
    MartModelConfig,
    apply_overrides,
    load_config,
    parse_model_config,
    parse_override_value,
    resolve_config_paths,
)
from tests.conftest import make_config


def test_override_helpers() -> None:
    config = make_config("baseline", use_velocity=False)
    overrides = {
        "experiment.device": "cpu",
        "training.epochs": 7,
        "features.use_velocity": False,
    }
    updated = apply_overrides(config, overrides)
    assert updated.experiment.device == "cpu"
    assert updated.training.epochs == 7
    assert updated.features.use_velocity is False
    assert parse_override_value("true") is True
    assert parse_override_value("13") == 13
    assert parse_override_value("1.25") == 1.25
    assert parse_override_value("mps") == "mps"
    updated = apply_overrides(config, {"runtime.optimization_backend": parse_override_value("none")})
    assert updated.runtime.optimization_backend == "none"


def test_nested_config_paths_resolve_from_project_root() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "tune_graph_transformer.toml"
    config, _ = load_config(config_path)
    resolved = resolve_config_paths(config, config_path)
    root = Path(__file__).parents[1]

    assert resolved.experiment.data_dir == str(root / "data")
    assert resolved.experiment.output_root == str(root / "runs")
    assert resolved.experiment.split_path == str(root / "artifacts" / "splits" / "default_split.json")


def test_only_base_config_defines_task_window_shape() -> None:
    config_dir = Path(__file__).parents[1] / "configs"
    offenders = []
    for config_path in sorted(config_dir.glob("*.toml")):
        if config_path.name == "base.toml":
            continue
        text = config_path.read_text()
        if "context_size" in text or "horizon_size" in text:
            offenders.append(config_path.name)
    assert offenders == []


def test_base_config_defines_shared_sections_only() -> None:
    base_text = (Path(__file__).parents[1] / "configs" / "base.toml").read_text()
    assert "[model" not in base_text


def test_parse_model_config_returns_family_specific_types() -> None:
    baseline = parse_model_config({"family": "baseline"})
    assert isinstance(baseline, BaselineModelConfig)
    assert not hasattr(baseline, "head")

    graph_transformer = parse_model_config(
        {
            "family": "graph_transformer",
            "head": {"head_type": "mdn", "mdn_components": 4},
        }
    )
    assert isinstance(graph_transformer, GraphTransformerModelConfig)
    assert graph_transformer.head.head_type == "mdn"
    assert graph_transformer.head.mdn_components == 4

    mart = parse_model_config({"family": "mart_hypergraph", "mart": {"mart_function_type": 2}})
    assert isinstance(mart, MartModelConfig)
    assert mart.mart.mart_function_type == 2

    joint = parse_model_config({"family": "joint_gru"})
    assert isinstance(joint, BaselineModelConfig)
    assert joint.family == "joint_gru"

    static = parse_model_config({"family": "static_graph", "graph": {"dynamic_edges": False}})
    assert isinstance(static, DynamicGraphModelConfig)
    assert static.family == "static_graph"
    assert static.graph.dynamic_edges is False

    autoreg = parse_model_config({"family": "dynamic_graph_autoregressive"})
    assert isinstance(autoreg, DynamicGraphModelConfig)
    assert autoreg.family == "dynamic_graph_autoregressive"
