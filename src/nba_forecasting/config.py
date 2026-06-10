from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback for cluster envs
    import tomli as tomllib  # type: ignore


DEFAULT_CONTEXT_SIZE = 8
DEFAULT_HORIZON_SIZE = 12
DEFAULT_NUM_ENTITIES = 11


def _filter_mapping(cls, data: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in data.items() if key in allowed}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path) -> dict[str, Any]:
    data = tomllib.loads(path.read_text())
    extends = data.pop("extends", None)
    if not extends:
        return data
    base_path = (path.parent / extends).resolve()
    return _deep_merge(_load_raw_config(base_path), data)


@dataclass(slots=True)
class ExperimentSection:
    name: str
    seed: int = 0
    data_dir: str = "data"
    output_root: str = "runs"
    split_path: str = "artifacts/splits/default_split.json"
    device: str = "auto"
    context_size: int = DEFAULT_CONTEXT_SIZE
    horizon_size: int = DEFAULT_HORIZON_SIZE
    num_entities: int = DEFAULT_NUM_ENTITIES
    checkpoint_metric: str = "ade"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ExperimentSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class FeatureSection:
    use_velocity: bool = True
    reflection: ReflectionSection = field(default_factory=lambda: ReflectionSection())
    augmentation: AugmentationSection = field(default_factory=lambda: AugmentationSection())

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> FeatureSection:
        reflection = data.get("reflection", {})
        if not isinstance(reflection, ReflectionSection):
            reflection = ReflectionSection.from_mapping(reflection)
        augmentation = data.get("augmentation", {})
        if not isinstance(augmentation, AugmentationSection):
            augmentation = AugmentationSection.from_mapping(augmentation)
        filtered = _filter_mapping(cls, data)
        return cls(
            use_velocity=filtered.get("use_velocity", True),
            reflection=reflection,
            augmentation=augmentation,
        )


@dataclass(slots=True)
class ReflectionSection:
    y_prob: float = 0.0  # probability of a y -> -y court mirror (left/right)
    x_prob: float = 0.0  # probability of an x -> -x court mirror (attack-direction)
    mode: str = "random"  # random | expand

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ReflectionSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class AugmentationSection:
    context_noise_std_ft: float = 0.0
    context_frame_dropout_prob: float = 0.0
    context_frame_dropout_mode: str = "previous"  # previous | zero_velocity
    agent_permutation_prob: float = 0.0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> AugmentationSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class CommonModelSection:
    hidden_dim: int = 128
    message_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> CommonModelSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class GraphModelSection:
    k_neighbors: int = 3
    use_semantic_flags: bool = True
    dynamic_edges: bool = True
    separate_ball_head: bool = False
    use_relation_types: bool = False
    use_entity_embeddings: bool = False
    entity_embedding_dim: int = 16
    decoder_type: str = "autoregressive"
    direct_decoder_heads: int = 4

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> GraphModelSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class GraphTransformerModelSection:
    transformer_heads: int = 4
    spatial_layers: int = 2
    temporal_layers: int = 2
    decoder_layers: int = 2
    transformer_ffn_multiplier: float = 2.0
    graph_ablation_mode: str = "normal"
    num_virtual_nodes: int = 0

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> GraphTransformerModelSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class ForecastHeadSection:
    head_type: str = "delta"  # delta | mdn
    mdn_components: int = 1  # K trajectory modes when head_type == "mdn"
    mdn_point_estimate: str = "mean"  # mean | mode | medoid | student for the point output
    oracle_student_head: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ForecastHeadSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class MartModelSection:
    mart_function_type: int = 2
    mart_hyper_scale: int = 2
    mart_decoder_hidden_dim: int = 128
    mart_sample_k: int = 1
    mart_point_aggregation: str = "mean"  # mean | learned_softmax | argmax | distribution_softmax | distribution_argmax
    mart_decoder_variant: str = "direct"  # direct | type_conditioned
    mart_mode_distribution: str = "none"  # none | gaussian
    mart_mixture_min_log_sigma: float = -7.0
    mart_mixture_max_log_sigma: float = 5.0
    mart_distribution_detach_features: bool = False

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> MartModelSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class BaselineModelConfig:
    family: str = "baseline"
    common: CommonModelSection = field(default_factory=CommonModelSection)


@dataclass(slots=True)
class DynamicGraphModelConfig:
    family: str = "dynamic_graph_direct"
    common: CommonModelSection = field(default_factory=CommonModelSection)
    graph: GraphModelSection = field(default_factory=GraphModelSection)


@dataclass(slots=True)
class GraphTransformerModelConfig:
    family: str = "graph_transformer"
    common: CommonModelSection = field(default_factory=CommonModelSection)
    graph: GraphModelSection = field(default_factory=GraphModelSection)
    graph_transformer: GraphTransformerModelSection = field(default_factory=GraphTransformerModelSection)
    head: ForecastHeadSection = field(default_factory=ForecastHeadSection)


@dataclass(slots=True)
class MartModelConfig:
    family: str = "mart_hypergraph"
    common: CommonModelSection = field(default_factory=CommonModelSection)
    graph_transformer: GraphTransformerModelSection = field(default_factory=GraphTransformerModelSection)
    mart: MartModelSection = field(default_factory=MartModelSection)


ModelConfig = BaselineModelConfig | DynamicGraphModelConfig | GraphTransformerModelConfig | MartModelConfig

_MODEL_FAMILY_ALIASES = {
    "baseline": "baseline",
    "joint_gru": "joint_gru",
    "static_graph": "static_graph",
    "dynamic_graph_direct": "dynamic_graph_direct",
    "dynamic_graph_autoregressive": "dynamic_graph_autoregressive",
    "graph_transformer": "graph_transformer",
    "mart_hypergraph": "mart_hypergraph",
}


def parse_model_config(data: dict[str, Any]) -> ModelConfig:
    if "family" not in data:
        raise KeyError("Missing required model.family.")
    raw_family = str(data["family"])
    if raw_family not in _MODEL_FAMILY_ALIASES:
        raise KeyError(f"Unknown model.family {raw_family!r}.")
    family = _MODEL_FAMILY_ALIASES[raw_family]
    common = CommonModelSection.from_mapping(data.get("common", {}))
    if family in {"baseline", "joint_gru"}:
        return BaselineModelConfig(family=family, common=common)
    if family in {"dynamic_graph_direct", "static_graph", "dynamic_graph_autoregressive"}:
        return DynamicGraphModelConfig(
            family=family,
            common=common,
            graph=GraphModelSection.from_mapping(data.get("graph", {})),
        )
    if family == "graph_transformer":
        return GraphTransformerModelConfig(
            family=family,
            common=common,
            graph=GraphModelSection.from_mapping(data.get("graph", {})),
            graph_transformer=GraphTransformerModelSection.from_mapping(data.get("graph_transformer", {})),
            head=ForecastHeadSection.from_mapping(data.get("head", {})),
        )
    return MartModelConfig(
        family=family,
        common=common,
        graph_transformer=GraphTransformerModelSection.from_mapping(data.get("graph_transformer", {})),
        mart=MartModelSection.from_mapping(data.get("mart", {})),
    )


@dataclass(slots=True)
class RuntimeSection:
    optimization_backend: str = "compile"
    optimizer_backend: str = "adamw_fused"
    mixed_precision: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> RuntimeSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class TrainingSection:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "none"  # none | multistep | cosine | cosine_warmup
    lr_milestones: list[int] = field(default_factory=list)
    lr_gamma: float = 0.1
    grad_clip: float = 1.0
    num_workers: int = -1
    patience: int = 8
    log_every: int = 50
    teacher_forcing_prob: float = 0.0
    scheduled_sampling_final_prob: float = 0.0
    ema_decay: float = 0.0
    swa_start_pct: float = 0.0
    lr_warmup_epochs: int = 0  # linear-warmup length used by cosine_warmup
    lr_min: float = 0.0  # cosine floor (eta_min)
    swa_lr: float | None = None  # SWALR target for the SWA tail; None -> keep constant LR
    max_train_batches: int | None = None
    max_val_batches: int | None = None
    train_window_sampling: str = "all_windows"
    val_window_sampling: str = "all_windows"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TrainingSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class LossSection:
    # Physics-prior auxiliary losses. All lambdas default 0.0 -> exact MSE-only behavior.
    lambda_smooth: float = 0.0  # weight on the second-difference (acceleration) penalty
    lambda_boundary: float = 0.0  # weight on the off-court hinge
    lambda_ball: float = 0.0  # weight on the ball-isolation hinge
    court_x_half: float = 47.0  # half court length (ft); frame is origin-centered
    court_y_half: float = 25.0  # half court width (ft)
    boundary_margin: float = 2.0  # ft of slack before the boundary hinge engages
    ball_max_dist: float = 8.0  # ft; hinge engages when nearest player is farther (~p99)
    sample_loss_type: str = "hard_min_ade"  # hard_min_ade | softmin_ade | mixture_nll
    sample_loss_space: str = "normalized"  # normalized | raw
    raw_loss_scale_ft: float = 0.0  # <=0 auto-scales raw ADE/MSE terms by normalizer sigma
    softmin_temperature: float = 1.0
    point_loss_weight: float = 0.0
    point_loss_start_epoch: int = 1
    point_loss_warmup_epochs: int = 0
    mixture_nll_weight: float = 0.0
    mixture_nll_start_epoch: int = 1
    mixture_nll_warmup_epochs: int = 0
    mixture_nll_detach_means: bool = False
    oracle_student_loss_weight: float = 0.0
    oracle_student_start_epoch: int = 5
    oracle_student_warmup_epochs: int = 10

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> LossSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class EvaluationSection:
    num_examples: int = 3
    tta_group: str = "none"  # none | y | x | xy

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> EvaluationSection:
        return cls(**_filter_mapping(cls, data))


@dataclass(slots=True)
class RunConfig:
    experiment: ExperimentSection
    features: FeatureSection = field(default_factory=FeatureSection)
    model: ModelConfig = field(default_factory=BaselineModelConfig)
    runtime: RuntimeSection = field(default_factory=RuntimeSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    loss: LossSection = field(default_factory=LossSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)

    @property
    def input_dim(self) -> int:
        return 6 if self.features.use_velocity else 4

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        if "model_name" in data.get("experiment", {}):
            raise KeyError("experiment.model_name was removed. Use [model].family.")
        if "model" not in data:
            raise KeyError("Missing required [model] section.")
        return cls(
            experiment=ExperimentSection.from_mapping(data["experiment"]),
            features=FeatureSection.from_mapping(
                {
                    **{k: v for k, v in data.get("features", {}).items() if k not in {"reflection", "augmentation"}},
                    "reflection": ReflectionSection.from_mapping(data.get("features", {}).get("reflection", {})),
                    "augmentation": AugmentationSection.from_mapping(data.get("features", {}).get("augmentation", {})),
                }
            ),
            model=parse_model_config(data.get("model", {})),
            runtime=RuntimeSection.from_mapping(data.get("runtime", {})),
            training=TrainingSection.from_mapping(data.get("training", {})),
            loss=LossSection.from_mapping(data.get("loss", {})),
            evaluation=EvaluationSection.from_mapping(data.get("evaluation", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def resolve_config_path(config_arg: str | Path) -> Path:
    candidate = Path(config_arg)
    if candidate.exists():
        return candidate.resolve()
    if candidate.suffix:
        candidate = Path("configs") / candidate
    else:
        candidate = Path("configs") / f"{candidate}.toml"
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve config {config_arg!r}.")


def load_config(config_arg: str | Path) -> tuple[RunConfig, Path]:
    path = resolve_config_path(config_arg)
    raw = _load_raw_config(path)
    config = RunConfig.from_dict(raw)
    from .models.builders import validate_loss_config

    validate_loss_config(config)
    return config, path


def project_root_for_config(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    for parent in path.parents:
        if parent.name == "configs":
            return parent.parent
    return path.parent


def _resolve_path(base_dir: Path, raw_path: str) -> str:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def resolve_config_paths(config: RunConfig, config_path: str | Path) -> RunConfig:
    base_dir = project_root_for_config(config_path)
    data = config.to_dict()
    experiment = data["experiment"]
    experiment["data_dir"] = _resolve_path(base_dir, experiment["data_dir"])
    experiment["output_root"] = _resolve_path(base_dir, experiment["output_root"])
    experiment["split_path"] = _resolve_path(base_dir, experiment["split_path"])
    return RunConfig.from_dict(data)


def parse_override_value(raw_value: str) -> Any:
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    for caster in (int, float):
        try:
            return caster(raw_value)
        except ValueError:
            continue
    if (raw_value.startswith('"') and raw_value.endswith('"')) or (raw_value.startswith("'") and raw_value.endswith("'")):
        return raw_value[1:-1]
    return raw_value


def apply_overrides(config: RunConfig, overrides: dict[str, Any]) -> RunConfig:
    if not overrides:
        return config
    data = config.to_dict()
    for path, value in overrides.items():
        parts = path.split(".")
        target: dict[str, Any] = data
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise KeyError(f"Unknown override path {path!r}.")
            target = target[part]
        leaf = parts[-1]
        if leaf not in target:
            raise KeyError(f"Unknown override path {path!r}.")
        if value is None and isinstance(target[leaf], str):
            value = "none"
        target[leaf] = value
    return RunConfig.from_dict(data)
