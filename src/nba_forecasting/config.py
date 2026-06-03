from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback for cluster envs
    import tomli as tomllib


DEFAULT_CONTEXT_SIZE = 8
DEFAULT_HORIZON_SIZE = 12
DEFAULT_NUM_ENTITIES = 11


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
    model_name: str
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
        return cls(**data)


@dataclass(slots=True)
class FeatureSection:
    use_velocity: bool = True
    reflect_prob: float = 0.0  # probability of a y -> -y court mirror (left/right)
    reflect_x_prob: float = 0.0  # probability of an x -> -x court mirror (attack-direction)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> FeatureSection:
        return cls(**data)


@dataclass(slots=True)
class ModelSection:
    hidden_dim: int = 128
    message_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    k_neighbors: int = 3
    use_semantic_flags: bool = True
    dynamic_edges: bool = True
    message_passing_type: str = "mpnn"
    separate_ball_head: bool = False
    use_relation_types: bool = False
    use_entity_embeddings: bool = False
    entity_embedding_dim: int = 16
    decoder_type: str = "autoregressive"
    direct_decoder_heads: int = 4
    edge_radius: float | None = None
    temporal_ssm_layers: int = 2
    social_ssm_layers: int = 1
    decoder_ssm_layers: int = 2
    ssm_conv_kernel: int = 3
    transformer_heads: int = 4
    spatial_layers: int = 2
    temporal_layers: int = 2
    decoder_layers: int = 2
    transformer_ffn_multiplier: float = 2.0
    graph_ablation_mode: str = "normal"
    optimization_backend: str = "none"
    graph_prep_backend: str = "torch"
    # Phase 7 (D1) mixture-density head. Defaults reproduce the deterministic delta head.
    head_type: str = "delta"  # delta | mdn
    mdn_components: int = 1  # K trajectory modes when head_type == "mdn"
    mdn_point_estimate: str = "mean"  # mean (pi-weighted) | mode (argmax pi) for the point output

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> ModelSection:
        return cls(**data)


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
    num_workers: int = 0
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
    mixed_precision: bool = True

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> TrainingSection:
        return cls(**data)


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

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> LossSection:
        return cls(**data)


@dataclass(slots=True)
class EvaluationSection:
    num_examples: int = 3

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> EvaluationSection:
        return cls(**data)


@dataclass(slots=True)
class RunConfig:
    experiment: ExperimentSection
    features: FeatureSection = field(default_factory=FeatureSection)
    model: ModelSection = field(default_factory=ModelSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    loss: LossSection = field(default_factory=LossSection)
    evaluation: EvaluationSection = field(default_factory=EvaluationSection)

    @property
    def input_dim(self) -> int:
        return 6 if self.features.use_velocity else 4

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        return cls(
            experiment=ExperimentSection.from_mapping(data["experiment"]),
            features=FeatureSection.from_mapping(data.get("features", {})),
            model=ModelSection.from_mapping(data.get("model", {})),
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
    return RunConfig.from_dict(raw), path


def project_root_for_config(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    if path.parent.name == "configs":
        return path.parent.parent
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
        target[leaf] = value
    return RunConfig.from_dict(data)
