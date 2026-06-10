from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm: Any = None

from ..config import RunConfig, apply_overrides, load_config, resolve_config_paths
from ..data import Normalizer
from ..models import build_model
from ..runtime_env import training_slots


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this environment. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r}, "
            f"torch.version.cuda={torch.version.cuda!r}, "
            f"torch.cuda.device_count()={torch.cuda.device_count()}."
        )
    if device_arg == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this environment.")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _mixed_precision_enabled(device: torch.device, requested: bool) -> bool:
    return requested and device.type == "cuda"


def _cuda_autocast(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.amp.autocast("cuda")


def _cuda_grad_scaler(enabled: bool) -> Any:
    return torch.amp.GradScaler("cuda", enabled=enabled)


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        import copy

        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def update(self, source_model: torch.nn.Module) -> None:
        with torch.no_grad():
            ema_state = self.model.state_dict()
            source_state = source_model.state_dict()
            for key, ema_value in ema_state.items():
                source_value = source_state[key].detach()
                if torch.is_floating_point(ema_value):
                    ema_value.mul_(self.decay).add_(source_value, alpha=1.0 - self.decay)
                else:
                    ema_value.copy_(source_value)


def _model_parameter_l2(model: torch.nn.Module) -> float:
    total = 0.0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                total += float(parameter.detach().float().square().sum().item())
    return total**0.5


def _mark_compiled_cuda_step_begin(model: torch.nn.Module, device: torch.device) -> None:
    if device.type != "cuda" or not hasattr(model, "_orig_mod"):
        return
    mark_step = getattr(getattr(torch, "compiler", None), "cudagraph_mark_step_begin", None)
    if mark_step is not None:
        mark_step()


def _write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _write_history_csv(path: str | Path, history: list[dict[str, float]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    fieldnames = list(history[0].keys())
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _write_rows_csv(path: str | Path, rows: list[dict[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with target.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return getattr(model, "_orig_mod", model).state_dict()


def _strip_state_dict_prefix(
    state_dict: dict[str, torch.Tensor],
    prefix: str,
) -> dict[str, torch.Tensor]:
    if not state_dict or not all(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix) :]: value for key, value in state_dict.items()}


def _normalize_checkpoint_model_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized = state_dict
    for _ in range(2):
        before = normalized
        normalized = _strip_state_dict_prefix(normalized, "_orig_mod.")
        normalized = _strip_state_dict_prefix(normalized, "module.")
        if normalized is before:
            break
    return normalized


def _load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[dict, RunConfig, Normalizer, torch.nn.Module]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = RunConfig.from_dict(checkpoint["config"])
    normalizer = Normalizer.from_state_dict(checkpoint["normalizer"])
    model = build_model(config).to(device)
    model.load_state_dict(_normalize_checkpoint_model_state_dict(checkpoint["model_state"]))
    return checkpoint, config, normalizer, model


@dataclass(slots=True)
class ExampleRecord:
    error: float
    sequence_id: int
    start_idx: int
    context_xy: torch.Tensor
    target_xy: torch.Tensor
    pred_xy: torch.Tensor
    static: torch.Tensor


METRIC_KEYS = ("loss", "mse", "ade", "rmse", "ade_l2", "fde_l2")


def _progress_disabled() -> bool:
    if os.environ.get("NBA_FORECASTING_DISABLE_TQDM") == "1":
        return True
    if training_slots() > 1:
        return True
    return not sys.stderr.isatty()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def log_epoch_progress(
    *,
    experiment_name: str,
    epoch: int,
    total_epochs: int,
    start_epoch: int,
    started_at: float,
    train_loss: float,
    val_metric: float,
    best_epoch: int,
    best_metric: float,
    checkpoint_metric: str,
    patience: int,
    patience_limit: int,
) -> None:
    """Print a single-line epoch status when tqdm is disabled (e.g. parallel GPU runs)."""
    if not _progress_disabled():
        return
    done_epochs = epoch - start_epoch + 1
    planned_epochs = total_epochs - start_epoch + 1
    elapsed = time.monotonic() - started_at
    remaining_epochs = max(0, planned_epochs - done_epochs)
    eta_seconds = (elapsed / done_epochs) * remaining_epochs if done_epochs > 0 else 0.0
    print(
        f"[{_timestamp()}] epochs:{experiment_name} {epoch}/{total_epochs} "
        f"train={train_loss:.4f} val_{checkpoint_metric}={val_metric:.2f} "
        f"best@{best_epoch}={best_metric:.2f} patience={patience}/{patience_limit} "
        f"eta={_format_duration(eta_seconds)}",
        flush=True,
    )


def _progress_bar(iterable: Any, *, total: int | None, desc: str, leave: bool = False) -> Any:
    if tqdm is None:
        return iterable
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        leave=leave,
        dynamic_ncols=True,
        disable=_progress_disabled(),
    )


def _scheduled_sampling_probability(config: RunConfig, epoch: int) -> float:
    final_prob = config.training.scheduled_sampling_final_prob
    if final_prob <= 0.0:
        return 0.0
    if config.training.epochs <= 1:
        return final_prob
    return final_prob * float(epoch - 1) / float(config.training.epochs - 1)


def _optional_batch_limit(raw_limit: int | None) -> int | None:
    if raw_limit is None or raw_limit <= 0:
        return None
    return raw_limit


def _run_dir(config: RunConfig) -> Path:
    root = Path(config.experiment.output_root) / config.experiment.name / _timestamp()
    root.mkdir(parents=True, exist_ok=False)
    return root.resolve()


def _checkpoint_run_dir(resume_state: dict[str, Any] | None) -> Path | None:
    if resume_state is None:
        return None
    raw_run_dir = resume_state.get("run_dir")
    if not raw_run_dir:
        return None
    return Path(raw_run_dir).expanduser().resolve()


def _assert_resume_compatible(config: RunConfig, resume_state: dict[str, Any], resume_checkpoint: Path) -> None:
    checkpoint_config = resume_state.get("config", {})
    checkpoint_experiment = checkpoint_config.get("experiment", {})
    checkpoint_model = checkpoint_config.get("model", {})
    expected = {
        "family": config.model.family,
        "context_size": config.experiment.context_size,
        "horizon_size": config.experiment.horizon_size,
        "num_entities": config.experiment.num_entities,
    }
    actual = {
        "family": checkpoint_model.get("family"),
        "context_size": checkpoint_experiment.get("context_size"),
        "horizon_size": checkpoint_experiment.get("horizon_size"),
        "num_entities": checkpoint_experiment.get("num_entities"),
    }
    if actual != expected:
        raise ValueError(f"Resume checkpoint {resume_checkpoint} is not compatible with the requested config.")


def _load_runtime_config(
    config_arg: str | Path,
    overrides: dict[str, object] | None = None,
) -> tuple[RunConfig, Path]:
    config, config_path = load_config(config_arg)
    if overrides:
        config = apply_overrides(config, overrides)
    config = resolve_config_paths(config, config_path)
    from ..models.builders import validate_loss_config

    validate_loss_config(config)
    return config, config_path
