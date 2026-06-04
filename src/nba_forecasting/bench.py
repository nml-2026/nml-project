from __future__ import annotations

import contextlib
import json
import math
import os
import statistics
import time
import traceback
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import torch

from .config import RunConfig, apply_overrides, load_config, resolve_config_paths
from .models import build_model
from .runner import (
    _apply_optimization_backend,
    _cuda_autocast,
    _mark_compiled_cuda_step_begin,
    _mixed_precision_enabled,
    build_optimizer,
    normalized_mse_loss,
    resolve_device,
    set_seed,
)


class BenchTimer:
    def __init__(self, *, warmup: int = 50, iterations: int = 200, device: torch.device | None = None) -> None:
        self.warmup = warmup
        self.iterations = iterations
        self.device = device or torch.device("cpu")

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()

    def time(self, fn: Callable[[], object]) -> dict[str, float]:
        for _ in range(self.warmup):
            fn()
        self._sync()
        timings_ms: list[float] = []
        for _ in range(self.iterations):
            if self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                end.synchronize()
                timings_ms.append(float(start.elapsed_time(end)))
            else:
                started = time.perf_counter()
                fn()
                self._sync()
                timings_ms.append((time.perf_counter() - started) * 1000.0)
        return _summarize_ms(timings_ms)


def _summarize_ms(values: list[float]) -> dict[str, float]:
    if not values:
        return {"median_ms": 0.0, "p95_ms": 0.0, "std_ms": 0.0}
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, math.ceil(0.95 * (len(ordered) - 1)))
    return {
        "median_ms": float(statistics.median(ordered)),
        "p95_ms": float(ordered[p95_idx]),
        "std_ms": float(statistics.pstdev(ordered)),
    }


def _parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _combinatorial_variants(
    *,
    baseline_name: str,
    knobs: list[tuple[str, dict[str, object]]],
    aliases: dict[str, dict[str, object]] | None = None,
    baseline_overrides: dict[str, object] | None = None,
) -> dict[str, dict[str, object]]:
    baseline = dict(baseline_overrides or {})
    variants: dict[str, dict[str, object]] = {baseline_name: baseline}
    for size in range(1, len(knobs) + 1):
        for combo in combinations(knobs, size):
            name = "__".join(token for token, _ in combo)
            overrides: dict[str, object] = dict(baseline)
            for _, knob_overrides in combo:
                overrides.update(knob_overrides)
            variants[name] = overrides
    if aliases:
        variants.update(aliases)
    return variants


GRAPH_TRANSFORMER_COMBINATORIAL_VARIANTS: dict[str, dict[str, object]] = _combinatorial_variants(
    baseline_name="baseline",
    baseline_overrides={"model.optimization_backend": "none", "model.optimizer_backend": "adamw"},
    knobs=[
        ("fused_graph_prep", {"model.graph_prep_backend": "fused"}),
        ("graph_cache", {"model.graph_cache_backend": "reuse"}),
        ("compile", {"model.optimization_backend": "compile"}),
        ("fused_adamw", {"model.optimizer_backend": "adamw_fused"}),
    ],
)
GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS: dict[str, dict[str, object]] = {
    **GRAPH_TRANSFORMER_COMBINATORIAL_VARIANTS,
    "compile_fused_graph_prep": {
        "model.optimization_backend": "compile",
        "model.optimizer_backend": "adamw",
        "model.graph_prep_backend": "fused",
    },
    "compile_fused_graph_prep_graph_cache": {
        "model.optimization_backend": "compile",
        "model.optimizer_backend": "adamw",
        "model.graph_prep_backend": "fused",
        "model.graph_cache_backend": "reuse",
    },
    "compile_fused_graph_prep_fused_adamw": {
        "model.optimization_backend": "compile",
        "model.graph_prep_backend": "fused",
        "model.optimizer_backend": "adamw_fused",
    },
    "compile_fused_graph_prep_graph_cache_fused_adamw": {
        "model.optimization_backend": "compile",
        "model.graph_prep_backend": "fused",
        "model.graph_cache_backend": "reuse",
        "model.optimizer_backend": "adamw_fused",
    },
}
GRAPH_TRANSFORMER_FOCUSED_OPTIMIZATION_VARIANTS: dict[str, dict[str, object]] = {
    name: GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS[name]
    for name in [
        "baseline",
        "compile_fused_graph_prep",
        "compile_fused_graph_prep_graph_cache",
        "compile_fused_graph_prep_graph_cache_fused_adamw",
    ]
}
MODEL_OPTIMIZATION_VARIANTS: dict[str, dict[str, object]] = {
    **GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS,
}


def _format_overrides(overrides: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(overrides.items()))


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _ratio_speedup(reference: float | None, candidate: float | None) -> float | None:
    if reference is None or candidate is None or reference <= 0.0 or candidate <= 0.0:
        return None
    return (reference / candidate - 1.0) * 100.0


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    header = "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(headers)) + " |"
    divider = "| " + " | ".join("-" * width for width in widths) + " |"
    body = ["| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _optional_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _resolve_mixed_precision_modes(modes: list[str] | None) -> list[bool | None]:
    selected = modes or ["false", "true"]
    resolved: list[bool | None] = []
    for mode in selected:
        normalized = mode.lower()
        if normalized == "config":
            resolved.append(None)
        elif normalized == "true":
            resolved.append(True)
        elif normalized == "false":
            resolved.append(False)
        else:
            raise ValueError("mixed precision modes must be one or more of: config, true, false.")
    return resolved


def _mixed_precision_label(value: bool | None) -> str:
    if value is None:
        return "config"
    return "true" if value else "false"


def _write_optimization_summary(output_dir: Path, payload: dict[str, object]) -> None:
    variants = payload["variants"]
    assert isinstance(variants, list)
    baselines: dict[str, dict[str, object]] = {}
    for row in variants:
        if isinstance(row, dict) and str(row.get("variant", "")) == "baseline":
            baselines[str(row.get("mixed_precision_mode", "config"))] = row

    table_rows: list[list[str]] = []
    csv_rows = [
        "variant,status,mixed_precision_mode,mixed_precision_effective,override_summary,synthetic_full_step_median_ms,synthetic_speedup_pct,real_mean_step_ms,real_speedup_pct,error"
    ]
    for raw in variants:
        assert isinstance(raw, dict)
        synthetic = _optional_float(raw.get("synthetic_full_step_median_ms"))
        real = _optional_float(raw.get("real_mean_step_ms"))
        baseline = baselines.get(str(raw.get("mixed_precision_mode", "config"))) or next(iter(baselines.values()), None)
        baseline_synthetic = _optional_float(baseline.get("synthetic_full_step_median_ms")) if isinstance(baseline, dict) else None
        baseline_real = _optional_float(baseline.get("real_mean_step_ms")) if isinstance(baseline, dict) else None
        synthetic_speedup = _ratio_speedup(baseline_synthetic, synthetic)
        real_speedup = _ratio_speedup(baseline_real, real)
        error = str(raw.get("error", ""))
        table_rows.append(
            [
                str(raw["variant"]),
                str(raw["status"]),
                str(raw.get("mixed_precision_mode", "config")),
                str(raw.get("mixed_precision_effective", raw.get("mixed_precision_effective_real", ""))),
                str(raw.get("override_summary", ""))[:120],
                f"{synthetic:.3f}" if synthetic is not None else "",
                f"{synthetic_speedup:.2f}%" if synthetic_speedup is not None else "",
                f"{real:.3f}" if real is not None else "",
                f"{real_speedup:.2f}%" if real_speedup is not None else "",
                error[:120],
            ]
        )
        csv_rows.append(
            ",".join(
                [
                    str(raw["variant"]),
                    str(raw["status"]),
                    str(raw.get("mixed_precision_mode", "config")),
                    str(raw.get("mixed_precision_effective", raw.get("mixed_precision_effective_real", ""))),
                    f'"{str(raw.get("override_summary", "")).replace(chr(34), chr(34) * 2)}"',
                    f"{synthetic:.6f}" if synthetic is not None else "",
                    f"{synthetic_speedup:.6f}" if synthetic_speedup is not None else "",
                    f"{real:.6f}" if real is not None else "",
                    f"{real_speedup:.6f}" if real_speedup is not None else "",
                    f'"{error.replace(chr(34), chr(34) * 2)}"',
                ]
            )
        )

    summary_md = "\n".join(
        [
            "# Model Optimization Benchmark",
            "",
            f"Config: `{payload['config']}`",
            f"Device: `{payload['device']}`",
            f"Torch: `{payload.get('torch_version', 'unknown')}`",
            f"CUDA: `{payload.get('torch_cuda_version', 'none')}`",
            f"Batch size: `{payload['batch_size']}`",
            f"Mixed precision modes: `{', '.join(str(item) for item in payload.get('mixed_precision_modes', []))}`",
            f"Component iterations: `{payload['component_iterations']}`",
            f"Real dataloader steps: `{payload['real_steps'] if payload['include_real_step'] else 'disabled'}`",
            "",
            _markdown_table(
                [
                    "variant",
                    "status",
                    "mixed precision",
                    "effective",
                    "overrides",
                    "synthetic full step ms",
                    "synthetic speedup",
                    "real step ms",
                    "real speedup",
                    "error",
                ],
                table_rows,
            ),
            "",
            "Synthetic timings use generated tensors and component isolation. Real step timings use the project dataloader.",
            "Failed variants are retained because correctness/build failures are relevant to the cleanup decision.",
            "",
        ]
    )
    (output_dir / "optimization_benchmark_summary.md").write_text(summary_md)
    (output_dir / "optimization_benchmark_summary.csv").write_text("\n".join(csv_rows) + "\n")


@contextlib.contextmanager
def _capture_compile_diagnostics(
    *,
    enabled: bool,
    variant_dir: Path,
    variant: str,
    overrides: dict[str, object],
):
    backend = overrides.get("model.optimization_backend")
    should_capture = enabled and backend in {"compile", "compile_default"}
    if not should_capture:
        yield None
        return

    diagnostics_dir = variant_dir / "compile_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    log_path = diagnostics_dir / "torch_compile_stderr.log"
    metadata_path = diagnostics_dir / "compile_diagnostics.json"
    metadata = {
        "enabled": enabled,
        "variant": variant,
        "overrides": overrides,
        "optimization_backend": backend,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "log_path": str(log_path.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))

    try:
        torch._logging.set_logs(graph_breaks=True, recompiles=True)
    except AttributeError:
        log_path.write_text("torch._logging.set_logs is unavailable in this PyTorch build.\n")
        yield log_path
        return

    with log_path.open("w") as log_handle, contextlib.redirect_stderr(log_handle):
        saved_stderr_fd = os.dup(2)
        log_handle.flush()
        os.dup2(log_handle.fileno(), 2)
        try:
            yield log_path
        finally:
            log_handle.flush()
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
            try:
                torch._logging.set_logs(graph_breaks=False, recompiles=False)
            except AttributeError:
                pass


def _synthetic_inputs(config: RunConfig, device: torch.device, batch_size: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    batch = batch_size or config.training.batch_size
    context = torch.randn(
        batch,
        config.experiment.context_size,
        config.experiment.num_entities,
        config.input_dim,
        device=device,
    )
    context[..., -2:-1] = 1.0
    if config.experiment.num_entities > 0:
        context[:, :, 0, -2] = 0.0
    context[..., -1:] = torch.where(
        torch.arange(config.experiment.num_entities, device=device).view(1, 1, -1, 1) % 2 == 0,
        torch.tensor(1.0, device=device),
        torch.tensor(-1.0, device=device),
    )
    target = torch.randn(
        batch,
        config.experiment.horizon_size,
        config.experiment.num_entities,
        2,
        device=device,
    )
    return context, target


def bench_graph_prep(model: torch.nn.Module, context: torch.Tensor, timer: BenchTimer) -> dict[str, float]:
    batch_size, context_size, num_entities, _ = context.shape
    static = context[:, 0, :, -2:]
    positions = context[:, :, :, :2].reshape(batch_size * context_size, num_entities, 2)
    flat_static = static.unsqueeze(1).expand(-1, context_size, -1, -1).reshape(batch_size * context_size, num_entities, 2)
    return timer.time(lambda: model.graph_inputs(positions, flat_static))


def bench_attention_forward(
    module: torch.nn.Module, node_state: torch.Tensor, edge_feat: torch.Tensor, mask: torch.Tensor, timer: BenchTimer
) -> dict[str, float]:
    module.eval()
    with torch.no_grad():
        return timer.time(lambda: module(node_state, edge_feat, mask))


def bench_attention_backward(
    module: torch.nn.Module, node_state: torch.Tensor, edge_feat: torch.Tensor, mask: torch.Tensor, timer: BenchTimer
) -> dict[str, float]:
    module.train()

    def step() -> None:
        state = node_state.detach().clone().requires_grad_(True)
        output = module(state, edge_feat, mask)
        output.square().mean().backward()

    return timer.time(step)


def bench_spatial_block(
    block: torch.nn.Module, node_state: torch.Tensor, edge_feat: torch.Tensor, mask: torch.Tensor, timer: BenchTimer
) -> dict[str, float]:
    block.train()

    def step() -> None:
        state = node_state.detach().clone().requires_grad_(True)
        block(state, edge_feat, mask).square().mean().backward()

    result = timer.time(step)
    result["parameters"] = _parameter_count(block)
    return result


def bench_temporal_block(block: torch.nn.Module, sequence: torch.Tensor, timer: BenchTimer) -> dict[str, float]:
    block.train()

    def step() -> None:
        seq = sequence.detach().clone().requires_grad_(True)
        block(seq).square().mean().backward()

    return timer.time(step)


def bench_full_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    context: torch.Tensor,
    target: torch.Tensor,
    timer: BenchTimer,
    *,
    mixed_precision: bool,
) -> dict[str, float]:
    model.train()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with _cuda_autocast(mixed_precision):
            _mark_compiled_cuda_step_begin(model, timer.device)
            prediction = model(context, future_xy=target)
            loss = normalized_mse_loss(prediction, target)
        loss.backward()
        optimizer.step()

    return timer.time(step)


def _default_components_for_model(model: torch.nn.Module) -> list[str]:
    del model
    return ["graph_prep", "attention", "spatial_block", "temporal_block", "full_step"]


def run_component_benchmarks(
    config_arg: str | Path,
    *,
    component: str,
    warmup: int,
    iterations: int,
    device_arg: str | None = None,
    batch_size: int | None = None,
    output_dir: str | Path | None = None,
    overrides: dict[str, object] | None = None,
) -> Path:
    config, config_path = load_config(config_arg)
    if overrides:
        config = apply_overrides(config, overrides)
    if device_arg is not None:
        config = apply_overrides(config, {"experiment.device": device_arg})
    if batch_size is not None:
        config = apply_overrides(config, {"training.batch_size": batch_size})
    config = resolve_config_paths(config, config_path)
    set_seed(config.experiment.seed)
    device = resolve_device(config.experiment.device)
    model_eager = build_model(config).to(device)
    model = _apply_optimization_backend(model_eager, config.model.optimization_backend, device)
    model_for_components = getattr(model, "_orig_mod", model)
    context, target = _synthetic_inputs(config, device)
    timer = BenchTimer(warmup=warmup, iterations=iterations, device=device)
    mixed_precision = _mixed_precision_enabled(device, config.training.mixed_precision)

    spatial_block = model_for_components.spatial_layers[0] if getattr(model_for_components, "spatial_layers", None) else None
    temporal_block = model_for_components.temporal_layers[0] if getattr(model_for_components, "temporal_layers", None) else None
    graph_edge = graph_mask = None
    if hasattr(model_for_components, "graph_inputs"):
        graph_edge, graph_mask = model_for_components.graph_inputs(
            context[:, :, :, :2].reshape(config.training.batch_size * config.experiment.context_size, config.experiment.num_entities, 2),
            context[:, 0, :, -2:]
            .unsqueeze(1)
            .expand(-1, config.experiment.context_size, -1, -1)
            .reshape(config.training.batch_size * config.experiment.context_size, config.experiment.num_entities, 2),
        )
    node_state = torch.randn(
        config.training.batch_size * config.experiment.context_size,
        config.experiment.num_entities,
        config.model.hidden_dim,
        device=device,
    )
    sequence = torch.randn(
        config.training.batch_size * config.experiment.num_entities,
        config.experiment.context_size,
        config.model.hidden_dim,
        device=device,
    )

    requested = _default_components_for_model(model_for_components) if component == "all" else [component]
    results: dict[str, dict[str, float]] = {}
    for item in requested:
        if item == "graph_prep":
            if graph_edge is None or graph_mask is None:
                raise ValueError("graph_prep benchmark requires a model with graph_inputs.")
            results[item] = bench_graph_prep(model_for_components, context, timer)
        elif item == "attention":
            if spatial_block is None:
                raise ValueError("attention benchmark requires a graph-transformer model with spatial layers.")
            assert graph_edge is not None and graph_mask is not None
            results[item] = bench_attention_forward(spatial_block.attention, node_state, graph_edge, graph_mask, timer)
            results["attention_backward"] = bench_attention_backward(spatial_block.attention, node_state, graph_edge, graph_mask, timer)
        elif item == "spatial_block":
            if spatial_block is None:
                raise ValueError("spatial_block benchmark requires spatial layers.")
            results[item] = bench_spatial_block(spatial_block, node_state, graph_edge, graph_mask, timer)
        elif item == "temporal_block":
            if temporal_block is None:
                raise ValueError("temporal_block benchmark requires temporal layers.")
            results[item] = bench_temporal_block(temporal_block, sequence, timer)
        elif item == "full_step":
            optimizer = build_optimizer(config, model, device)
            results[item] = bench_full_step(model, optimizer, context, target, timer, mixed_precision=mixed_precision)
        else:
            raise ValueError("component must be one of: graph_prep, attention, spatial_block, temporal_block, full_step, all.")

    bench_dir = Path(output_dir) if output_dir is not None else Path(config.experiment.output_root) / "benchmarks" / "components"
    bench_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "component": component,
        "config": str(config_arg),
        "device": str(device),
        "warmup": warmup,
        "iterations": iterations,
        "mixed_precision_requested": config.training.mixed_precision,
        "mixed_precision_effective": mixed_precision,
        "component_autocast_enabled": False,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "results": results,
    }
    (bench_dir / "component_benchmarks.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return bench_dir.resolve()


def _run_optimization_suite(
    config_arg: str | Path,
    *,
    variant_registry: dict[str, dict[str, object]],
    default_root: Path,
    variants: list[str] | None = None,
    component_warmup: int = 100,
    component_iterations: int = 500,
    real_warmup_steps: int = 20,
    real_steps: int = 300,
    device_arg: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    output_dir: str | Path | None = None,
    include_real_step: bool = True,
    fail_fast: bool = False,
    capture_compile_logs: bool = False,
    mixed_precision_modes: list[str] | None = None,
    overrides: dict[str, object] | None = None,
) -> Path:
    selected_variants = variants or list(variant_registry)
    unknown = sorted(set(selected_variants) - set(variant_registry))
    if unknown:
        known = ", ".join(variant_registry)
        raise ValueError(f"Unknown benchmark variant(s): {', '.join(unknown)}. Known variants: {known}.")

    root = Path(output_dir) if output_dir is not None else default_root
    root.mkdir(parents=True, exist_ok=True)
    suite_started = time.perf_counter()

    base_overrides = dict(overrides or {})
    if device_arg is not None:
        base_overrides["experiment.device"] = device_arg
    if batch_size is not None:
        base_overrides["training.batch_size"] = batch_size
    if num_workers is not None:
        base_overrides["training.num_workers"] = num_workers
    resolved_mixed_precision_modes = _resolve_mixed_precision_modes(mixed_precision_modes)

    rows: list[dict[str, object]] = []
    for variant in selected_variants:
        for mixed_precision_override in resolved_mixed_precision_modes:
            mixed_precision_mode = _mixed_precision_label(mixed_precision_override)
            variant_run = f"{variant}__mp_{mixed_precision_mode}"
            variant_dir = root / variant / f"mp_{mixed_precision_mode}"
            variant_dir.mkdir(parents=True, exist_ok=True)
            variant_overrides = {**base_overrides, **variant_registry[variant]}
            if mixed_precision_override is not None:
                variant_overrides["training.mixed_precision"] = mixed_precision_override
            row: dict[str, object] = {
                "variant": variant,
                "variant_run": variant_run,
                "mixed_precision_mode": mixed_precision_mode,
                "mixed_precision_requested": variant_overrides.get("training.mixed_precision", "config"),
                "status": "ok",
                "overrides": variant_overrides,
                "override_summary": _format_overrides(variant_overrides),
                "output_dir": str(variant_dir.resolve()),
            }
            print(f"Benchmarking optimization variant: {variant} (mixed_precision={mixed_precision_mode})")
            try:
                with _capture_compile_diagnostics(
                    enabled=capture_compile_logs,
                    variant_dir=variant_dir,
                    variant=variant_run,
                    overrides=variant_overrides,
                ) as compile_log:
                    if compile_log is not None:
                        row["compile_diagnostics"] = str(compile_log)
                    component_dir = run_component_benchmarks(
                        config_arg,
                        component="all",
                        warmup=component_warmup,
                        iterations=component_iterations,
                        output_dir=variant_dir / "components",
                        overrides=variant_overrides,
                    )
                    component_payload = _read_json(component_dir / "component_benchmarks.json")
                    row["component_summary"] = str(component_dir / "component_benchmarks.json")
                    row["mixed_precision_effective"] = component_payload.get("mixed_precision_effective")
                    results = component_payload.get("results", {})
                    if isinstance(results, dict):
                        full_step = results.get("full_step", {})
                        if isinstance(full_step, dict):
                            row["synthetic_full_step_median_ms"] = _optional_float(full_step.get("median_ms"))
                            row["synthetic_full_step_p95_ms"] = _optional_float(full_step.get("p95_ms"))

                    if include_real_step:
                        from .runner import benchmark_experiment

                        benchmark_dir = benchmark_experiment(
                            config_arg,
                            overrides=variant_overrides,
                            warmup_steps=real_warmup_steps,
                            steps=real_steps,
                            profile=False,
                            output_dir=variant_dir / "real_step",
                        )
                        benchmark_payload = _read_json(benchmark_dir / "benchmark_summary.json")
                        row["real_step_summary"] = str(benchmark_dir / "benchmark_summary.json")
                        row["mixed_precision_effective_real"] = benchmark_payload.get("mixed_precision")
                        mean_step_s = _optional_float(benchmark_payload.get("mean_step_s"))
                        if mean_step_s is not None:
                            row["real_mean_step_ms"] = mean_step_s * 1000.0
            except Exception as exc:  # noqa: BLE001 - benchmark suites should report failing variants.
                row["status"] = "failed"
                row["error"] = f"{type(exc).__name__}: {exc}"
                row["traceback"] = traceback.format_exc()
                (variant_dir / "error.txt").write_text(str(row["traceback"]))
                rows.append(row)
                if fail_fast:
                    break
                continue
            rows.append(row)
            if fail_fast:
                break
        if fail_fast and rows and rows[-1].get("status") == "failed":
            break

    payload: dict[str, object] = {
        "config": str(config_arg),
        "device": device_arg or "config",
        "batch_size": batch_size,
        "num_workers": num_workers,
        "component_warmup": component_warmup,
        "component_iterations": component_iterations,
        "include_real_step": include_real_step,
        "real_warmup_steps": real_warmup_steps,
        "real_steps": real_steps,
        "capture_compile_logs": capture_compile_logs,
        "mixed_precision_modes": [_mixed_precision_label(item) for item in resolved_mixed_precision_modes],
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "elapsed_s": time.perf_counter() - suite_started,
        "variants": rows,
    }
    (root / "optimization_benchmark_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_optimization_summary(root, payload)
    print((root / "optimization_benchmark_summary.md").read_text(), end="")
    return root.resolve()


def run_model_optimization_suite(
    config_arg: str | Path,
    *,
    variants: list[str] | None = None,
    component_warmup: int = 100,
    component_iterations: int = 500,
    real_warmup_steps: int = 20,
    real_steps: int = 300,
    device_arg: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    output_dir: str | Path | None = None,
    include_real_step: bool = True,
    fail_fast: bool = False,
    capture_compile_logs: bool = False,
    mixed_precision_modes: list[str] | None = None,
    overrides: dict[str, object] | None = None,
) -> Path:
    return _run_optimization_suite(
        config_arg,
        variant_registry=MODEL_OPTIMIZATION_VARIANTS,
        default_root=Path("runs") / "benchmarks" / "model_optimizations",
        variants=variants,
        component_warmup=component_warmup,
        component_iterations=component_iterations,
        real_warmup_steps=real_warmup_steps,
        real_steps=real_steps,
        device_arg=device_arg,
        batch_size=batch_size,
        num_workers=num_workers,
        output_dir=output_dir,
        include_real_step=include_real_step,
        fail_fast=fail_fast,
        capture_compile_logs=capture_compile_logs,
        mixed_precision_modes=mixed_precision_modes,
        overrides=overrides,
    )
