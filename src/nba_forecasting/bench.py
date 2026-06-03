from __future__ import annotations

import json
import math
import statistics
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import torch

from .config import RunConfig, apply_overrides, load_config, resolve_config_paths
from .models import build_model
from .runner import _apply_optimization_backend, _cuda_autocast, _mixed_precision_enabled, normalized_mse_loss, resolve_device, set_seed


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


GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS: dict[str, dict[str, object]] = {
    "baseline": {},
    "compile": {"model.optimization_backend": "compile"},
    "compile_fused_graph_prep": {
        "model.optimization_backend": "compile",
        "model.graph_prep_backend": "fused",
    },
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


def _write_optimization_summary(output_dir: Path, payload: dict[str, object]) -> None:
    variants = payload["variants"]
    assert isinstance(variants, list)
    baseline = next((row for row in variants if isinstance(row, dict) and row.get("variant") == "baseline"), None)
    baseline_synthetic = _optional_float(baseline.get("synthetic_full_step_median_ms")) if isinstance(baseline, dict) else None
    baseline_real = _optional_float(baseline.get("real_mean_step_ms")) if isinstance(baseline, dict) else None

    table_rows: list[list[str]] = []
    csv_rows = ["variant,status,synthetic_full_step_median_ms,synthetic_speedup_pct,real_mean_step_ms,real_speedup_pct,error"]
    for raw in variants:
        assert isinstance(raw, dict)
        synthetic = _optional_float(raw.get("synthetic_full_step_median_ms"))
        real = _optional_float(raw.get("real_mean_step_ms"))
        synthetic_speedup = _ratio_speedup(baseline_synthetic, synthetic)
        real_speedup = _ratio_speedup(baseline_real, real)
        error = str(raw.get("error", ""))
        table_rows.append(
            [
                str(raw["variant"]),
                str(raw["status"]),
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
            "# Graph Transformer Optimization Benchmark",
            "",
            f"Config: `{payload['config']}`",
            f"Device: `{payload['device']}`",
            f"Batch size: `{payload['batch_size']}`",
            f"Component iterations: `{payload['component_iterations']}`",
            f"Real dataloader steps: `{payload['real_steps'] if payload['include_real_step'] else 'disabled'}`",
            "",
            _markdown_table(
                [
                    "variant",
                    "status",
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

    return timer.time(step)


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
            prediction = model(context, future_xy=target)
            loss = normalized_mse_loss(prediction, target)
        loss.backward()
        optimizer.step()

    return timer.time(step)


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
    model = _apply_optimization_backend(build_model(config).to(device), config.model.optimization_backend)
    model_for_components = getattr(model, "_orig_mod", model)
    context, target = _synthetic_inputs(config, device)
    timer = BenchTimer(warmup=warmup, iterations=iterations, device=device)
    mixed_precision = _mixed_precision_enabled(device, config.training.mixed_precision)

    spatial_block = model_for_components.spatial_layers[0] if getattr(model_for_components, "spatial_layers", None) else None
    temporal_block = model_for_components.temporal_layers[0] if getattr(model_for_components, "temporal_layers", None) else None
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

    requested = ["graph_prep", "attention", "spatial_block", "temporal_block", "full_step"] if component == "all" else [component]
    results: dict[str, dict[str, float]] = {}
    for item in requested:
        if item == "graph_prep":
            results[item] = bench_graph_prep(model_for_components, context, timer)
        elif item == "attention":
            if spatial_block is None:
                raise ValueError("attention benchmark requires a graph-transformer model with spatial layers.")
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
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay)
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
        "results": results,
    }
    (bench_dir / "component_benchmarks.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return bench_dir.resolve()


def run_graph_transformer_optimization_suite(
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
    overrides: dict[str, object] | None = None,
) -> Path:
    selected_variants = variants or list(GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS)
    unknown = sorted(set(selected_variants) - set(GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS))
    if unknown:
        known = ", ".join(GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS)
        raise ValueError(f"Unknown benchmark variant(s): {', '.join(unknown)}. Known variants: {known}.")

    root = Path(output_dir) if output_dir is not None else Path("runs") / "benchmarks" / "graph_transformer_optimizations"
    root.mkdir(parents=True, exist_ok=True)
    suite_started = time.perf_counter()

    base_overrides = dict(overrides or {})
    if device_arg is not None:
        base_overrides["experiment.device"] = device_arg
    if batch_size is not None:
        base_overrides["training.batch_size"] = batch_size
    if num_workers is not None:
        base_overrides["training.num_workers"] = num_workers

    rows: list[dict[str, object]] = []
    for variant in selected_variants:
        variant_dir = root / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_overrides = {**base_overrides, **GRAPH_TRANSFORMER_OPTIMIZATION_VARIANTS[variant]}
        row: dict[str, object] = {
            "variant": variant,
            "status": "ok",
            "overrides": variant_overrides,
            "override_summary": _format_overrides(variant_overrides),
            "output_dir": str(variant_dir.resolve()),
        }
        print(f"Benchmarking graph Transformer optimization variant: {variant}")
        try:
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
        "elapsed_s": time.perf_counter() - suite_started,
        "variants": rows,
    }
    (root / "optimization_benchmark_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_optimization_summary(root, payload)
    print((root / "optimization_benchmark_summary.md").read_text(), end="")
    return root.resolve()
