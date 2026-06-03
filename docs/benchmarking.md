# Benchmarking

Use the benchmark commands when comparing model throughput or validating graph Transformer optimizations. They are not required for normal training or tuning.

## Single Config

```bash
python -m nba_forecasting benchmark \
  --config e5_graph_transformer \
  --device cuda \
  --batch-size 192 \
  --num-workers 4 \
  --warmup-steps 5 \
  --steps 30
```

Add `--profile` to write a PyTorch profiler Chrome trace and operator table under `runs/benchmarks/`.

## Graph Transformer Optimizations

The optimized graph Transformer preset is:

```bash
python -m nba_forecasting benchmark \
  --config e5_graph_transformer_compile_fused_prep \
  --device cuda \
  --batch-size 192
```

That config enables:

```text
model.optimization_backend = "compile"
model.graph_prep_backend = "fused"
```

To compare retained variants:

```bash
python -m nba_forecasting bench-transformer-optimizations \
  --variants baseline,compile,compile_fused_graph_prep \
  --device cuda \
  --batch-size 192
```

The matrix writes `optimization_benchmark_summary.json`, `.md`, and `.csv` under the output directory.

## Izar

Submit the Slurm benchmark job:

```bash
sbatch cluster/izar_bench_graph_transformer.slurm
```

For a short debug run:

```bash
COMPONENT_WARMUP=2 COMPONENT_ITERATIONS=5 INCLUDE_REAL_STEP=0 \
  sbatch cluster/izar_bench_graph_transformer.slurm
```

To benchmark only selected variants:

```bash
VARIANTS=baseline,compile,compile_fused_graph_prep \
  sbatch cluster/izar_bench_graph_transformer.slurm
```

The job records failed variants instead of stopping by default. Set `FAIL_FAST=1` when debugging a specific failure.
