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
model.graph_cache_backend = "reuse"
```

Global training configs request `torch.compile`, mixed precision, and fused AdamW by default. The runner enables those only on CUDA; CPU and MPS runs fall back to eager execution, fp32, and standard AdamW.

To compare retained variants:

```bash
python -m nba_forecasting bench-model-optimizations \
  --config e5_graph_transformer \
  --variants baseline,compile,compile_fused_graph_prep,compile_fused_graph_prep_graph_cache_fused_adamw \
  --device cuda \
  --batch-size 192 \
  --mixed-precision-modes false,true
```

The matrix writes `optimization_benchmark_summary.json`, `.md`, and `.csv` under the output directory. Mixed precision is an explicit matrix dimension; summaries report both the requested mode and the effective mode because autocast only takes effect on CUDA.

Add `--capture-compile-logs` when benchmarking compile variants to save Dynamo graph-break and recompile diagnostics under each compile variant:

```text
<output-dir>/<variant>/compile_diagnostics/torch_compile_stderr.log
<output-dir>/<variant>/compile_diagnostics/compile_diagnostics.json
```

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
VARIANTS=baseline,compile,compile_fused_graph_prep,compile_fused_graph_prep_graph_cache_fused_adamw \
  sbatch cluster/izar_bench_graph_transformer.slurm
```

The job records failed variants instead of stopping by default. Set `FAIL_FAST=1` when debugging a specific failure. Set `MIXED_PRECISION_MODES=false,true` to compare fp32 and CUDA autocast; this is the default for model optimization matrices.

To specifically check whether `torch.compile`/Inductor can run on the CUDA device:

```bash
python -m nba_forecasting doctor --expect-cuda --check-inductor
```
