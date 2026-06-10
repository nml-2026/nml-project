# Experiments

Supported configs, the model-comparison CLI, and Ray Tune sweeps.

## Configs

| Config | Model family |
| --- | --- |
| `baseline` | Per-entity GRU |
| `joint_gru` | Joint GRU |
| `static_graph` | Static full-graph MPNN |
| `dynamic_graph_autoregressive` | Dynamic MPNN autoregressive |
| `dynamic_graph_direct` | Dynamic MPNN direct decoder |
| `graph_transformer` | Graph Transformer delta head |
| `graph_transformer_mdn` | Graph Transformer MDN head ($K{=}3$) |
| `mart` | MART with reflect xy ($p{=}0.5$ each) and agent permutation ($p{=}0.5$); softmin-raw recipe in `mart.toml` |
| `mart_aug_none` | MART ablation: no train-time reflection or agent permutation |
| `mart_aug_reflect` | MART ablation: reflect xy only |
| `mart_aug_perm` | MART ablation: agent permutation $p{=}0.5$ only |
| `tune_graph_transformer` | Ray Tune base for graph Transformer |
| `tune_mart` | Ray Tune base for MART |

All runnable configs inherit shared defaults from `base.toml`:

- `features.reflection`: $x$ and $y$ mirrors, each with probability $0.5$
- `features.augmentation.agent_permutation_prob = 0.5`
- Training without TTA; model-comparison evaluates xy TTA explicitly after training

## Scripts

| Script | Purpose |
| --- | --- |
| `run_tune.sh` | Ray Tune sweeps for graph Transformer and MART |
| `collect_results.py` | Aggregate metrics and print report-table numbers |
| `demo.sh` | Doctor + prepare-data + 5-epoch bounded train (requires `data/`) |

Model-comparison training uses the CLI (`model-comparison` subcommand). Everything else uses `train`, `evaluate`, `submit`, etc. directly.

## Model comparison

Protocol:

1. Train on the frozen 80/10/10 split with early stopping on validation $\mathrm{ADE}_{\mathrm{sq}}$
2. Report the **best validation checkpoint**, not a fixed final epoch
3. Cap training at **80 epochs** (patience 80) for ladder and graph Transformer; **100 epochs** (patience 100) for MART
4. Train with `evaluation.tta_group=none`, then evaluate val/holdout with and without full-group xy TTA

```bash
# 1. Pre-flight (~15 min on 1 GPU)
python -m nba_forecasting model-comparison --smoke

# 2. Full batch (default: all 10 jobs)
python -m nba_forecasting model-comparison
python -m nba_forecasting model-comparison --gpus 0,1,2,3

# Subset
python -m nba_forecasting model-comparison --jobs static_graph,graph_transformer --gpus 0
python -m nba_forecasting model-comparison --skip-jobs mart --gpus 0,1,2

# 3. Collect
python3 scripts/collect_results.py --report-table-only
```

On Izar: `sbatch cluster/izar_model_comparison.slurm` (array 0–9). Run `--smoke` first.

Single job: `python -m nba_forecasting model-comparison --job graph_transformer`

Output:

| Path | Contents |
| --- | --- |
| `runs/model_comparison/` | Ladder, GT delta, GT MDN, MART, optional MART aug ablations |
| `runs/connectivity_ablations/` | `no_graph` and `shuffled_graph` ablations |

Collect metrics:

```bash
python3 scripts/collect_results.py
python3 scripts/collect_results.py --report-table-only
```

Writes `report/generated/model_comparison.{md,tex}` and prints holdout+xy-TTA numbers for the report table.

## MART augmentation ablations

Optional model-comparison jobs for Table~3 in the report. Same softmin-raw recipe and 100-epoch budget as `mart`; only train-time augmentation differs.

| Config | Reflection ($x$, $y$ each $p{=}0.5$) | Agent permutation |
| --- | ---: | ---: |
| `mart_aug_none` | off | off |
| `mart_aug_reflect` | on | off |
| `mart_aug_perm` | off | $p{=}0.5$ |
| `mart` | on | $p{=}0.5$ |

```bash
python -m nba_forecasting model-comparison --gpus 0,1,2 \
  --jobs mart_aug_none,mart_aug_reflect,mart_aug_perm
```

## CLI workflows

Single-model training:

```bash
python -m nba_forecasting train --config graph_transformer_mdn --device cuda
python -m nba_forecasting train --config mart --device cuda --epochs 100
```

Post-train evaluation:

```bash
python -m nba_forecasting evaluate --checkpoint runs/.../best.pt --split holdout --tta xy --device cuda
```

Kaggle submission:

```bash
python -m nba_forecasting submit --checkpoint runs/.../best.pt --output submissions/predictions.csv --tta xy --device cuda
```

## MART scope

MART rows in the model-comparison table compare against the published hypergraph architecture ([arXiv:2407.21635](https://arxiv.org/abs/2407.21635)), not a model we designed. Our contribution is the **integration and training recipe**: NBA feature adapter, softmin-raw multi-mode loss, type-conditioned decoder, learned-softmax mode aggregation, Ray Tune hyperparameters in `mart.toml`, and evaluation under the shared protocol above. See [models.md](models.md#mart) for the full split between prior work and our changes.

## Ray Tune

```bash
GPUS=0,1 NUM_SAMPLES=24 bash scripts/run_tune.sh
TARGET=mart GPUS=0 bash scripts/run_tune.sh
```

MART sweeps tune architecture capacity (`mart_function_type`, widths, layers) and the softmin-raw loss (`loss.sample_loss_type`, `loss.softmin_temperature`, `loss.raw_loss_scale_ft`) on top of the ported encoder.

After a sweep:

```bash
python -m nba_forecasting summarize-results --format markdown
python -m nba_forecasting export-best-config --summary <path>/ray_tune_summary.json
```

For Izar-specific launch details, see [setup/izar.md](setup/izar.md).
