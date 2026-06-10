# SCITAS Izar

Slurm scripts live under `cluster/`. Configure jobs with `CONFIG`, `CONFIGS`, and other environment variables.

## Script layout

| Script               | Purpose                                                      |
| -------------------- | ------------------------------------------------------------ |
| `create_env.sh`      | Generic uv bootstrap (any host)                              |
| `izar_create_env.sh` | Izar wrapper: loads CUDA modules, then calls `create_env.sh` |
| `izar_env.sh`        | Shared module/venv/runtime setup sourced by Slurm jobs       |
| `izar_smoke.slurm`   | Quick GPU smoke test (1 epoch, capped batches)               |
| `izar_train.slurm`   | Full training via `train` (one or more configs)              |
| `izar_tune.slurm`    | Ray Tune search (one or more configs)                        |
| `izar_submit.slurm`  | Kaggle submission from checkpoint or latest Tune result      |

## First-time setup

Install uv on Izar if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Prepare portable artifacts locally first; see [local.md](local.md#prepare-portable-artifacts).

From the local repository root:

```bash
uv sync --extra cpu
source .venv/bin/activate
python -m nba_forecasting prepare-data
rsync -av --delete --exclude-from=.rsyncignore ./ izar:~/nml-project/
```

The rsync ignore rules skip the thousands of individual `data/train/*.pt` and `data/test/*.pt` files while copying consolidated artifacts.

Then log in to Izar and create the environment:

```bash
ssh izar
cd ~/nml-project
mkdir -p logs
bash cluster/izar_create_env.sh
```

`cluster/izar_create_env.sh` loads `gcc/11.3.0 cuda/12.1.1` and installs the uv Torch extra `cu121`. To use a different CUDA/PyTorch extra, add it to `pyproject.toml` and set `UV_EXTRA` when creating the environment.

For interactive work on a login node or inside an allocation:

```bash
cd ~/nml-project
module purge
module load gcc/11.3.0 cuda/12.1.1
source .venv/bin/activate
python -m nba_forecasting doctor --expect-cuda
```

## Smoke test

```bash
sbatch cluster/izar_smoke.slurm
squeue -u "$USER"
tail -f logs/nba-smoke-<jobid>.out
```

Override the config or batch caps:

```bash
CONFIG=baseline EPOCHS=1 MAX_TRAIN_BATCHES=4 MAX_VAL_BATCHES=2 sbatch cluster/izar_smoke.slurm
```

## Model comparison

Pre-flight on one GPU before launching the array:

```bash
cd ~/nml-project
source cluster/izar_env.sh
load_izar_modules
activate_izar_env
configure_izar_runtime
python -m nba_forecasting model-comparison --smoke
```

Full comparison batch (10 independent array tasks):

```bash
sbatch cluster/izar_model_comparison.slurm
squeue -u "$USER"
```

Collect metrics on Izar after jobs finish:

```bash
python3 scripts/collect_results.py --report-table-only
```

## Email notifications

Email notifications are disabled by default in the Slurm scripts. To opt in for one submission, pass Slurm mail options on the `sbatch` command:

```bash
sbatch \
  --mail-user=first.last@epfl.ch \
  --mail-type=END,FAIL \
  cluster/izar_tune.slurm
```

Common notification types are `BEGIN`, `END`, `FAIL`, `REQUEUE`, `TIME_LIMIT`, and `ALL`.

## Ray Tune

Quick test run:

```bash
CONFIG=graph_transformer SEARCH_PROFILE=graph_transformer NUM_SAMPLES=2 MAX_CONCURRENT_TRIALS=2 \
  EPOCHS=1 MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=1 \
  sbatch cluster/izar_tune.slurm
```

The Tune script defaults to explicit local Ray resources for a 2-GPU Izar node: `RAY_NUM_CPUS=40`, `RAY_NUM_GPUS=2`, `CPUS_PER_TRIAL=19`, `GPUS_PER_TRIAL=1`, and `MAX_CONCURRENT_TRIALS=2`. Ray temp files and Tune's own storage use `$TMPDIR`; persistent training outputs and summaries stay under `RUN_ROOT`. Set `TRACE_TRAINABLE=1` to write per-trial startup/runtime timing traces under the Tune output directory.

Single-config search:

```bash
CONFIG=dynamic_graph_direct SEARCH_PROFILE=graph NUM_SAMPLES=16 MAX_CONCURRENT_TRIALS=2 \
  sbatch cluster/izar_tune.slurm
```

Candidate ladder (multiple configs in one job; auto-summarizes at the end):

```bash
CONFIGS="dynamic_graph_direct graph_transformer" \
SEARCH_PROFILE=graph_transformer NUM_SAMPLES=8 MAX_CONCURRENT_TRIALS=2 \
  sbatch --time=11:45:00 cluster/izar_tune.slurm
```

Graph Transformer matched ablation run:

```bash
CONFIGS="graph_transformer" \
SEARCH_PROFILE=graph_transformer NUM_SAMPLES=8 MAX_CONCURRENT_TRIALS=2 \
EXTRA_SEARCH="model.graph_transformer.graph_ablation_mode=choice(normal,no_graph,shuffled_graph)" \
  sbatch cluster/izar_tune.slurm
```

Runtime-estimation sanity run:

```bash
CONFIGS="graph_transformer" \
SEARCH_PROFILE=graph_transformer NUM_SAMPLES=2 MAX_CONCURRENT_TRIALS=2 EPOCHS=2 \
ESTIMATE_TARGET_SAMPLES=4 ESTIMATE_TARGET_EPOCHS=35 \
  sbatch --time=03:00:00 cluster/izar_tune.slurm
```

Runtime estimates are only printed when both `ESTIMATE_TARGET_SAMPLES` and `ESTIMATE_TARGET_EPOCHS` are set. The estimate is still approximate because ASHA can stop different trials at different epochs.

For an interactive debug allocation, run the same script body by hand after:

```bash
srun --partition=gpu --qos=ee-452 --account=ee-452 \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 \
  --mem=64G --time=01:00:00 --pty bash

cd ~/nml-project
source cluster/izar_env.sh
load_izar_modules
activate_izar_env
configure_izar_runtime
check_izar_python
export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp/$USER}/torch_extensions"
```

Available search profiles are `graph`, `graph_fixed_message`, `graph_transformer`, `mart`, `aux_losses`, and `lr_scheduler`. Use `graph_fixed_message` when the config should keep its message-passing type fixed while tuning graph capacity.

Summarize completed Tune and batch jobs:

```bash
python -m nba_forecasting summarize-results \
  --root /scratch/izar/$USER/nba_forecasting \
  --format markdown \
  --output /scratch/izar/$USER/nba_forecasting/results.md
```

Export a reusable best-trial config:

```bash
python -m nba_forecasting export-best-config \
  --summary /scratch/izar/$USER/nba_forecasting/<jobid>/ray_tune/<config>/<timestamp>/ray_tune_summary.json \
  --output configs/tuned_<config>.toml
```

## Kaggle submission

Generate a submission from the latest Ray Tune result for a config:

```bash
CONFIG=dynamic_graph_direct sbatch cluster/izar_submit.slurm
```

Or generate from a specific checkpoint:

```bash
CHECKPOINT=/scratch/izar/$USER/nba_forecasting/<jobid>/ray_tune/<config>/<timestamp>/trial_runs/<run>/best.pt \
  sbatch cluster/izar_submit.slurm
```

Download the generated CSV from the path printed in the job log.

## Batch training

Default ladder:

```bash
sbatch cluster/izar_train.slurm
```

Custom config list or single model:

```bash
CONFIGS="baseline dynamic_graph_direct graph_transformer" \
SEEDS="0" EPOCHS=30 \
  sbatch cluster/izar_train.slurm

CONFIG=graph_transformer SEEDS="0 1 2" EPOCHS=50 sbatch cluster/izar_train.slurm
```

## Environment variables

The scripts assume the EE-452 GPU allocation, use the uv Torch extra `cu121`, and write large run outputs to `/scratch/izar/$USER/nba_forecasting/<jobid>`.

| Variable                                             | Scripts                    | Meaning                                                 |
| ---------------------------------------------------- | -------------------------- | ------------------------------------------------------- |
| `CONFIG`                                             | smoke, tune, train, submit | Single config stem (without `.toml`)                    |
| `CONFIGS`                                            | tune, train                | Space-separated config list; overrides `CONFIG` default |
| `SEARCH_PROFILE`                                     | tune                       | Ray Tune search space profile                           |
| `NUM_SAMPLES`                                        | tune                       | Number of Tune trials per config                        |
| `MAX_CONCURRENT_TRIALS`                              | tune                       | Parallel trials on the node                             |
| `SEEDS`                                              | train                      | Space-separated training seeds                          |
| `EPOCHS`                                             | smoke, tune, train         | Training epoch override                                 |
| `BATCH_SIZE`                                         | train                      | Batch size override                                     |
| `MAX_TRAIN_BATCHES` / `MAX_VAL_BATCHES`              | smoke, tune                | Cap batches for quick tests                             |
| `CHECKPOINT` / `SUMMARY`                             | submit                     | Explicit artifact paths                                 |
| `RUN_ROOT`                                           | tune, train                | Persistent output directory                             |
| `ESTIMATE_TARGET_SAMPLES` / `ESTIMATE_TARGET_EPOCHS` | tune                       | Post-run runtime estimate knobs                         |
| `EXTRA_SET` / `EXTRA_SEARCH`                         | tune                       | Extra `--set` / `--search` overrides                    |
