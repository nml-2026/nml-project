# Running on Izar

## First-Time Setup

Install uv on Izar if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For Windows and other systems, see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

From the local repository root:

```bash
uv sync --extra cpu
source .venv/bin/activate
python -m nba_forecasting prepare-data
rsync -av --delete --exclude-from=.rsyncignore ./ izar:~/nml-project/
```

The prep step creates `data/train_consolidated.pt`, `data/test_consolidated.pt`, and `artifacts/splits/default_split.json`. The rsync ignore rules skip the thousands of individual `data/train/*.pt` and `data/test/*.pt` files while copying those consolidated artifacts.

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

## Smoke Test

```bash
sbatch cluster/izar_smoke_gpu.slurm
squeue -u "$USER"
tail -f logs/nba-smoke-gpu-<jobid>.out
```

## Email Notifications

Email notifications are disabled by default in the Slurm scripts. To opt in for one submission, pass Slurm mail options on the `sbatch` command:

```bash
sbatch \
  --mail-user=first.last@epfl.ch \
  --mail-type=END,FAIL \
  cluster/izar_tune_single_node.slurm
```

Common notification types are `BEGIN`, `END`, `FAIL`, `REQUEUE`, `TIME_LIMIT`, and `ALL`.

## Ray Tune

Quick test run:

```bash
SEARCH_PROFILE=graph_ssm NUM_SAMPLES=2 MAX_CONCURRENT_TRIALS=2 EPOCHS=1 MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=1 \
  sbatch cluster/izar_tune_single_node.slurm
```

The single-node Tune script defaults to explicit local Ray resources for a 2-GPU Izar node: `RAY_NUM_CPUS=40`, `RAY_NUM_GPUS=2`, `CPUS_PER_TRIAL=19`, `GPUS_PER_TRIAL=1`, and `MAX_CONCURRENT_TRIALS=2`. Ray temp files and Tune's own storage use `$TMPDIR`; persistent training outputs and summaries stay under `RUN_ROOT`. Set `TRACE_TRAINABLE=1` to write per-trial startup/runtime timing traces under the Tune output directory.

Runtime-estimation sanity run:

```bash
CONFIGS="e5_graph_transformer e5_graph_transformer_no_graph e5_graph_transformer_shuffled_graph" \
SEARCH_PROFILE=graph_transformer NUM_SAMPLES=2 MAX_CONCURRENT_TRIALS=2 EPOCHS=2 \
ESTIMATE_TARGET_SAMPLES=4 ESTIMATE_TARGET_EPOCHS=35 \
  sbatch --time=03:00:00 cluster/izar_tune_model_ladder.slurm
```

Runtime estimates are only printed when both `ESTIMATE_TARGET_SAMPLES` and `ESTIMATE_TARGET_EPOCHS` are set. The estimate is still approximate because ASHA can stop different trials at different epochs.

Single-config search:

```bash
CONFIG=e3_dynamic_graph_direct SEARCH_PROFILE=graph NUM_SAMPLES=16 MAX_CONCURRENT_TRIALS=2 \
  sbatch cluster/izar_tune_single_node.slurm
```

Candidate ladder:

```bash
CONFIGS="e3_dynamic_graph_direct e3_graph_ssm e3_graph_ssm_no_social e3_dynamic_graph_mp_ssm" \
SEARCH_PROFILE=graph_ssm NUM_SAMPLES=8 MAX_CONCURRENT_TRIALS=2 \
  sbatch cluster/izar_tune_model_ladder.slurm
```

Graph Transformer matched ablation run:

```bash
CONFIGS="e5_graph_transformer e5_graph_transformer_no_graph e5_graph_transformer_shuffled_graph" \
SEARCH_PROFILE=graph_transformer NUM_SAMPLES=8 MAX_CONCURRENT_TRIALS=2 \
  sbatch cluster/izar_tune_model_ladder.slurm
```

Benchmarking commands, including `cluster/izar_bench_graph_transformer.slurm`, are documented in [../docs/benchmarking.md](../docs/benchmarking.md).

For an interactive debug allocation, run the same script body by hand after:

```bash
srun --partition=gpu --qos=ee-452 --account=ee-452 \
  --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 \
  --mem=64G --time=01:00:00 --pty bash

cd ~/nml-project
module purge
module load gcc/11.3.0 cuda/12.1.1
source .venv/bin/activate
python -m nba_forecasting doctor --expect-cuda
export TORCH_EXTENSIONS_DIR="${TMPDIR:-/tmp/$USER}/torch_extensions"
```

Available search profiles are `basic`, `graph`, `graph_fixed_message`, `graph_ssm`, and `graph_transformer`. Use `graph_fixed_message` when the config should keep its message-passing type fixed while tuning graph capacity.

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

## Kaggle Submission

Generate a submission from the latest Ray Tune result for a config:

```bash
CONFIG=e3_dynamic_graph_direct sbatch cluster/izar_submit.slurm
```

Or generate from a specific checkpoint:

```bash
CHECKPOINT=/scratch/izar/$USER/nba_forecasting/<jobid>/ray_tune/<config>/<timestamp>/trial_runs/<run>/best.pt \
  sbatch cluster/izar_submit.slurm
```

Download the generated CSV from the path printed in the job log.

## Batch Training

```bash
CONFIGS="e0_reference e1_joint_gru e1_social_ssm e3_dynamic_graph_direct e3_graph_ssm_no_social" \
SEEDS="0" EPOCHS=30 \
  sbatch cluster/izar_train_batch.slurm
```

## Notes

The scripts assume the EE-452 GPU allocation, use the uv Torch extra `cu121`, and write large run outputs to `/scratch/izar/$USER/nba_forecasting/<jobid>`. Config names are normal full-run defaults; quick tests use runtime overrides such as `EPOCHS`, `MAX_TRAIN_BATCHES`, and `MAX_VAL_BATCHES`. Override script defaults with environment variables such as `CONFIG`, `CONFIGS`, `SEARCH_PROFILE`, `NUM_SAMPLES`, `SEEDS`, `BATCH_SIZE`, `CHECKPOINT`, `SUMMARY`, `OUTPUT`, `ESTIMATE_TARGET_SAMPLES`, and `ESTIMATE_TARGET_EPOCHS`.
