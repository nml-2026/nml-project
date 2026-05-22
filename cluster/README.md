# Running on Izar

## First-Time Setup

From the local repository root:

```bash
rsync -av --delete --exclude-from=.rsyncignore ./ izar:~/nml-project/
```

Then log in to Izar and create the module-backed virtual environment:

```bash
ssh izar
cd ~/nml-project
mkdir -p logs
bash cluster/izar_create_module_venv.sh
```

## Smoke Test

```bash
sbatch cluster/izar_smoke_gpu.slurm
squeue -u "$USER"
tail -f logs/nba-smoke-gpu-<jobid>.out
```

## Ray Tune

Quick test run:

```bash
SEARCH_PROFILE=graph_ssm NUM_SAMPLES=2 MAX_CONCURRENT_TRIALS=2 EPOCHS=1 MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=1 \
  sbatch cluster/izar_tune_single_node.slurm
```

Runtime-estimation sanity run:

```bash
CONFIGS="e5_graph_transformer e5_graph_transformer_no_graph e5_graph_transformer_shuffled_graph" \
SEARCH_PROFILE=graph_transformer NUM_SAMPLES=2 MAX_CONCURRENT_TRIALS=2 EPOCHS=2 \
ESTIMATE_TARGET_SAMPLES=4 ESTIMATE_TARGET_EPOCHS=35 \
  sbatch --time=03:00:00 cluster/izar_tune_model_ladder.slurm
```

This runs full train/validation batches for two epochs, so the printed runtime projection is more useful than a batch-limited smoke test. The estimate is still approximate because ASHA can stop different trials at different epochs.

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

Available search profiles are `basic`, `graph`, `graph_fixed_message`, `graph_ssm`, and `graph_transformer`. Use `graph_fixed_message` when the config should keep its message-passing type fixed while tuning graph capacity.

Summarize completed Tune and batch jobs:

```bash
PYTHONPATH=src python -m nba_forecasting summarize-results \
  --root /scratch/izar/$USER/nba_forecasting \
  --format markdown \
  --output /scratch/izar/$USER/nba_forecasting/results.md
```

Export a reusable best-trial config:

```bash
PYTHONPATH=src python -m nba_forecasting export-best-config \
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

The scripts assume the EE-452 GPU allocation and write large run outputs to `/scratch/izar/$USER/nba_forecasting/<jobid>`. Config names are normal full-run defaults; quick tests use runtime overrides such as `EPOCHS`, `MAX_TRAIN_BATCHES`, and `MAX_VAL_BATCHES`. Override script defaults with environment variables such as `CONFIG`, `CONFIGS`, `SEARCH_PROFILE`, `NUM_SAMPLES`, `SEEDS`, `BATCH_SIZE`, `CHECKPOINT`, `SUMMARY`, `OUTPUT`, `PRINT_RUNTIME_ESTIMATE`, `ESTIMATE_TARGET_SAMPLES`, and `ESTIMATE_TARGET_EPOCHS`.
