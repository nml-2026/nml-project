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

```bash
NUM_SAMPLES=2 MAX_CONCURRENT_TRIALS=2 EPOCHS=1 MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=1 \
  sbatch cluster/izar_tune_single_node.slurm
```

```bash
CONFIG=e1_social_ssm NUM_SAMPLES=16 MAX_CONCURRENT_TRIALS=2 \
  sbatch cluster/izar_tune_single_node.slurm
```

```bash
CONFIGS="e3_dynamic_graph_direct e1_social_ssm e3_graph_ssm_no_social" \
NUM_SAMPLES=8 MAX_CONCURRENT_TRIALS=2 \
  sbatch cluster/izar_tune_model_ladder.slurm
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

The scripts assume the EE-452 GPU allocation and write large run outputs to `/scratch/izar/$USER/nba_forecasting/<jobid>`. Override script defaults with environment variables such as `CONFIG`, `CONFIGS`, `NUM_SAMPLES`, `SEEDS`, `EPOCHS`, and `BATCH_SIZE`.
