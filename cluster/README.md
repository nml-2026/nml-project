# Izar Slurm scripts

| Script | Purpose |
|--------|---------|
| `izar_smoke.slurm` | Quick GPU smoke test |
| `izar_model_comparison.slurm` | Model-comparison array (1 GPU × job; see `model-comparison` CLI) |
| `izar_train.slurm` | `train` for one or more configs |
| `izar_tune.slurm` | Ray Tune for one or more configs |
| `izar_submit.slurm` | Kaggle submission from checkpoint or Tune result |
| `izar_env.sh` | Shared module/venv setup sourced by the jobs above |
| `izar_create_env.sh` | Izar environment bootstrap (CUDA modules + uv) |
| `create_env.sh` | Generic uv bootstrap for any host |

Full setup, `sbatch` examples, and environment variables: [docs/setup/izar.md](../docs/setup/izar.md).

For non-Izar GPU hosts, see [docs/setup/remote-gpu.md](../docs/setup/remote-gpu.md).
