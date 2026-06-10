# Remote GPU (not Izar)

Most on-demand GPU hosts already run inside a CUDA base container and often include uv. This project uses uv only.

## Install uv

On Linux and macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other systems, see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

The examples below assume the remote repository path is `/workspace/nml-project`. Adjust `DEST` and `PROJECT_ROOT` for your host.

## Sync the project

Prepare portable artifacts locally first; see [local.md](local.md#prepare-portable-artifacts).

From your local machine:

```bash
SSH_CMD="ssh -i ~/.ssh/remote-gpu -p 12345"
REMOTE="root@1.1.1.1"
DEST="/workspace/nml-project/"

uv sync --extra cpu
source .venv/bin/activate
python -m nba_forecasting prepare-data
rsync -av -e "$SSH_CMD" --delete --exclude-from=.rsyncignore ./ "$REMOTE:$DEST"
```

`.rsyncignore` skips the thousands of individual sequence files under `data/train/` and `data/test/` while copying consolidated artifacts and code.

## Create the environment

On the remote host, create the environment once. Pick the extra that matches the CUDA/PyTorch version you want:

```bash
cd /workspace/nml-project
UV_EXTRA=cu124 bash cluster/create_env.sh
```

`cluster/create_env.sh` is the generic bootstrap script (no Slurm modules). Available Torch extras are defined in `pyproject.toml`: `cpu`, `cu121`, `cu124`, and `cu128`.

## Interactive work

```bash
cd /workspace/nml-project
source .venv/bin/activate
python -m nba_forecasting doctor --expect-cuda
```

For SCITAS Izar with Slurm, use [izar.md](izar.md) instead.
