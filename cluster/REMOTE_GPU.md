# Remote GPU (not Izar)

Most on-demand GPU hosts already run inside a CUDA base container and often include uv. This project uses uv only.

On Linux and macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other systems, see the [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

The examples below assume the remote repository path is `/workspace/nml-project`. Adjust `DEST` and `PROJECT_ROOT` for your host.

From your local machine, sync the prepared project to the remote host:

```bash
SSH_CMD="ssh -i ~/.ssh/remote-gpu -p 12345"
REMOTE="root@1.1.1.1"
DEST="/workspace/nml-project/"

uv sync --extra cpu
source .venv/bin/activate
python -m nba_forecasting prepare-data
rsync -av -e "$SSH_CMD" --delete --exclude-from=.rsyncignore ./ "$REMOTE:$DEST"
```

`prepare-data` creates `data/train_consolidated.pt`, `data/test_consolidated.pt`, and `artifacts/splits/default_split.json`. The single rsync command copies those files plus the code and small CSV files, while `.rsyncignore` skips the thousands of individual sequence files under `data/train/` and `data/test/`.

On the remote host, create the environment once. Pick the environment that matches the CUDA/PyTorch version you want:

```bash
cd /workspace/nml-project
UV_EXTRA=cu124 bash cluster/create_env.sh
```

Interactive work:

```bash
cd /workspace/nml-project
source .venv/bin/activate
python -m nba_forecasting doctor --expect-cuda
```

Available Torch extras are defined in `pyproject.toml`: `cpu`, `cu121`, `cu124`, and `cu128`.
