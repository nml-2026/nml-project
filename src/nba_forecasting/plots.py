from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch


def _color_for_team(team_value: float) -> str:
    if team_value == 0:
        return "#111111"
    if team_value < 0:
        return "#1e88e5"
    return "#d32f2f"


def save_history_plot(history: list[dict[str, float]], output_path: str | Path) -> None:
    if not history:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_ade_l2 = [row.get("val_ade_l2") for row in history]
    val_fde_l2 = [row.get("val_fde_l2") for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].set_title("Normalized Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(epochs, val_ade_l2, label="val_ade_l2")
    axes[1].plot(epochs, val_fde_l2, label="val_fde_l2")
    axes[1].set_title("Validation Error (court feet)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_per_step_error_plot(per_step_errors: list[float], output_path: str | Path) -> None:
    if not per_step_errors:
        return
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    steps = list(range(1, len(per_step_errors) + 1))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, per_step_errors, marker="o", label="ADE L2")
    ax.set_title("Per-Step Error (court feet)")
    ax.set_xlabel("Horizon step")
    ax.set_ylabel("Mean L2 displacement")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def save_prediction_plot(
    output_path: str | Path,
    context_xy: torch.Tensor,
    target_xy: torch.Tensor,
    pred_xy: torch.Tensor,
    static: torch.Tensor,
    title: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for entity_idx in range(context_xy.shape[1]):
        team_value = float(static[entity_idx, 1].item())
        color = _color_for_team(team_value)
        context = context_xy[:, entity_idx].cpu()
        future = target_xy[:, entity_idx].cpu()
        pred = pred_xy[:, entity_idx].cpu()
        ax.plot(context[:, 0], context[:, 1], color=color, alpha=0.3, linewidth=1.5)
        ax.plot(future[:, 0], future[:, 1], color=color, linestyle="--", linewidth=1.5)
        ax.plot(pred[:, 0], pred[:, 1], color=color, linestyle=":", linewidth=2.0)
        ax.scatter(context[-1, 0], context[-1, 1], color=color, s=18)
        ax.scatter(future[-1, 0], future[-1, 1], color=color, marker="x", s=28)
        ax.scatter(pred[-1, 0], pred[-1, 1], color=color, marker="o", facecolors="none", s=28)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
