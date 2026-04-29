from __future__ import annotations

import torch


def normalized_mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2)


def per_example_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((prediction - target) ** 2, dim=(1, 2, 3))


def compute_raw_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    diff = prediction - target
    mse = torch.mean(diff**2).item()
    ade = torch.mean(torch.sum(diff**2, dim=-1)).item()
    rmse = mse**0.5
    return {
        "mse": mse,
        "ade": ade,
        "rmse": rmse,
    }
