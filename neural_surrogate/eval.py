"""Evaluation helpers for neural surrogate models."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def unpack_batch(
    batch: Mapping[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(x, y)`` from common dataloader batch formats."""
    if isinstance(batch, Mapping):
        return batch["x"], batch["y"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Batch must be a mapping with x/y or a tuple/list of (x, y).")


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = unpack_batch(batch)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module | None = None,
    device: torch.device | str | None = None,
) -> dict[str, float]:
    """Evaluate a model and return average loss plus MSE."""
    device = device or next(model.parameters()).device
    loss_fn = loss_fn or nn.MSELoss()
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_examples = 0

    for batch in dataloader:
        x, y = move_batch_to_device(batch, device)
        prediction = model(x)
        y = _match_target_shape(y, prediction)

        batch_size = x.shape[0]
        total_loss += float(loss_fn(prediction, y).item()) * batch_size
        total_mse += float(nn.functional.mse_loss(prediction, y).item()) * batch_size
        total_examples += batch_size

    if was_training:
        model.train()

    if total_examples == 0:
        return {"loss": 0.0, "mse": 0.0}
    return {
        "loss": total_loss / total_examples,
        "mse": total_mse / total_examples,
    }


@torch.no_grad()
def predict(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Run model inference without changing the caller's training mode."""
    device = device or next(model.parameters()).device
    was_training = model.training
    model.eval()
    prediction = model(x.to(device))
    if was_training:
        model.train()
    return prediction


def _match_target_shape(y: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    y = y.to(dtype=prediction.dtype)
    if y.shape == prediction.shape:
        return y
    if y.ndim == 1 and prediction.ndim == 2 and prediction.shape[-1] == 1:
        return y.unsqueeze(-1)
    if y.ndim == 2 and y.shape[-1] == 1 and prediction.ndim == 1:
        return y.squeeze(-1)
    return y


__all__ = ["evaluate", "move_batch_to_device", "predict", "unpack_batch"]
