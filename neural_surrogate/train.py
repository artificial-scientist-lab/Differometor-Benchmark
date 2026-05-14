"""Training helpers for neural surrogate models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
import logging
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from .eval import evaluate, move_batch_to_device
except ImportError:  # Allows running this file directly during quick experiments.
    from eval import evaluate, move_batch_to_device

train_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@dataclass
class TrainConfig:
    """Basic training-loop settings."""

    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float | None = 1.0
    device: str | torch.device = "cpu"
    checkpoint_path: str | Path | None = None


@dataclass
class TrainHistory:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_mse: list[float] = field(default_factory=list)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module | None = None,
    device: torch.device | str | None = None,
    grad_clip_norm: float | None = None,
) -> float:
    """Train for one epoch and return mean loss."""
    device = device or next(model.parameters()).device
    loss_fn = loss_fn or nn.MSELoss()
    model.train()

    total_loss = 0.0
    total_examples = 0
    for batch in dataloader:
        x, y = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        y = _match_target_shape(y, prediction)
        loss = loss_fn(prediction, y)
        loss.backward()

        if grad_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None = None,
    config: TrainConfig | None = None,
    loss_fn: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> TrainHistory:
    """Fit a model and optionally evaluate/checkpoint after each epoch."""
    config = config or TrainConfig()
    device = torch.device(config.device)
    model.to(device)
    loss_fn = loss_fn or nn.MSELoss()
    optimizer = optimizer or torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    history = TrainHistory()
    best_val_loss = float("inf")

    for _ in range(config.epochs):
        train_logger.info(f"Starting epoch {_+1}/{config.epochs}...")
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        history.train_loss.append(train_loss)

        if val_loader is None:
            continue

        train_logger.info("Running validation")
        metrics = evaluate(model, val_loader, loss_fn=loss_fn, device=device)
        history.val_loss.append(metrics["loss"])
        history.val_mse.append(metrics["mse"])

        if config.checkpoint_path is not None and metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            train_logger.info("New best validation loss found. Saving checkpoint.")
            save_checkpoint(model, config.checkpoint_path)

    return history


def save_checkpoint(model: nn.Module, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    train_logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str | None = None,
) -> nn.Module:
    device = device or next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    return model


def _match_target_shape(y: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
    y = y.to(dtype=prediction.dtype)
    if y.shape == prediction.shape:
        return y
    if y.ndim == 1 and prediction.ndim == 2 and prediction.shape[-1] == 1:
        return y.unsqueeze(-1)
    if y.ndim == 2 and y.shape[-1] == 1 and prediction.ndim == 1:
        return y.squeeze(-1)
    return y


__all__ = [
    "TrainConfig",
    "TrainHistory",
    "fit",
    "load_checkpoint",
    "save_checkpoint",
    "train_epoch",
]
