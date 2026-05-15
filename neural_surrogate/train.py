"""Training helpers for neural surrogate models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import logging
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
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
    rank: int = 0


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
    val_loader: DataLoader,
    config: TrainConfig,
    topology_strategy: str = "hashing",
    parameter_strategy: str = "bounds",
) -> TrainHistory:
    """Fit a model and optionally evaluate/checkpoint after each epoch."""
    device = torch.device(config.device)
    model.to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    kwargs = {
        "topology_strategy": topology_strategy,
        "parameter_strategy": parameter_strategy,
    }

    history = TrainHistory()
    best_val_loss = float("inf")

    for _ in range(config.epochs):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(_)
        if config.rank == 0:
            print("=" * 50)
            train_logger.info(f"Starting epoch {_+1}/{config.epochs}...")
        train_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            grad_clip_norm=config.grad_clip_norm,
        )
        if config.rank == 0:
            train_logger.info(f"Epoch {_+1} train loss: {train_loss:.6f}")
        history.train_loss.append(train_loss)

        if val_loader is None:
            continue

        if config.rank == 0:
            train_logger.info("\nRunning validation")
        metrics = evaluate(model, val_loader, loss_fn=loss_fn, device=device)
        if config.rank == 0:
            train_logger.info(f"Epoch {_+1} validation loss: {metrics['loss']:.6f}")
        history.val_loss.append(metrics["loss"])

        if (
            config.rank == 0
            and config.checkpoint_path is not None
            and metrics["loss"] < best_val_loss
        ):
            best_val_loss = metrics["loss"]
            train_logger.info("\nNew best validation loss found. Saving checkpoint.")
            save_checkpoint(model, config.checkpoint_path, kwargs=kwargs)

    return history


def save_checkpoint(model: nn.Module, path: str | Path, kwargs: dict) -> None:
    path = Path(path) / (
        f"{kwargs.get('topology_strategy', 'hashing')}_{kwargs.get('parameter_strategy', 'bounds')}_checkpoint.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    model_to_save = model.module if isinstance(model, DistributedDataParallel) else model
    torch.save(model_to_save.state_dict(), path)
    train_logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    model: nn.Module,
    path: str | Path,
    device: torch.device | str | None = None,
) -> nn.Module:
    device = device or next(model.parameters()).device
    state_dict = torch.load(path, map_location=device)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    state_dict = {
        key.removeprefix("module."): value for key, value in state_dict.items()
    }
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
