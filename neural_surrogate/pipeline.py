"""End-to-end training pipeline for campaign H5 surrogate data."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
import logging

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore", message="CUDA initialization:.*")

pipeline_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from .encodings import make_campaign_dataset
    from .eval import evaluate, predict
    from .model import TransformerEncoderConfig, TransformerEncoderSurrogate
    from .train import TrainConfig, fit
except ImportError:  # Allows running this file directly.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from neural_surrogate.encodings import make_campaign_dataset
    from neural_surrogate.eval import evaluate, predict
    from neural_surrogate.model import TransformerEncoderConfig, TransformerEncoderSurrogate
    from neural_surrogate.train import TrainConfig, fit


def find_h5_files(data_path: str | Path) -> list[Path]:
    path = Path(data_path)
    if path.is_file():
        return [path]

    files = sorted(path.glob("*.h5"))
    if files:
        return files

    fallback = sorted(path.parent.glob("*.h5"))
    if fallback:
        return fallback

    raise FileNotFoundError(f"No .h5 files found in {path} or {path.parent}.")


def run_pipeline(
    data_path: str | Path,
    *,
    loss_key: str = "loss_senspow",
    epochs: int = 250,
    batch_size: int = 8,
    lr: float = 1e-3,
    topology_dim: int = 128,
    seed: int = 0,
    val_fraction: float = 0.2,
    device: str = "auto",
    multi_gpu: str = "off",
    checkpoint_path: str | Path | None = None,
    topology_strategy: str = "hashing",
    parameter_strategy: str = "bounds",
    dataset_workers: int = 0,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 2,
    dim_feedforward: int = 128,
    max_input_dim: int | None = None,
) -> dict[str, float]:
    torch.manual_seed(seed)
    ddp = multi_gpu == "ddp"
    rank, local_rank, world_size = setup_distributed(ddp)
    train_device = resolve_device(device, local_rank=local_rank if ddp else None)
    is_main = rank == 0

    h5_files = find_h5_files(data_path)

    if is_main:
        pipeline_logger.info(f"Found {len(h5_files)} H5 file(s) for training.")
    dataset = make_campaign_dataset(
        h5_files,
        topology_strategy=topology_strategy,
        parameter_strategy=parameter_strategy,
        topology_dim=topology_dim,
        loss_key=loss_key,
        num_workers=dataset_workers,
    )
    if len(dataset) == 0:
        raise RuntimeError("No trainable samples found in the H5 campaign data.")
    if max_input_dim is not None and dataset.encoder.input_dim > max_input_dim:
        raise RuntimeError(
            f"Encoded input_dim={dataset.encoder.input_dim} exceeds "
            f"--max-input-dim={max_input_dim}. Reduce --topology-dim, use "
            "hashing/vocabulary, or increase the limit if you have enough memory."
        )
    
    if is_main:
        pipeline_logger.info(
            "Created dataset containing %s samples with input dimension %s.",
            len(dataset),
            dataset.encoder.input_dim,
        )

    train_dataset, eval_dataset = split_dataset(
        dataset,
        seed=seed,
        val_fraction=val_fraction,
    )
    train_sampler = (
        DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        if ddp
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model = TransformerEncoderSurrogate(
        TransformerEncoderConfig(
            input_dim=dataset.encoder.input_dim,
            output_dim=1,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            norm_first=False,
        )
    )
    model.to(train_device)
    model = maybe_parallelize_model(
        model,
        train_device,
        multi_gpu,
        local_rank=local_rank,
    )

    try:
        fit(
            model=model,
            train_loader=train_loader,
            val_loader=eval_loader,
            topology_strategy=topology_strategy,
            parameter_strategy=parameter_strategy,
            config=TrainConfig(
                epochs=epochs,
                lr=lr,
                grad_clip_norm=1.0,
                device=train_device,
                checkpoint_path=checkpoint_path,
                rank=rank,
            ),
        )
        if ddp:
            dist.barrier()

        if not is_main:
            return {}

        eval_model = model.module if isinstance(model, DistributedDataParallel) else model
        metrics = evaluate(eval_model, eval_loader, device=train_device)

        first = dataset[0]
        predicted = float(
            predict(eval_model, first["x"].unsqueeze(0), device=train_device).item()
        )
        target = float(first["y"].item())
        absolute_error = abs(predicted - target)

        return {
            "samples": float(len(dataset)),
            "train_samples": float(len(train_dataset)),
            "val_samples": float(len(eval_dataset)),
            "input_dim": float(dataset.encoder.input_dim),
            "gpu_count": float(world_size if ddp else int(train_device.type == "cuda")),
            "target": target,
            "prediction": predicted,
            "absolute_error": absolute_error,
            "eval_loss": metrics["loss"],
            "topology_strategy": topology_strategy,
            "parameter_strategy": parameter_strategy,
        }
    finally:
        if ddp and dist.is_initialized():
            dist.destroy_process_group()


def setup_distributed(enabled: bool) -> tuple[int, int, int]:
    if not enabled:
        return 0, 0, 1
    if not torch.cuda.is_available():
        raise RuntimeError("DDP requires CUDA, but torch.cuda.is_available() is false.")
    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            "DDP must be launched with torchrun; missing environment "
            f"variables: {missing}"
        )
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def resolve_device(device: str, local_rank: int | None = None) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda", local_rank or 0)
        return torch.device("cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    if resolved.type == "cuda" and local_rank is not None:
        return torch.device("cuda", local_rank)
    return resolved


def maybe_parallelize_model(
    model: torch.nn.Module,
    device: torch.device,
    multi_gpu: str,
    local_rank: int = 0,
) -> torch.nn.Module:
    if multi_gpu == "off":
        return model
    if multi_gpu != "ddp":
        raise ValueError("multi_gpu must be 'off' or 'ddp'.")
    if device.type != "cuda":
        raise RuntimeError("--multi-gpu ddp requires --device cuda or auto CUDA.")
    return DistributedDataParallel(model, device_ids=[local_rank])


def split_dataset(
    dataset: torch.utils.data.Dataset,
    *,
    seed: int,
    val_fraction: float = 0.2,
) -> tuple[object, object]:
    if len(dataset) == 1:
        return dataset, dataset
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1.")

    eval_size = max(1, int(round(val_fraction * len(dataset))))
    train_size = len(dataset) - eval_size
    if eval_size == 0:
        train_size -= 1
        eval_size = 1
    if train_size == 0:
        train_size = 1
        eval_size = len(dataset) - 1

    return random_split(
        dataset,
        [train_size, eval_size],
        generator=torch.Generator().manual_seed(seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate the H5 loss surrogate.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="H5 file or directory containing campaign .h5 files.",
    )
    parser.add_argument("--loss-key", default="loss_senspow")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--topology-dim", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument(
        "--max-input-dim",
        type=int,
        default=None,
        help="Abort early if encoded input dimension is too large for memory.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of encoded samples held out for validation.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--multi-gpu",
        default="off",
        choices=("off", "ddp"),
        help="Use DistributedDataParallel. Launch with torchrun.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="If specified, saves the best model checkpoint to this path.",
    )
    parser.add_argument(
        "--topology-strategy",
        default="hashing",
        choices=("hashing", "vocabulary", "exact"),
        help="Strategy for encoding topology information.",
    )
    parser.add_argument(
        "--parameter-strategy",
        default="bounds",
        choices=("identity", "standard", "bounds"),
        help="Strategy for encoding parameter information.",
    )
    parser.add_argument(
        "--dataset-workers",
        type=int,
        default=0,
        help="Parallel H5 loading workers. Use -1 to auto-use available CPUs.",
    )
    args = parser.parse_args()

    if isinstance(args.data, str):
        args.data = Path(args.data)
    if isinstance(args.checkpoint_path, str):
        args.checkpoint_path = Path(args.checkpoint_path)

    metrics = run_pipeline(
        args.data,
        loss_key=args.loss_key,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        topology_dim=args.topology_dim,
        seed=args.seed,
        val_fraction=args.val_fraction,
        device=args.device,
        multi_gpu=args.multi_gpu,
        checkpoint_path=args.checkpoint_path,
        topology_strategy=args.topology_strategy,
        parameter_strategy=args.parameter_strategy,
        dataset_workers=args.dataset_workers,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        max_input_dim=args.max_input_dim,
    )
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.9g}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()


__all__ = [
    "find_h5_files",
    "maybe_parallelize_model",
    "resolve_device",
    "run_pipeline",
    "setup_distributed",
    "split_dataset",
]
