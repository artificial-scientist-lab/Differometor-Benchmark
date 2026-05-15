"""Gradient-based inverse design over continuous surrogate parameters."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from .encodings import (
        CampaignSample,
        flatten_parameters,
        make_campaign_dataset,
        pad_or_truncate,
        select_loss,
    )
    from .model import TransformerEncoderConfig, TransformerEncoderSurrogate
    from .pipeline import find_h5_files, resolve_device
    from .train import load_checkpoint
except ImportError:  # Allows running this file directly.
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from neural_surrogate.encodings import (
        CampaignSample,
        flatten_parameters,
        make_campaign_dataset,
        pad_or_truncate,
        select_loss,
    )
    from neural_surrogate.model import TransformerEncoderConfig, TransformerEncoderSurrogate
    from neural_surrogate.pipeline import find_h5_files, resolve_device
    from neural_surrogate.train import load_checkpoint


@dataclass(frozen=True)
class InverseDesignResult:
    source: str
    predicted_loss: float
    objective: float
    parameters: list[float]


def run_inverse_design(
    data_path: str | Path,
    checkpoint_path: str | Path,
    *,
    topology_dim: int = 128,
    topology_strategy: str = "hashing",
    parameter_strategy: str = "bounds",
    loss_key: str = "loss_senspow",
    target_loss: float | None = None,
    reference_index: int | None = None,
    steps: int = 1000,
    lr: float = 1e-2,
    device: str = "auto",
    output_path: str | Path | None = None,
) -> InverseDesignResult:
    train_device = resolve_device(device)
    h5_files = find_h5_files(data_path)
    dataset = make_campaign_dataset(
        h5_files,
        topology_strategy=topology_strategy,
        parameter_strategy=parameter_strategy,
        topology_dim=topology_dim,
        loss_key=loss_key,
    )
    if len(dataset) == 0:
        raise RuntimeError("No samples found to fit encoders for inverse design.")

    reference = choose_reference_sample(
        dataset.samples,
        loss_key=loss_key,
        reference_index=reference_index,
    )
    encoder = dataset.encoder
    model = build_surrogate(dataset.encoder.input_dim).to(train_device)
    load_checkpoint(model, checkpoint_path, device=train_device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    topology = encoder.topology_encoder.encode(reference.setup_graph).to(train_device)
    initial_parameters = pad_or_truncate(
        flatten_parameters(reference.best_parameters),
        encoder.parameter_encoder.output_dim,
    )
    optimizer_variable = make_optimizer_variable(
        encoder.parameter_encoder,
        initial_parameters,
        train_device,
    )
    optimizer = torch.optim.Adam([optimizer_variable], lr=lr)

    best_objective = float("inf")
    best_encoded: torch.Tensor | None = None
    best_prediction = float("inf")

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        encoded_parameters = variable_to_encoded_parameters(
            optimizer_variable,
            encoder.parameter_encoder,
        )
        prediction = predict_from_encoded(model, topology, encoded_parameters)
        objective = inverse_objective(prediction, target_loss)
        objective.backward()
        optimizer.step()

        objective_value = float(objective.detach().cpu().item())
        if objective_value < best_objective:
            best_objective = objective_value
            best_encoded = encoded_parameters.detach().cpu()
            best_prediction = float(prediction.detach().cpu().item())

    if best_encoded is None:
        raise RuntimeError("Inverse design did not run any optimization steps.")

    raw_parameters = decode_parameters(best_encoded, encoder.parameter_encoder)
    result = InverseDesignResult(
        source=reference.source,
        predicted_loss=best_prediction,
        objective=best_objective,
        parameters=[float(value) for value in raw_parameters.tolist()],
    )
    if output_path is not None:
        write_result(result, output_path)
    return result


def build_surrogate(input_dim: int) -> TransformerEncoderSurrogate:
    return TransformerEncoderSurrogate(
        TransformerEncoderConfig(
            input_dim=input_dim,
            output_dim=1,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            norm_first=False,
        )
    )


def choose_reference_sample(
    samples: list[CampaignSample],
    *,
    loss_key: str,
    reference_index: int | None,
) -> CampaignSample:
    if reference_index is not None:
        return samples[reference_index]
    return min(samples, key=lambda sample: select_loss(sample.losses, loss_key))


def make_optimizer_variable(
    parameter_encoder: Any,
    initial_parameters: torch.Tensor,
    device: torch.device,
) -> torch.nn.Parameter:
    initial_parameters = initial_parameters.to(torch.float32)
    if parameter_encoder.strategy == "bounds":
        if parameter_encoder.low is None or parameter_encoder.high is None:
            raise RuntimeError("Bounds encoder is missing fitted low/high values.")
        encoded = (initial_parameters - parameter_encoder.low) / (
            parameter_encoder.high - parameter_encoder.low
        ).clamp_min(1e-8)
    elif parameter_encoder.strategy == "standard":
        if parameter_encoder.mean is None or parameter_encoder.std is None:
            raise RuntimeError("Standard encoder is missing fitted mean/std values.")
        encoded = (initial_parameters - parameter_encoder.mean) / parameter_encoder.std
    else:
        encoded = initial_parameters

    encoded = encoded.to(device)
    if parameter_encoder.strategy == "bounds":
        encoded = encoded.clamp(1e-4, 1.0 - 1e-4)
        encoded = torch.logit(encoded)
    return torch.nn.Parameter(encoded.clone().detach())


def variable_to_encoded_parameters(
    variable: torch.Tensor,
    parameter_encoder: Any,
) -> torch.Tensor:
    if parameter_encoder.strategy == "bounds":
        return torch.sigmoid(variable)
    return variable


def predict_from_encoded(
    model: torch.nn.Module,
    topology: torch.Tensor,
    encoded_parameters: torch.Tensor,
) -> torch.Tensor:
    x = torch.cat([topology, encoded_parameters], dim=0).unsqueeze(0)
    return model(x).reshape(())


def inverse_objective(prediction: torch.Tensor, target_loss: float | None) -> torch.Tensor:
    if target_loss is None:
        return prediction
    target = prediction.new_tensor(float(target_loss))
    return F.mse_loss(prediction, target)


def decode_parameters(encoded_parameters: torch.Tensor, parameter_encoder: Any) -> torch.Tensor:
    encoded_parameters = encoded_parameters.detach().cpu()
    if parameter_encoder.strategy == "bounds":
        if parameter_encoder.low is None or parameter_encoder.high is None:
            raise RuntimeError("Bounds encoder is missing fitted low/high values.")
        low = parameter_encoder.low.cpu()
        high = parameter_encoder.high.cpu()
        return low + encoded_parameters * (high - low)
    if parameter_encoder.strategy == "standard":
        if parameter_encoder.mean is None or parameter_encoder.std is None:
            raise RuntimeError("Standard encoder is missing fitted mean/std values.")
        return encoded_parameters * parameter_encoder.std.cpu() + parameter_encoder.mean.cpu()
    return encoded_parameters


def write_result(result: InverseDesignResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": result.source,
        "predicted_loss": result.predicted_loss,
        "objective": result.objective,
        "parameters": result.parameters,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize continuous inputs for a fixed surrogate."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--loss-key", default="loss_senspow")
    parser.add_argument("--target-loss", type=float, default=None)
    parser.add_argument("--reference-index", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--topology-dim", type=int, default=128)
    parser.add_argument(
        "--topology-strategy",
        default="hashing",
        choices=("hashing", "vocabulary", "exact"),
    )
    parser.add_argument(
        "--parameter-strategy",
        default="bounds",
        choices=("identity", "standard", "bounds"),
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("neural_surrogate/inverse_design_result.json"),
    )
    args = parser.parse_args()

    result = run_inverse_design(
        args.data,
        args.checkpoint_path,
        topology_dim=args.topology_dim,
        topology_strategy=args.topology_strategy,
        parameter_strategy=args.parameter_strategy,
        loss_key=args.loss_key,
        target_loss=args.target_loss,
        reference_index=args.reference_index,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        output_path=args.output_path,
    )
    print(f"source: {result.source}")
    print(f"predicted_loss: {result.predicted_loss:.9g}")
    print(f"objective: {result.objective:.9g}")
    print(f"parameters_written: {args.output_path}")


if __name__ == "__main__":
    main()


__all__ = ["InverseDesignResult", "run_inverse_design"]
