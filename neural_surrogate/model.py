"""Small transformer encoder surrogate model.

The model treats each scalar input feature as a token, lets a single
``nn.TransformerEncoder`` attend across those tokens, then predicts from a
pooled representation with a feedforward head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TransformerEncoderConfig:
    """Configuration for :class:`TransformerEncoderSurrogate`."""

    input_dim: int
    output_dim: int = 1
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    norm_first: bool = True


class TransformerEncoderSurrogate(nn.Module):
    """Simple feedforward surrogate with transformer encoder attention.

    Args:
        config: Model dimensions and transformer settings.

    Input shape:
        ``x`` can be ``[input_dim]`` for one sample or ``[batch, input_dim]``.

    Output shape:
        ``[output_dim]`` for one sample or ``[batch, output_dim]``.
    """

    def __init__(self, config: TransformerEncoderConfig) -> None:
        super().__init__()
        if config.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if config.output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if config.d_model % config.nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.config = config

        self.value_projection = nn.Sequential(
            nn.Linear(1, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model),
        )
        self.feature_embedding = nn.Parameter(
            torch.zeros(config.input_dim, config.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=config.norm_first,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.d_model),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, config.output_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        unbatched = x.ndim == 1
        if unbatched:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"x must have shape [batch, {self.config.input_dim}] "
                f"or [{self.config.input_dim}]."
            )

        tokens = self.value_projection(x.unsqueeze(-1))
        tokens = tokens + self.feature_embedding.unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        output = self.head(pooled)
        return output.squeeze(0) if unbatched else output


NeuralSurrogate = TransformerEncoderSurrogate


__all__ = [
    "NeuralSurrogate",
    "TransformerEncoderConfig",
    "TransformerEncoderSurrogate",
]
