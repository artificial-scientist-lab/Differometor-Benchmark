from __future__ import annotations

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "torch is required for this algorithm. Install with:  uv add 'dfbench[bo]'"
    ) from exc
from typing import Literal


def create_initial_design(
    dimensions: int,
    n_initial: int,
    random_seed: int | None = 42,
    sampler: Literal["sobol", "uniform", "prior"] = "sobol",
) -> torch.Tensor:
    match sampler:
        case "sobol":
            # Generate initial Sobol samples
            sobol = torch.quasirandom.SobolEngine(
                dimension=dimensions, scramble=True, seed=random_seed
            )
            train_X = sobol.draw(n=n_initial)

        case "uniform":
            # Generate initial uniform random samples
            train_X = torch.rand(
                size=(n_initial, dimensions),
                generator=torch.Generator().manual_seed(random_seed),
            )

        case "prior":
            # Sample from a provided prior distribution
            raise NotImplementedError("Prior sampling not implemented yet.")

    return train_X
