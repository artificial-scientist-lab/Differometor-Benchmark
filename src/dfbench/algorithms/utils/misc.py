from __future__ import annotations

import numpy as np
import torch
from jaxtyping import Float
from botorch.utils.transforms import unnormalize

from dfbench.core.objective import Objective
from dfbench.core.utils import t2j

def evaluate_y(
        X: Float[torch.Tensor, "... d"],
        bounds: Float[torch.Tensor, "2 d"],
        obj: Objective,
        max_retries: int = 3,
        perturbation_scale: float = 1e-6,
    ) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]:
        """Evaluate objective function at given input(s) through Objective wrapper.

        If NaN/Inf is encountered, attempts to perturb the point and retry.
        Returns both values and a validity mask so invalid points can be
        filtered from GP training data.

        Args:
            X: Input(s) in normalized [0,1] space.
            bounds: Original bounds for unnormalization.
            obj: Objective wrapper for evaluation tracking.
            max_retries: Number of retries with perturbation for NaN values.
            perturbation_scale: Scale of random perturbation for retries.

        Returns:
            Tuple of (negated objective values, validity mask).
            Invalid entries have NaN values and False in mask.
        """
        unnormalized_X = unnormalize(X, bounds)
        X_jax = t2j(unnormalized_X)

        # Handle batch dimension - use Objective for tracking
        if X_jax.ndim == 1:
            Y_jax = obj.value(X_jax)
            Y_torch = torch.tensor([Y_jax.item()], device=X.device, dtype=X.dtype)
        else:
            Y_jax = obj.vmap_value(X_jax)
            Y_torch = torch.from_numpy(np.array(Y_jax)).to(
                device=X.device, dtype=X.dtype
            )

        # Track validity
        invalid_mask = torch.isnan(Y_torch) | torch.isinf(Y_torch)

        # Retry invalid points with small perturbations
        if torch.any(invalid_mask) and max_retries > 0:
            invalid_indices = torch.where(invalid_mask)[0]
            print(
                f"Warning: {len(invalid_indices)} NaN/Inf values detected, retrying with perturbation..."
            )

            for idx in invalid_indices:
                for retry in range(max_retries):
                    X_perturbed = X[idx].clone()
                    perturbation = (
                        torch.randn_like(X_perturbed) * perturbation_scale * (retry + 1)
                    )
                    X_perturbed = torch.clamp(X_perturbed + perturbation, 0.0, 1.0)

                    unnorm_perturbed = unnormalize(X_perturbed, bounds)
                    X_jax_perturbed = t2j(unnorm_perturbed)
                    Y_retry = obj.value(X_jax_perturbed)
                    Y_retry_torch = torch.tensor(
                        Y_retry.item(), device=X.device, dtype=X.dtype
                    )

                    if torch.isfinite(Y_retry_torch):
                        Y_torch[idx] = Y_retry_torch
                        invalid_mask[idx] = False
                        break

            remaining_invalid = torch.sum(invalid_mask).item()
            if remaining_invalid > 0:
                print(
                    f"Warning: {remaining_invalid} points still invalid after retries"
                )

        valid_mask = ~invalid_mask
        return -Y_torch, valid_mask  # Negate for maximization