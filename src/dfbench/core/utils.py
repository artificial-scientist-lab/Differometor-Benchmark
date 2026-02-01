"""Utility functions for dfbench.

Provides torch<->jax conversion utilities and other helpers.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Float


def t2j(tensor: torch.Tensor) -> jax.Array:
    """Convert torch tensor to JAX array via NumPy."""
    return jax.numpy.asarray(tensor.detach().cpu().numpy())


def j2t(arr: jax.Array) -> torch.Tensor:
    """Convert JAX array to torch tensor via NumPy.

    Creates a writable copy to avoid PyTorch warnings about non-writable arrays.
    """
    return torch.from_numpy(np.array(arr))


def inverse_sigmoid_bounding(
    bounded_params: Float[jax.Array, " N"],
    bounds: Float[jax.Array, "2 N"],
) -> Float[jax.Array, " N"]:
    """Map bounded parameters back to unbounded space (inverse of sigmoid_bounding).

    This is the inverse of differometor.utils.sigmoid_bounding, which maps
    unbounded params to bounded ones via: sigmoid(x) * (upper - lower) + lower.

    Args:
        bounded_params: Parameters in bounded space [lower, upper].
        bounds: Shape (2, n_params) array where bounds[0] is lower and bounds[1] is upper.

    Returns:
        Unbounded parameters suitable for use with sigmoid_objective_function.
    """
    normalised = (bounded_params - bounds[0]) / (bounds[1] - bounds[0])
    normalised = jnp.clip(normalised, 1e-7, 1.0 - 1e-7)
    return jnp.log(normalised / (1.0 - normalised))
