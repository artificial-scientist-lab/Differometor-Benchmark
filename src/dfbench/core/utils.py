"""Utility functions for dfbench.

Provides torch<->jax conversion utilities and other helpers.
"""

import jax
import numpy as np
import torch


def t2j(tensor: torch.Tensor) -> jax.Array:
    """Convert torch tensor to JAX array via NumPy."""
    return jax.numpy.asarray(tensor.detach().cpu().numpy())


def j2t(arr: jax.Array) -> torch.Tensor:
    """Convert JAX array to torch tensor via NumPy.

    Creates a writable copy to avoid PyTorch warnings about non-writable arrays.
    """
    return torch.from_numpy(np.array(arr))
