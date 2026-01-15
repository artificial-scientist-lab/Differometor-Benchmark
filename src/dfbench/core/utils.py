import jax
import numpy as np
import torch
from jaxtyping import Array


def t2j(a: Array) -> jax.Array:
    """WARNING: This could be extremely slow somehow. Use t2j_numpy instead for now.
    Convert torch array to jax array.
    """
    return jax.dlpack.from_dlpack(a)


def j2t(a: Array) -> torch.Tensor:
    """WARNING: This could be extremely slow somehow. Use j2t_numpy instead for now.
    Convert torch array to jax array.
    """
    return torch.utils.dlpack.from_dlpack(a)


def t2j_numpy(tensor: torch.Tensor) -> jax.Array:
    """Convert torch tensor to JAX array via NumPy."""
    return jax.numpy.asarray(tensor.detach().cpu().numpy())


def j2t_numpy(arr: jax.Array) -> torch.Tensor:
    """Convert JAX array to torch tensor via NumPy.

    Creates a writable copy to avoid PyTorch warnings about non-writable arrays.
    """
    return torch.from_numpy(np.array(arr))
