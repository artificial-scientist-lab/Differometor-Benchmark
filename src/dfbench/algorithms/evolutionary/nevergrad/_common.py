"""Shared helpers for Nevergrad-based algorithm wrappers.

The Nevergrad ask/tell loop expects ``optimizer.tell(candidate, finite_loss)``.
When the underlying Objective returns NaN or Inf — which is by design for
some Differometor problem geometries — we apply an escalating perturbation
to the candidate and re-evaluate until a finite loss is obtained.

The schedule mirrors the Optax loop: start at 1e-10 and double each miss,
falling back to a large penalty after ``_MAX_NAN_STREAK`` failures so the
optimizer can still ``tell`` and continue exploring.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dfbench.core.objective import Objective


_NAN_PERTURB_BASE: float = 1e-10
_MAX_NAN_STREAK: int = 20
_NAN_PENALTY: float = 1e30


def safe_evaluate(
    obj: Objective,
    params: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, np.ndarray]:
    """Evaluate ``obj.value(params)`` with a NaN/Inf escape hatch.

    On non-finite output the candidate is perturbed in-place by Gaussian
    noise whose scale starts at 1e-10 and doubles per attempt. The perturbed
    candidate is clipped to ``[lb, ub]`` so it remains feasible for the
    bounded Nevergrad parametrisation. After ``_MAX_NAN_STREAK`` failures we
    return a large finite penalty so the optimizer can still record the
    point and move on.

    Args:
        obj: The pre-configured Objective.
        params: Initial candidate (already inside the box).
        lb, ub: Box bounds.
        rng: NumPy generator used for the perturbations.

    Returns:
        ``(finite_loss, candidate_used)``.
    """
    params_jax = jnp.asarray(params, dtype=jnp.float32)
    loss = obj.value(params_jax)
    if bool(jnp.isfinite(loss)):
        return float(loss), params

    cur = np.asarray(params, dtype=np.float64)
    for k in range(_MAX_NAN_STREAK):
        if obj.budget_exceeded:
            break
        scale = _NAN_PERTURB_BASE * (2**k)
        perturbed = np.clip(cur + rng.normal(size=cur.shape) * scale, lb, ub)
        loss = obj.value(jnp.asarray(perturbed, dtype=jnp.float32))
        if bool(jnp.isfinite(loss)):
            return float(loss), perturbed.astype(params.dtype)

    return _NAN_PENALTY, params
