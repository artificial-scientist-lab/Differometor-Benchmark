"""Shared utilities for SciPy-based derivative-free optimizers.

Provides numpy<->JAX wrappers, bounds conversion, and budget-aware
callbacks so individual algorithm modules stay DRY.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.optimize import Bounds

from dfbench.core.objective import Objective


class SciPyBudgetExceeded(RuntimeError):
    """Raised inside a SciPy objective/callback when the Objective budget is spent."""


# Large but finite penalty returned when the objective produces NaN/Inf.
# Using ``np.inf`` causes SciPy's derivative-free solvers (Nelder-Mead,
# Powell, basin-hopping, dual-annealing) to terminate prematurely or crash.
_NAN_PENALTY = 1e30


# ---------------------------------------------------------------------------
# Bounds helpers
# ---------------------------------------------------------------------------


def scipy_bounds(obj: Objective) -> Bounds:
    """Convert problem bounds to :class:`scipy.optimize.Bounds`."""
    lb = np.asarray(obj.problem.bounds[0], dtype=np.float64)
    ub = np.asarray(obj.problem.bounds[1], dtype=np.float64)
    return Bounds(lb, ub)


def scipy_bounds_list(obj: Objective) -> list[tuple[float, float]]:
    """Return bounds as ``[(lb0, ub0), ...]`` (format used by *dual_annealing*)."""
    lb = np.asarray(obj.problem.bounds[0], dtype=np.float64)
    ub = np.asarray(obj.problem.bounds[1], dtype=np.float64)
    return list(zip(lb.tolist(), ub.tolist()))


# ---------------------------------------------------------------------------
# Objective wrappers
# ---------------------------------------------------------------------------


def make_scipy_fun(obj: Objective):
    """Return a numpy-in / float-out wrapper around ``obj.value()``.

    Raises :class:`SciPyBudgetExceeded` when the evaluation budget is spent so
    the calling SciPy routine can be interrupted cleanly.
    """

    def fun(x_np: np.ndarray) -> float:
        if obj.budget_exceeded:
            raise SciPyBudgetExceeded
        loss = float(obj.value(jnp.asarray(x_np)))
        if not np.isfinite(loss):
            return _NAN_PENALTY
        return loss

    return fun


def make_scipy_fun_and_grad(obj: Objective):
    """Return a wrapper calling ``obj.value_and_grad()`` -> ``(float, ndarray)``.

    Intended for use with ``jac=True`` in :func:`scipy.optimize.minimize` so
    that value and gradient are obtained in a single Objective call, avoiding
    double-counted evaluations.
    """

    def fun(x_np: np.ndarray) -> tuple[float, np.ndarray]:
        if obj.budget_exceeded:
            raise SciPyBudgetExceeded
        loss, grad = obj.value_and_grad(jnp.asarray(x_np))
        loss_f = float(loss)
        grad_np = np.asarray(grad, dtype=np.float64)
        if not np.isfinite(loss_f) or not np.all(np.isfinite(grad_np)):
            return _NAN_PENALTY, np.zeros_like(x_np, dtype=np.float64)
        return loss_f, grad_np

    return fun


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def make_budget_callback(obj: Objective):
    """Return a ``callback(xk)`` that raises :class:`SciPyBudgetExceeded`."""

    def callback(xk, *_args, **_kwargs):
        if obj.budget_exceeded:
            raise SciPyBudgetExceeded

    return callback


# ---------------------------------------------------------------------------
# Bounded step-taker for basin-hopping
# ---------------------------------------------------------------------------


class BoundedStep:
    """Random displacement clipped to problem bounds.

    Drop-in replacement for SciPy's default ``RandomDisplacement`` that
    guarantees the perturbed point remains feasible.
    """

    def __init__(self, stepsize: float, lb: np.ndarray, ub: np.ndarray) -> None:
        self.stepsize = stepsize
        self.lb = lb
        self.ub = ub

    def __call__(self, x: np.ndarray) -> np.ndarray:
        scale = self.stepsize * (self.ub - self.lb)
        x_new = x + np.random.uniform(-scale, scale, size=x.shape)
        return np.clip(x_new, self.lb, self.ub)
