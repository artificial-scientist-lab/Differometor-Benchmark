"""Common derivative-free optimization wrapper utilities.

Provides shared infrastructure for wrapping DFO solvers (PDFO, Py-BOBYQA, etc.)
so they integrate cleanly with the Objective / OptimizationAlgorithm interface.

Key responsibilities:
  - Convert between JAX arrays and NumPy arrays expected by solvers.
  - Evaluate ``obj.value(...)`` in bounded space and log results.
  - Enforce Objective budgets via a callback / early-termination mechanism.
  - Support multistart restarts with fresh random starting points.
  - Handle NaN / Inf returns and solver failure codes gracefully.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float

from dfbench.core.objective import Objective


# Large but finite penalty returned when the solver evaluates an infeasible
# or NaN-producing point.  Using ``np.inf`` would crash most DFO solvers.
_NAN_PENALTY = 1e30


class _BudgetExhausted(Exception):
    """Raised inside a DFO callback when the Objective budget is exhausted.

    For pure-Python solvers (e.g. Py-BOBYQA) this propagates immediately and
    terminates the optimisation loop.  For Fortran-backed solvers (PDFO/prima)
    f2py stores the exception and re-raises it after the current Fortran call
    returns, which also causes prima to abort early.
    """


def dfo_objective_wrapper(
    obj: Objective,
) -> callable:
    """Return a NumPy-compatible scalar function that evaluates ``obj.value``.

    The returned callable accepts a 1-D ``np.ndarray``, converts it to a JAX
    array, evaluates through ``obj.value`` (which logs the evaluation), and
    returns a Python float.

    NaN / Inf losses are replaced with ``_NAN_PENALTY`` so that DFO solvers
    do not crash.  When the Objective budget is exceeded the wrapper raises
    ``_BudgetExhausted`` instead of calling the (expensive) physics evaluation.

    Returns:
        ``fun(x: np.ndarray) -> float``
    """

    def _fun(x: np.ndarray) -> float:
        if obj.budget_exceeded:
            raise _BudgetExhausted
        params = jnp.asarray(x)
        loss = obj.value(params)
        loss_f = float(loss)
        if not np.isfinite(loss_f):
            return _NAN_PENALTY
        return loss_f

    return _fun


def random_bounded_start(
    obj: Objective,
    key,
) -> tuple[np.ndarray, "jax.Array"]:
    """Sample a uniformly random starting point within problem bounds.

    Args:
        obj: Objective (must be in bounded mode).
        key: JAX PRNG key; a new sub-key is split off and the updated key
            is returned alongside the sample.

    Returns:
        (x0, new_key): x0 is a 1-D ``np.ndarray`` of shape ``(n_params,)``.
    """
    key, subkey = random.split(key)
    lower, upper = obj.problem.bounds
    x0_jax = random.uniform(subkey, shape=(obj.n_params,), minval=lower, maxval=upper)
    return np.asarray(x0_jax, dtype=np.float64), key


def clip_to_bounds(x: np.ndarray, obj: Objective) -> np.ndarray:
    """Clip a parameter vector to the problem bounds (in-place safe).

    Returns:
        Clipped copy of *x*.
    """
    lower = np.asarray(obj.problem.bounds[0], dtype=np.float64)
    upper = np.asarray(obj.problem.bounds[1], dtype=np.float64)
    return np.clip(x, lower, upper)


def solver_bounds_np(obj: Objective) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(lower, upper)`` as plain NumPy float64 arrays."""
    lower = np.asarray(obj.problem.bounds[0], dtype=np.float64)
    upper = np.asarray(obj.problem.bounds[1], dtype=np.float64)
    return lower, upper


def multistart_loop(
    obj: Objective,
    key,
    solve_fn: callable,
    n_restarts: int = 1,
    init_params: Float[Array, "..."] | None = None,
) -> None:
    """Run *solve_fn* up to *n_restarts* times with fresh random starts.

    ``solve_fn(x0: np.ndarray)`` should call the DFO solver once from
    starting point *x0*.  It may raise or return; either way the best
    result is tracked by the Objective automatically.

    The first restart uses *init_params* (if provided); subsequent restarts
    sample uniformly within bounds.

    Args:
        obj: The Objective instance (already logging).
        key: JAX PRNG key for sampling restarts.
        solve_fn: ``(x0: np.ndarray) -> None``: runs one solver call.
        n_restarts: Total number of solver invocations.
        init_params: Optional starting point for the first restart.
    """
    for i in range(n_restarts):
        if obj.budget_exceeded:
            break

        if i == 0 and init_params is not None:
            x0 = np.asarray(init_params, dtype=np.float64)
        else:
            x0, key = random_bounded_start(obj, key)

        try:
            solve_fn(x0)
        except _BudgetExhausted:
            break
        except Exception as exc:  # noqa: BLE001
            if obj.budget_exceeded:
                # Exception was likely triggered (directly or indirectly) by
                # _BudgetExhausted propagating through the solver; stop quietly.
                break
            # DFO solvers may raise on degenerate geometry, singular models,
            # etc.  Log and continue with next restart.
            print(
                f"[DFO restart {i + 1}/{n_restarts}] solver raised {type(exc).__name__}: {exc}"
            )
