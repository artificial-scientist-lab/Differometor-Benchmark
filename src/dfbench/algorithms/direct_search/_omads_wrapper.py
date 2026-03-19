"""Common OMADS wrapper for MADS and OrthoMADS algorithms.

This module provides ``_run_omads_poll``, the shared thin wrapper around the
OMADS POLL step.  It is intentionally kept minimal:

* Converts repo ``Objective`` evaluations to an OMADS blackbox callable.
* Enforces the Objective's eval and time budgets.
* Handles the samplersLib API mismatch present in OMADS ≤ 2408.x.
* Does **not** fork or replicate any OMADS logic.

Supported mesh types:
    ``"ORTHO"``  – OrthoMADS orthogonal directions (default in OMADS).
    ``"GMESH"``  – Generalised-mesh MADS (standard MADS poll).
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from dfbench.core.objective import Objective

# ---------------------------------------------------------------------------
# samplersLib compatibility shim
# ---------------------------------------------------------------------------
# OMADS ≤ 2408.x references lowercase class aliases
# (``activeSampling``, ``sampling``, ``halton``) that were renamed to
# CamelCase in samplersLib.  Apply the shim once at module import so that
# both ``OMADS.POLL`` and ``OMADS.SEARCH`` import cleanly.
# ---------------------------------------------------------------------------
try:
    import samplersLib.samplers as _sl

    for _old, _new in [
        ("activeSampling", "ActiveSampling"),
        ("sampling", "Sampling"),
        ("halton", "Halton"),
    ]:
        if not hasattr(_sl, _old) and hasattr(_sl, _new):
            setattr(_sl, _old, getattr(_sl, _new))
    del _sl, _old, _new
except ImportError:
    pass  # samplersLib not present; OMADS import will fail loudly below.

# ---------------------------------------------------------------------------
# Lazy OMADS import – raise a clear error if the package is missing.
# ---------------------------------------------------------------------------
try:
    import OMADS.POLL as _omads_poll  # noqa: F401
except ImportError as _exc:
    raise ImportError(
        "The MADS/OrthoMADS algorithms require the 'OMADS' package. "
        "Install it with:  pip install OMADS samplersLib"
    ) from _exc


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

class _BudgetExhausted(Exception):
    """Internal signal: Objective budget was exceeded inside the blackbox."""


def _run_omads_poll(
    obj: Objective,
    init_params: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    random_seed: int,
    mesh_type: str,
    poll_size_init: float = 1.0,
    min_poll_size: float = 1e-9,
    opportunistic: bool = False,
    rich_direction: bool = False,
) -> None:
    """Run one OMADS POLL step and log all evaluations into *obj*.

    Parameters
    ----------
    obj:
        Pre-configured :class:`~dfbench.core.objective.Objective` instance.
        ``obj.start_logging()`` **must** have been called before this function.
    init_params:
        1-D NumPy array with the starting point (in bounded physical space).
    lower, upper:
        1-D NumPy arrays with the variable bounds.
    random_seed:
        Integer seed for OMADS's internal RNG.
    mesh_type:
        ``"ORTHO"`` for OrthoMADS, ``"GMESH"`` for standard MADS.
    poll_size_init:
        Initial poll-frame size (relative to the variable range scaling).
    min_poll_size:
        Termination tolerance on the frame size.
    opportunistic:
        If ``True``, OMADS stops evaluating poll candidates as soon as an
        improvement is found (reduces evaluations per iteration).
    rich_direction:
        If ``True``, OMADS uses the last successful direction to bias the mesh
        update (can speed convergence on smooth landscapes).

    Notes
    -----
    The blackbox callable checks ``obj.budget_exceeded`` before each
    evaluation and returns ``[inf, [inf]]`` to signal infeasibility once
    the budget is exhausted, causing OMADS to stagnate and eventually stop
    on its own convergence criterion.
    """
    import OMADS.POLL as POLL

    n = len(init_params)

    # ------------------------------------------------------------------
    # Evaluation budget passed to OMADS.  OMADS tracks its own counter
    # independently from Objective; we set it to the remaining budget so
    # OMADS stops on its own when the budget is used up.
    # ------------------------------------------------------------------
    if obj._max_evals is not None:
        eval_budget = max(1, obj._max_evals - obj.eval_count)
    else:
        eval_budget = 10_000  # effectively unlimited

    # ------------------------------------------------------------------
    # Blackbox callable: every OMADS candidate point is evaluated here.
    # ------------------------------------------------------------------
    def _blackbox(x: list[float]) -> list[float | list[float]]:
        if obj.budget_exceeded:
            return [float("inf"), [float("inf")]]

        params = jnp.array(x, dtype=jnp.float32)
        loss = float(obj.value(params))
        return [loss, [0.0]]

    # ------------------------------------------------------------------
    # Temporary directory for OMADS log/post files (we do not need them).
    # ------------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="omads_") as post_dir:
        data: dict = {
            "param": {
                "baseline": init_params.tolist(),
                "lb": lower.tolist(),
                "ub": upper.tolist(),
                "var_names": [f"x_{i}" for i in range(n)],
                "scaling": 1.0,
                "post_dir": post_dir,
                "meshType": mesh_type,
            },
            "options": {
                "seed": (int(random_seed) % (2**31 - 2)) + 1,
                "budget": eval_budget,
                "tol": float(min_poll_size),
                "psize_init": float(poll_size_init),
                "display": False,
                "opportunistic": opportunistic,
                "check_cache": False,
                "store_cache": False,
                "save_results": False,
                "save_coordinates": False,
                "rich_direction": rich_direction,
            },
            "evaluator": {
                "blackbox": _blackbox,
            },
        }

        POLL.main(data)
