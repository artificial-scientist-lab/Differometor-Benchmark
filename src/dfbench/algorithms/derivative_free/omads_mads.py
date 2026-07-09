"""MADS and OrthoMADS via the OMADS library.

Mesh Adaptive Direct Search (MADS) is a derivative-free optimization method
designed for bound-constrained blackbox problems. It refines a mesh and
polls structured direction sets around the current incumbent. OrthoMADS
uses orthogonal (Householder) poll directions for richer exploration.

Both algorithms operate in **bounded physical space** (``unbounded=False``).
They are suited for rugged-landscape local exploration, not global
black-box optimization.

Requires:
    OMADS >= 2408.0 (``uv add 'dfbench[dfo]'``)

Note:
    OMADS has a known attribute-name mismatch with recent ``samplersLib``
    versions. A compatibility shim is applied at import time.
"""

from __future__ import annotations

import tempfile
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective

# ---------------------------------------------------------------------------
# OMADS compatibility shim: fix attribute name mismatch in samplersLib
# ---------------------------------------------------------------------------

_OMADS_IMPORT_ERROR: str | None = None

try:
    import samplersLib.samplers as _samplers

    if not hasattr(_samplers, "activeSampling") and hasattr(
        _samplers, "ActiveSampling"
    ):
        _samplers.activeSampling = _samplers.ActiveSampling
    if not hasattr(_samplers, "sampling") and hasattr(_samplers, "Sampling"):
        _samplers.sampling = _samplers.Sampling

    import OMADS as _omads_pkg

    _mads_main = _omads_pkg.mads.main
    _poll_main = _omads_pkg.poll.main
except Exception as exc:  # pragma: no cover
    _OMADS_IMPORT_ERROR = str(exc)
    _mads_main = None  # type: ignore[assignment]
    _poll_main = None  # type: ignore[assignment]


def _check_omads_available() -> None:
    """Raise ImportError if OMADS could not be loaded."""
    if _OMADS_IMPORT_ERROR is not None:
        raise ImportError(
            f"OMADS is required for MADS/OrthoMADS algorithms but could not be "
            f"imported: {_OMADS_IMPORT_ERROR}\n"
            f"Install with: uv add 'dfbench[dfo]'"
        )


# ---------------------------------------------------------------------------
# Shared OMADS wrapper
# ---------------------------------------------------------------------------


def _run_omads(
    obj: Objective,
    *,
    use_search_step: bool,
    psize_init: float,
    tol: float,
    seed: int,
    ns: int,
    rich_direction: bool,
) -> None:
    """Common wrapper that drives an OMADS run against an :class:`Objective`.

    The wrapper:
    * builds the config dict expected by OMADS,
    * provides a blackbox callable that evaluates ``obj.value()``,
    * converts candidate points to JAX arrays for proper logging,
    * respects the Objective's evaluation budget (``obj.budget_exceeded``), and
    * suppresses OMADS's own file I/O where possible.

    Args:
        obj: Ready-to-log Objective (``start_logging`` already called by caller).
        use_search_step: If True run ``MADS.main`` (search + poll); else
            ``POLL.main`` (poll only).
        psize_init: Initial poll-step size. Controls how far the first poll
            directions reach.
        tol: Convergence tolerance on frame/mesh size.
        seed: Numpy random seed forwarded to OMADS.
        ns: Number of search samples per search step.
        rich_direction: If True use orthogonal Householder directions (OrthoMADS).
    """
    _check_omads_available()

    bounds = obj.bounds  # (2, n_params)
    lower = np.array(bounds[0], dtype=np.float64)
    upper = np.array(bounds[1], dtype=np.float64)
    n_params = obj.n_params

    # Starting point: use best_params if available, else random bounded point
    if obj.best_params is not None:
        baseline = np.array(obj.best_params, dtype=np.float64).tolist()
    else:
        init = obj.random_params_bounded()
        baseline = np.array(init, dtype=np.float64).tolist()

    # OMADS budget: set generously; Objective budget is the real limiter.
    omads_budget = obj.max_evals if obj.max_evals is not None else 1_000_000

    # --- blackbox callable ---------------------------------------------------

    # NaN/Inf escape hatch: when the objective returns a non-finite value we
    # perturb the candidate by Gaussian noise whose scale starts at 1e-10 and
    # doubles per attempt. Perturbations are clipped to the box. After
    # ``_MAX_NAN_STREAK`` failures we return a large finite penalty so MADS
    # can keep its mesh state consistent.
    _NAN_PERTURB_BASE = 1e-10
    _MAX_NAN_STREAK = 20
    _NAN_PENALTY = 1e30
    rng = np.random.default_rng(seed)

    def _safe_eval(x: np.ndarray) -> float:
        loss = float(obj.value(jnp.asarray(x, dtype=jnp.float32)))
        if np.isfinite(loss):
            return loss
        cur = x.astype(np.float64)
        for k in range(_MAX_NAN_STREAK):
            if obj.budget_exceeded:
                break
            scale = _NAN_PERTURB_BASE * (2**k)
            perturbed = np.clip(cur + rng.normal(size=cur.shape) * scale, lower, upper)
            loss = float(obj.value(jnp.asarray(perturbed, dtype=jnp.float32)))
            if np.isfinite(loss):
                return loss
        return _NAN_PENALTY

    def _blackbox(x: list[float]) -> list[Any]:
        """Blackbox evaluation for OMADS.

        Returns [f_val, [0.0]] (no constraints).
        """
        if obj.budget_exceeded:
            return [float("inf"), [0.0]]

        return [_safe_eval(np.asarray(x, dtype=np.float64)), [0.0]]

    # --- build config dict ---------------------------------------------------

    # Use a temp dir that gets cleaned up for OMADS's post-processing files
    post_dir = tempfile.mkdtemp(prefix="omads_")

    scaling = float(np.max(upper - lower))

    config: dict[str, Any] = {
        "evaluator": {"blackbox": _blackbox},
        "param": {
            "baseline": baseline,
            "lb": lower.tolist(),
            "ub": upper.tolist(),
            "var_names": [f"x{i}" for i in range(n_params)],
            "scaling": scaling,
            "post_dir": post_dir,
            "failure_stop": False,
        },
        "options": {
            "seed": seed,
            "budget": omads_budget,
            "tol": tol,
            "display": False,
            "check_cache": True,
            "store_cache": True,
            "rich_direction": rich_direction,
            "psize_init": psize_init,
            "precision": "high",
        },
    }

    if use_search_step:
        config["search"] = {
            "ns": ns,
            "visualize": False,
        }

    # --- run OMADS -----------------------------------------------------------

    run_fn = _mads_main if use_search_step else _poll_main
    try:
        run_fn(config)
    except Exception:
        # OMADS can raise on extreme mesh refinement, when budget is exhausted
        # mid-poll, or on numerical edge cases (e.g. matrix singularities).
        # All evaluations performed so far have already been logged via the
        # Objective; we just stop the run here.
        pass

    # --- cleanup temp files --------------------------------------------------

    try:
        import shutil

        shutil.rmtree(post_dir, ignore_errors=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Algorithm classes
# ---------------------------------------------------------------------------


class OmadsMADS(OptimizationAlgorithm):
    """Mesh Adaptive Direct Search (MADS) using OMADS.

    Runs the full MADS algorithm (search step + poll step). The search step
    samples the mesh broadly before the poll step refines around the incumbent.

    Operates in **bounded physical space** (``unbounded=False``).
    Suited for rugged-landscape local exploration with moderate budgets.

    Hyperparameters:
        psize_init (float): Initial poll-step (frame) size. Larger values
            explore further from the starting point. Default 1.0.
        tol (float): Convergence tolerance on mesh size. Default 1e-9.
        ns (int): Number of search samples per search step. Default 4.

    Example:
        >>> from dfbench import Objective
        >>> from dfbench.problems import VoyagerProblem
        >>> problem = VoyagerProblem()
        >>> obj = Objective(problem, max_evals=500, verbose=1, print_every=50)
        >>> optimizer = OmadsMADS(psize_init=1.0, tol=1e-9, ns=4)
        >>> optimizer.optimize(objective=obj, random_seed=42)
        >>> print(f"Best loss: {obj.best_loss:.6f}")
    """

    algorithm_str: str = "omads_mads"
    algorithm_type: AlgorithmType = AlgorithmType.DERIVATIVE_FREE

    def __init__(
        self,
        psize_init: float = 1.0,
        tol: float = 1e-9,
        ns: int = 4,
    ) -> None:
        """Initialize MADS optimizer.

        Args:
            psize_init: Initial poll-step (frame) size. Default 1.0.
            tol: Convergence tolerance on mesh/frame size. Default 1e-9.
            ns: Number of search samples per search step. Default 4.
        """
        _check_omads_available()
        self.psize_init = psize_init
        self.tol = tol
        self.ns = ns

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        max_iterations: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Run MADS optimization (search + poll).

        Args:
            objective: Pre-configured Objective instance.
            init_params: Ignored (OMADS manages its own starting point from
                Objective's best_params or a random bounded sample).
            random_seed: Random seed for reproducibility. If None, generated.
            max_iterations: Not used directly; budget is controlled by the
                Objective's ``max_evals`` / ``max_time``.
            **kwargs: Unused.
        """
        obj = objective
        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        # JIT warmup
        obj.warmup_value()

        obj.start_logging()

        _run_omads(
            obj,
            use_search_step=True,
            psize_init=self.psize_init,
            tol=self.tol,
            seed=random_seed,
            ns=self.ns,
            rich_direction=True,
        )


class OmadsOrthoMADS(OptimizationAlgorithm):
    """OrthoMADS poll-only algorithm using OMADS.

    Runs the OrthoMADS poll step with orthogonal Householder directions,
    without the search step. This yields a leaner, more predictable
    per-iteration cost and tighter local convergence.

    Operates in **bounded physical space** (``unbounded=False``).
    Suited for local refinement on rugged landscapes.

    Hyperparameters:
        psize_init (float): Initial poll-step (frame) size. Default 1.0.
        tol (float): Convergence tolerance on mesh size. Default 1e-9.

    Example:
        >>> from dfbench import Objective
        >>> from dfbench.problems import VoyagerProblem
        >>> problem = VoyagerProblem()
        >>> obj = Objective(problem, max_evals=500, verbose=1, print_every=50)
        >>> optimizer = OmadsOrthoMADS(psize_init=1.0, tol=1e-9)
        >>> optimizer.optimize(objective=obj, random_seed=42)
        >>> print(f"Best loss: {obj.best_loss:.6f}")
    """

    algorithm_str: str = "omads_orthomads"
    algorithm_type: AlgorithmType = AlgorithmType.DERIVATIVE_FREE

    def __init__(
        self,
        psize_init: float = 1.0,
        tol: float = 1e-9,
    ) -> None:
        """Initialize OrthoMADS optimizer.

        Args:
            psize_init: Initial poll-step (frame) size. Default 1.0.
            tol: Convergence tolerance on mesh/frame size. Default 1e-9.
        """
        _check_omads_available()
        self.psize_init = psize_init
        self.tol = tol

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        max_iterations: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Run OrthoMADS optimization (poll only).

        Args:
            objective: Pre-configured Objective instance.
            init_params: Ignored (OMADS manages its own starting point from
                Objective's best_params or a random bounded sample).
            random_seed: Random seed for reproducibility. If None, generated.
            max_iterations: Not used directly; budget is controlled by the
                Objective's ``max_evals`` / ``max_time``.
            **kwargs: Unused.
        """
        obj = objective
        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        # JIT warmup
        obj.warmup_value()

        obj.start_logging()

        _run_omads(
            obj,
            use_search_step=False,
            psize_init=self.psize_init,
            tol=self.tol,
            seed=random_seed,
            ns=0,  # unused for poll-only
            rich_direction=True,
        )
