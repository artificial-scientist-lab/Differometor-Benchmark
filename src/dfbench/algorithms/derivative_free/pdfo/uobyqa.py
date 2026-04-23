"""UOBYQA — Unconstrained Optimization BY Quadratic Approximation (via PDFO).

UOBYQA is a derivative-free trust-region method by M. J. D. Powell that
builds a full quadratic interpolation model of the objective.  It is designed
for *unconstrained* problems with a moderate number of variables.

Because UOBYQA does **not** support bound constraints natively, this wrapper
operates in bounded space by clipping evaluations to the problem bounds.
If a point proposed by the solver lies outside bounds it is projected back;
the objective value at the clipped point is returned.  This is a pragmatic
approach — for strictly bounded problems prefer NEWUOA or BOBYQA.

Defaults are conservative and benchmark-oriented: one restart, ``rhobeg``
derived from the bound range, ``maxfev`` deferred to the Objective budget.

Reference:
    Powell, M. J. D. (2002). UOBYQA: unconstrained optimization by quadratic
    approximation. *Mathematical Programming*, 92(3), 555–582.
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective
from dfbench.algorithms.derivative_free._dfo_common import (
    dfo_objective_wrapper,
    random_bounded_start,
    multistart_loop,
    solver_bounds_np,
    clip_to_bounds,
)


class PDFOUOBYQA(OptimizationAlgorithm):
    """UOBYQA derivative-free optimizer (via PDFO).

    Hyperparameters:
        radius_init: Initial trust-region radius.  Defaults to 10% of the mean
            bound range.
        radius_final: Final trust-region radius (convergence tolerance).
        n_restarts: Number of multistart runs within the evaluation budget.

    Space mode:
        Bounded (``unbounded=False``).  UOBYQA itself is unconstrained; the
        wrapper clips proposed points to the problem bounds before evaluation.
    """

    algorithm_str: str = "pdfo_uobyqa"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self,
        radius_init: float | None = None,
        radius_final: float = 1e-6,
        n_restarts: int = 1,
    ) -> None:
        self.radius_init = radius_init
        self.radius_final = radius_final
        self.n_restarts = n_restarts

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        try:
            import pdfo
        except ImportError as exc:
            raise ImportError(
                "PDFO is required for UOBYQA.  Install with: pip install pdfo"
            ) from exc

        obj = problem_objective
        random_seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lower, upper = solver_bounds_np(obj)
        radius_init = self.radius_init if self.radius_init is not None else float(0.1 * np.mean(upper - lower))

        # Build wrapper that clips to bounds before evaluating
        raw_fun = dfo_objective_wrapper(obj)

        def clipped_fun(x: np.ndarray) -> float:
            xc = clip_to_bounds(x, obj)
            return raw_fun(xc)

        # JIT warmup
        _ = obj.value(jnp.asarray(clip_to_bounds(np.zeros(obj.n_params), obj)))

        obj.start_logging()

        def _solve(x0: np.ndarray) -> None:
            if obj.budget_exceeded:
                return
            if max_iterations is not None:
                maxfev = max_iterations
            elif obj.evals_left is not None:
                maxfev = max(obj.evals_left, 2 * obj.n_params + 2)
            else:
                maxfev = int(1e8)
            pdfo.pdfo(
                clipped_fun,
                x0,
                method="uobyqa",
                options={
                    "radius_init": radius_init,
                    "radius_final": self.radius_final,
                    "maxfev": maxfev,
                    "quiet": True,
                },
            )

        multistart_loop(
            obj, key, _solve,
            n_restarts=self.n_restarts,
            init_params=init_params,
        )
