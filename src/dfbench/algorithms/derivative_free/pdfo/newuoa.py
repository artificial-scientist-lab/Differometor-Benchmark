"""NEWUOA: NEW Unconstrained Optimization Algorithm (via PDFO).

NEWUOA is a derivative-free trust-region method by M. J. D. Powell.  It
builds a quadratic interpolation model using *fewer* interpolation points
than UOBYQA (``2n+1`` by default instead of ``(n+1)(n+2)/2``), making it
more practical for moderate-to-large dimensional problems.

Like UOBYQA, NEWUOA is designed for **unconstrained** problems.  This
wrapper operates in bounded physical space by clipping any out-of-bounds
proposals to the problem bounds before evaluation.  For problems where
strict bound handling is essential, prefer BOBYQA or LINCOA.

Defaults are conservative and benchmark-oriented.

Reference:
    Powell, M. J. D. (2006). The NEWUOA software for unconstrained
    optimization without derivatives. *Large-Scale Nonlinear Optimization*,
    Springer, 255-297.
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective
from dfbench.algorithms.derivative_free._dfo_common import (
    dfo_objective_wrapper,
    multistart_loop,
    solver_bounds_np,
    clip_to_bounds,
)


class PDFONEWUOA(OptimizationAlgorithm):
    """NEWUOA derivative-free optimizer (via PDFO).

    Hyperparameters:
        radius_init: Initial trust-region radius.  Defaults to 10% of the mean
            bound range.
        radius_final: Final trust-region radius (convergence tolerance).
        npt: Number of interpolation points.  Must satisfy
            ``n+2 <= npt <= (n+1)*(n+2)/2``.  Defaults to ``2*n+1`` (PDFO
            default).
        n_restarts: Number of multistart runs within the evaluation budget.

    Space mode:
        Bounded (``unbounded=False``).  NEWUOA itself is unconstrained; the
        wrapper clips proposals to problem bounds.
    """

    algorithm_str: str = "pdfo_newuoa"
    algorithm_type: AlgorithmType = AlgorithmType.DERIVATIVE_FREE

    def __init__(
        self,
        radius_init: float | None = None,
        radius_final: float = 1e-6,
        npt: int | None = None,
        n_restarts: int = 1,
    ) -> None:
        self.radius_init = radius_init
        self.radius_final = radius_final
        self.npt = npt
        self.n_restarts = n_restarts

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        try:
            import pdfo
        except ImportError as exc:
            raise ImportError(
                "PDFO is required for NEWUOA.  Install with: uv add 'dfbench[dfo]'"
            ) from exc

        obj = objective
        random_seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lower, upper = solver_bounds_np(obj)
        radius_init = (
            self.radius_init
            if self.radius_init is not None
            else float(0.1 * np.mean(upper - lower))
        )
        npt = self.npt if self.npt is not None else 2 * obj.n_params + 1

        raw_fun = dfo_objective_wrapper(obj)

        def clipped_fun(x: np.ndarray) -> float:
            xc = clip_to_bounds(x, obj)
            return raw_fun(xc)

        # JIT warmup
        obj.warmup_value()

        obj.start_logging()

        def _solve(x0: np.ndarray) -> None:
            if obj.budget_exceeded:
                return
            if max_iterations is not None:
                maxfev = max_iterations
            elif obj.evals_left is not None:
                maxfev = max(obj.evals_left, 2 * obj.n_params + 2)
            else:
                maxfev = 500 * (obj.n_params + 1)
            pdfo.pdfo(
                clipped_fun,
                x0,
                method="newuoa",
                options={
                    "radius_init": radius_init,
                    "radius_final": self.radius_final,
                    "npt": npt,
                    "maxfev": maxfev,
                    "quiet": True,
                },
            )

        multistart_loop(
            obj,
            key,
            _solve,
            n_restarts=self.n_restarts,
            init_params=init_params,
        )
