"""LINCOA: LINearly Constrained Optimization Algorithm (via PDFO).

LINCOA is a derivative-free trust-region method by M. J. D. Powell that
handles *bound constraints* and *linear inequality constraints* natively.

When the problem exposes only box bounds (the common case in this benchmark),
LINCOA automatically uses them.  If the problem also exposes linear
constraints via ``problem.linear_constraints`` (returning a dict with keys
``A_ub``, ``b_ub`` for ``A_ub @ x <= b_ub``), those are forwarded to the
solver as well.  Problems without any constraints are also supported -
LINCOA will behave like a bounded NEWUOA.

Defaults are conservative and benchmark-oriented: bounded physical space,
one restart, ``rhobeg`` derived from the bound range.

Reference:
    Powell, M. J. D. (2015). On fast trust region methods for quadratic
    models with linear constraints. *Mathematical Programming Computation*,
    7(3), 237-267.
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
)


class PDFOLINCOA(OptimizationAlgorithm):
    """LINCOA derivative-free optimizer (via PDFO).

    Hyperparameters:
        radius_init: Initial trust-region radius.  Defaults to 10% of the mean
            bound range.
        radius_final: Final trust-region radius (convergence tolerance).
        npt: Number of interpolation points.  Defaults to ``2*n+1``.
        n_restarts: Number of multistart runs within the evaluation budget.

    Space mode:
        Bounded (``unbounded=False``).  LINCOA handles box bounds and optional
        linear inequality constraints natively.
    """

    algorithm_str: str = "pdfo_lincoa"
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
                "PDFO is required for LINCOA.  Install with: uv add 'dfbench[dfo]'"
            ) from exc
        from scipy.optimize import LinearConstraint, Bounds  # noqa: E402

        obj = objective
        random_seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lower, upper = solver_bounds_np(obj)
        radius_init = (
            self.radius_init
            if self.radius_init is not None
            else float(0.1 * np.mean(upper - lower))
        )
        npt = self.npt if self.npt is not None else 2 * obj.n_params + 1

        fun = dfo_objective_wrapper(obj)
        bounds = Bounds(lb=lower, ub=upper)

        # Build constraint list: always include bounds; add linear if available.
        constraints: list = []
        problem = obj.problem
        if hasattr(problem, "linear_constraints"):
            lc = problem.linear_constraints  # type: ignore[attr-defined]
            if lc is not None:
                A_ub = np.asarray(lc["A_ub"], dtype=np.float64)
                b_ub = np.asarray(lc["b_ub"], dtype=np.float64)
                constraints.append(LinearConstraint(A_ub, ub=b_ub))

        # JIT warmup
        obj.warmup_value()

        obj.start_logging()

        def _solve(x0: np.ndarray) -> None:
            if obj.budget_exceeded:
                return
            x0_clipped = np.clip(x0, lower, upper)
            if max_iterations is not None:
                maxfev = max_iterations
            elif obj.evals_left is not None:
                maxfev = max(obj.evals_left, 2 * obj.n_params + 2)
            else:
                maxfev = 500 * (obj.n_params + 1)
            pdfo.pdfo(
                fun,
                x0_clipped,
                method="lincoa",
                bounds=bounds,
                constraints=constraints if constraints else (),
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
