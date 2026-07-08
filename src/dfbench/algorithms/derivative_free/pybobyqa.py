"""Py: BOBYQA: Bound Optimization BY Quadratic Approximation.

Py-BOBYQA is a flexible Python implementation of Powell's BOBYQA algorithm
with modern enhancements for noisy objectives, automatic restarts, and
robust trust-region management.  It is well-suited as a *serious* local
derivative-free method for expensive, possibly noisy, bounded objectives.

This wrapper operates in **bounded physical space** (``unbounded=False``)
because BOBYQA handles box bounds natively and efficiently.

Key features exposed:
  - ``rhobeg`` / ``rhoend``: trust-region radii (initial / final).
  - ``seek_global_minimum``: enables automatic soft restarts within the
    evaluation budget, useful for multi-modal landscapes.
  - ``objfun_has_noise``: enables the noisy-objective variant that uses
    regression models instead of interpolation, appropriate when the
    problem evaluation has stochastic components.
  - ``n_restarts``: number of external multistart restarts managed by
    our wrapper (orthogonal to Py-BOBYQA's own restart logic).
  - ``npt``: number of interpolation points (``n+1 <= npt <= (n+1)(n+2)/2``).
  - ``scaling_within_bounds``: let Py-BOBYQA scale variables to [0, 1]
    internally for better numerical conditioning.

Reference:
    Cartis, C., Fiala, J., Marber, B., & Roberts, L. (2019).
    Improving the flexibility and robustness of model-based
    derivative-free optimization solvers. *ACM Transactions on
    Mathematical Software (TOMS)*, 45(3), 1-41.
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


class PyBOBYQA(OptimizationAlgorithm):
    """Py-BOBYQA derivative-free bound-constrained optimizer.

    Hyperparameters:
        rhobeg: Initial trust-region radius.  Defaults to 10% of the mean
            bound range.
        rhoend: Final trust-region radius (convergence tolerance).
        npt: Number of interpolation points.  If ``None``, Py-BOBYQA uses
            ``2*n+1`` by default.
        seek_global_minimum: If ``True``, Py-BOBYQA performs automatic
            restarts to attempt a global search.  Consumes more evaluations.
        objfun_has_noise: If ``True``, use the noise-aware regression model
            variant.  Appropriate for stochastic objectives.
        scaling_within_bounds: If ``True``, Py-BOBYQA internally rescales
            all variables to [0, 1] for better conditioning.
        n_restarts: Number of *external* multistart restarts managed by the
            wrapper (independent of Py-BOBYQA's own restarts).

    Space mode:
        Bounded (``unbounded=False``).  Py-BOBYQA handles box bounds
        natively.
    """

    algorithm_str: str = "pybobyqa"
    algorithm_type: AlgorithmType = AlgorithmType.DERIVATIVE_FREE

    def __init__(
        self,
        rhobeg: float | None = None,
        rhoend: float = 1e-8,
        npt: int | None = None,
        seek_global_minimum: bool = False,
        objfun_has_noise: bool = False,
        scaling_within_bounds: bool = True,
        n_restarts: int = 1,
    ) -> None:
        self.rhobeg = rhobeg
        self.rhoend = rhoend
        self.npt = npt
        self.seek_global_minimum = seek_global_minimum
        self.objfun_has_noise = objfun_has_noise
        self.scaling_within_bounds = scaling_within_bounds
        self.n_restarts = n_restarts

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        try:
            import pybobyqa
        except ImportError as exc:
            raise ImportError(
                "Py-BOBYQA is required.  Install with: pip install Py-BOBYQA"
            ) from exc

        obj = objective
        random_seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lower, upper = solver_bounds_np(obj)
        npt = self.npt  # None -> Py-BOBYQA default (2n+1)

        fun = dfo_objective_wrapper(obj)

        # JIT warmup
        obj.warmup_value()

        obj.start_logging()

        def _solve(x0: np.ndarray) -> None:
            if obj.budget_exceeded:
                return
            x0_clipped = np.clip(x0, lower, upper)

            # Cap maxfun to remaining eval budget so the solver doesn't run
            # far beyond when the Objective stops logging.
            if max_iterations is not None:
                maxfun = max_iterations
            elif obj.evals_left is not None:
                maxfun = max(obj.evals_left, 2 * obj.n_params + 2)
            else:
                maxfun = 500 * (obj.n_params + 1)

            # Compute rhobeg respecting Py-BOBYQA constraints.
            # When scaling_within_bounds=True, Py-BOBYQA maps to [0, 1], so
            # rhobeg must satisfy 2*rhobeg <= 1.0 (the scaled gap).
            # When scaling_within_bounds=False, rhobeg must satisfy
            # 2*rhobeg <= min(upper - lower).
            if self.scaling_within_bounds:
                effective_gap = 1.0  # scaled space
            else:
                effective_gap = float(np.min(upper - lower))
            max_rhobeg = effective_gap / 2.0 - 1e-10  # strict inequality
            rhobeg = self.rhobeg if self.rhobeg is not None else 0.1 * effective_gap
            rhobeg = min(rhobeg, max_rhobeg)
            rhobeg = max(rhobeg, self.rhoend * 2)  # must be > rhoend

            kwargs: dict = {
                "rhobeg": rhobeg,
                "rhoend": self.rhoend,
                "maxfun": maxfun,
                "seek_global_minimum": self.seek_global_minimum,
                "objfun_has_noise": self.objfun_has_noise,
                "scaling_within_bounds": self.scaling_within_bounds,
            }
            if npt is not None:
                kwargs["npt"] = npt

            result = pybobyqa.solve(
                fun,
                x0_clipped,
                bounds=(lower, upper),
                **kwargs,
            )
            if result.flag == -1:
                raise RuntimeError(f"Py-BOBYQA input error: {result.msg}")

        multistart_loop(
            obj,
            key,
            _solve,
            n_restarts=self.n_restarts,
            init_params=init_params,
        )
