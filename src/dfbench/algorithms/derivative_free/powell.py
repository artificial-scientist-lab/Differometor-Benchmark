"""Powell's conjugate-direction method via :func:`scipy.optimize.minimize`."""

from __future__ import annotations

import numpy as np
from jaxtyping import Array, Float
from scipy.optimize import minimize

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective
from dfbench.algorithms.derivative_free._scipy_common import (
    SciPyBudgetExceeded,
    make_scipy_fun,
    scipy_bounds,
    make_budget_callback,
)


class Powell(OptimizationAlgorithm):
    """Powell's conjugate-direction derivative-free local optimizer.

    Uses ``scipy.optimize.minimize(method='Powell')`` in **bounded**
    parameter space (requires SciPy >= 1.8).

    Powell's method minimises by performing sequential one-dimensional
    searches along each direction in a set of conjugate directions.
    It does not use gradients.

    Attributes:
        algorithm_str: ``"powell"``
        algorithm_type: :attr:`AlgorithmType.EVOLUTIONARY`
    """

    algorithm_str: str = "powell"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        xtol: float = 1e-8,
        ftol: float = 1e-8,
    ) -> None:
        """Run Powell conjugate-direction optimization.

        Args:
            objective: Pre-configured Objective instance.
            init_params: Starting point. If *None*, sampled uniformly in bounds.
            random_seed: Seed for reproducibility (controls initial point).
            xtol: Relative parameter tolerance for convergence.
            ftol: Relative function-value tolerance for convergence.
        """
        obj = objective

        random_seed, _key = self.prepare(obj, unbounded=False, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_bounded()
        else:
            params = init_params

        # JIT warmup
        obj.warmup_value()

        obj.start_logging()

        fun = make_scipy_fun(obj)
        x0 = np.asarray(params, dtype=np.float64)
        bounds = scipy_bounds(obj)
        callback = make_budget_callback(obj)
        maxfev = obj.evals_left if obj.evals_left is not None else 10_000

        try:
            minimize(
                fun,
                x0,
                method="Powell",
                bounds=bounds,
                callback=callback,
                options={
                    "maxfev": maxfev,
                    "xtol": xtol,
                    "ftol": ftol,
                },
            )
        except SciPyBudgetExceeded:
            pass
