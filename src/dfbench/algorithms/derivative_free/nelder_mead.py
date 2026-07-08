"""Nelder-Mead simplex algorithm via :func:`scipy.optimize.minimize`."""

from __future__ import annotations

import numpy as np
from jaxtyping import Array, Float
try:
    from scipy.optimize import minimize
except ImportError as exc:
    raise ImportError(
        "scipy is required for this algorithm. Install with:  uv add 'dfbench[scipy]'"
    ) from exc

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective
from dfbench.algorithms.derivative_free._scipy_common import (
    SciPyBudgetExceeded,
    make_scipy_fun,
    scipy_bounds,
    make_budget_callback,
)


class NelderMead(OptimizationAlgorithm):
    """Nelder-Mead simplex derivative-free local optimizer.

    Uses ``scipy.optimize.minimize(method='Nelder-Mead')`` in **bounded**
    parameter space (requires SciPy >= 1.7).

    Suitable as a local refinement method or for low-dimensional problems
    where gradient information is unavailable or unreliable.

    Attributes:
        algorithm_str: ``"nelder_mead"``
        algorithm_type: :attr:`AlgorithmType.DERIVATIVE_FREE`
    """

    algorithm_str: str = "nelder_mead"
    algorithm_type: AlgorithmType = AlgorithmType.DERIVATIVE_FREE

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        xatol: float = 1e-8,
        fatol: float = 1e-8,
        adaptive: bool = True,
    ) -> None:
        """Run Nelder-Mead simplex optimization.

        Args:
            objective: Pre-configured Objective instance.
            init_params: Starting point. If *None*, sampled uniformly in bounds.
            random_seed: Seed for reproducibility (controls initial point).
            xatol: Absolute parameter tolerance for convergence.
            fatol: Absolute function-value tolerance for convergence.
            adaptive: Adapt simplex parameters to dimensionality (recommended
                for n > 1).
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
                method="Nelder-Mead",
                bounds=bounds,
                callback=callback,
                options={
                    "maxfev": maxfev,
                    "xatol": xatol,
                    "fatol": fatol,
                    "adaptive": adaptive,
                },
            )
        except SciPyBudgetExceeded:
            pass
