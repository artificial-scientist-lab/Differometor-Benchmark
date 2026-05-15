"""Dual Annealing global optimizer via :func:`scipy.optimize.dual_annealing`."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.optimize import dual_annealing, minimize

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective
from dfbench.algorithms.derivative_free._scipy_common import (
    SciPyBudgetExceeded,
    make_scipy_fun,
    scipy_bounds,
    scipy_bounds_list,
    make_budget_callback,
)


class DualAnnealing(OptimizationAlgorithm):
    """Dual Annealing global optimizer.

    Uses ``scipy.optimize.dual_annealing`` in bounded parameter space.
    Combines classical simulated annealing with a fast local-search strategy.

    An optional local refinement pass can be enabled to polish the best
    incumbent with Nelder-Mead after the main annealing loop finishes.

    Attributes:
        algorithm_str: ``"dual_annealing"``
        algorithm_type: :attr:`AlgorithmType.EVOLUTIONARY`
    """

    algorithm_str: str = "dual_annealing"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        initial_temp: float = 5230.0,
        restart_temp_ratio: float = 2e-5,
        visit: float = 2.62,
        accept: float = -5.0,
        local_search: bool = True,
        local_refinement: bool = False,
    ) -> None:
        """Run Dual Annealing optimisation.

        Args:
            problem_objective: Pre-configured Objective instance.
            init_params: Starting point (``x0``). If *None*, sampled uniformly.
            random_seed: Seed for reproducibility.
            initial_temp: Initial temperature for generalised simulated
                annealing (must be > 0).
            restart_temp_ratio: Ratio of restart temperature to *initial_temp*.
            visit: Visiting-distribution parameter (> 1; default 2.62).
            accept: Acceptance-distribution parameter (< 0).
            local_search: Whether to apply a local search during annealing
                (``no_local_search=False``).
            local_refinement: If *True*, refine the best incumbent with a
                bounded Nelder-Mead pass after the main annealing loop.
        """
        obj = problem_objective

        random_seed, _key = self.prepare(
            obj, unbounded=False, random_seed=random_seed
        )

        if init_params is None:
            params = obj.random_params_bounded()
        else:
            params = init_params

        # JIT warmup
        obj.warmup_value()

        obj.start_logging()

        fun = make_scipy_fun(obj)
        bounds_list = scipy_bounds_list(obj)
        x0 = np.asarray(params, dtype=np.float64)

        def da_callback(x, f, context):
            """Return *True* to stop dual annealing early."""
            return bool(obj.budget_exceeded)

        try:
            dual_annealing(
                fun,
                bounds=bounds_list,
                x0=x0,
                maxiter=int(1e9),
                maxfun=int(1e9),
                initial_temp=initial_temp,
                restart_temp_ratio=restart_temp_ratio,
                visit=visit,
                accept=accept,
                no_local_search=not local_search,
                callback=da_callback,
                seed=random_seed,
            )
        except SciPyBudgetExceeded:
            pass

        # ── Optional local refinement of the best incumbent ───────────
        if local_refinement and not obj.budget_exceeded and obj.best_params is not None:
            x_best = np.asarray(obj.best_params, dtype=np.float64)
            bounds_obj = scipy_bounds(obj)
            remaining = obj.evals_left if obj.evals_left is not None else 1_000
            callback_refine = make_budget_callback(obj)

            try:
                minimize(
                    fun,
                    x_best,
                    method="Nelder-Mead",
                    bounds=bounds_obj,
                    callback=callback_refine,
                    options={
                        "maxfev": remaining,
                        "xatol": 1e-10,
                        "fatol": 1e-10,
                        "adaptive": True,
                    },
                )
            except SciPyBudgetExceeded:
                pass
