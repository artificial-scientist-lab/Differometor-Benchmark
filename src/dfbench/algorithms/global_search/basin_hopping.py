"""Basin-Hopping global optimizer via :func:`scipy.optimize.basinhopping`."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jaxtyping import Array, Float
from scipy.optimize import basinhopping

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective
from dfbench.algorithms.derivative_free._scipy_common import (
    SciPyBudgetExceeded,
    BoundedStep,
    make_scipy_fun,
    make_scipy_fun_and_grad,
    scipy_bounds,
)


class BasinHopping(OptimizationAlgorithm):
    """Basin-Hopping global optimizer with configurable local solver.

    Uses ``scipy.optimize.basinhopping`` in bounded parameter space.
    The global strategy applies random perturbations followed by local
    minimisation, accepting or rejecting moves via a Metropolis criterion.

    The local solver is configurable; the default (``L-BFGS-B``) is
    gradient-aware and uses ``obj.value_and_grad()`` via ``jac=True`` to
    avoid double-counted evaluations.  For derivative-free local solvers
    (e.g. ``Nelder-Mead``) only ``obj.value()`` is called.

    Perturbation steps are clipped to the problem bounds via a custom
    :class:`BoundedStep` step-taker so the solver never evaluates
    infeasible points.

    Attributes:
        algorithm_str: ``"basin_hopping"``
        algorithm_type: :attr:`AlgorithmType.EVOLUTIONARY`
        local_method: SciPy minimizer name used for the local search
            (default ``"L-BFGS-B"``).
    """

    algorithm_str: str = "basin_hopping"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    # Local solvers that accept a jac argument
    _GRADIENT_METHODS = frozenset(
        {"L-BFGS-B", "BFGS", "CG", "trust-constr", "Newton-CG", "TNC"}
    )
    # Local solvers that accept a bounds argument
    _BOUNDED_METHODS = frozenset(
        {"L-BFGS-B", "trust-constr", "TNC", "Nelder-Mead", "Powell", "COBYLA"}
    )

    def __init__(self, local_method: str = "L-BFGS-B") -> None:
        """Initialise Basin-Hopping.

        Args:
            local_method: SciPy method string for the local minimiser.
                Gradient-aware methods (``L-BFGS-B``, ``trust-constr``, …) use
                ``value_and_grad``; derivative-free methods use ``value`` only.
        """
        self.local_method = local_method

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        T: float = 1.0,
        stepsize: float = 0.5,
    ) -> None:
        """Run Basin-Hopping global optimisation.

        Args:
            objective: Pre-configured Objective instance.
            init_params: Starting point. If *None*, sampled uniformly in bounds.
            random_seed: Seed for reproducibility.
            T: Temperature parameter for the Metropolis acceptance criterion.
            stepsize: Relative step size for the random perturbation (fraction
                of the bound range per dimension).
        """
        obj = objective
        problem = obj.problem

        random_seed, _key = self.prepare(
            obj, unbounded=False, random_seed=random_seed
        )

        if init_params is None:
            params = obj.random_params_bounded()
        else:
            params = init_params

        uses_grad = self.local_method in self._GRADIENT_METHODS
        uses_bounds = self.local_method in self._BOUNDED_METHODS

        # JIT warmup
        if uses_grad:
            obj.warmup_value_and_grad()
        else:
            obj.warmup_value()

        obj.start_logging()

        # ── Build SciPy minimizer kwargs ──────────────────────────────
        minimizer_kwargs: dict = {"method": self.local_method}

        if uses_grad:
            fun = make_scipy_fun_and_grad(obj)
            minimizer_kwargs["jac"] = True
        else:
            fun = make_scipy_fun(obj)

        if uses_bounds:
            minimizer_kwargs["bounds"] = scipy_bounds(obj)

        # ── Bounded step-taker ────────────────────────────────────────
        lb = np.asarray(problem.bounds[0], dtype=np.float64)
        ub = np.asarray(problem.bounds[1], dtype=np.float64)
        take_step = BoundedStep(stepsize, lb, ub)

        x0 = np.asarray(params, dtype=np.float64)

        def bh_callback(x, f, accept):
            """Return *True* to stop basin-hopping early."""
            return bool(obj.budget_exceeded)

        try:
            basinhopping(
                fun,
                x0,
                niter=int(1e9),
                T=T,
                minimizer_kwargs=minimizer_kwargs,
                take_step=take_step,
                callback=bh_callback,
                seed=random_seed,
            )
        except SciPyBudgetExceeded:
            pass
