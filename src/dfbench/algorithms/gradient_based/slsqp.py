"""SciPy SLSQP optimizer."""

from __future__ import annotations

from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based._scipy_common import (
    SciPyConfig,
    ScipyMinimizeAlgorithm,
)
from dfbench.core.objective import Objective


class SLSQP(ScipyMinimizeAlgorithm):
    """SciPy SLSQP in bounded physical space."""

    algorithm_str = "slsqp"
    scipy_config = SciPyConfig(method="SLSQP", unbounded=False, use_bounds=True)

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        ftol: float = 1e-6,
        maxiter: int = 200,
        tol: float | None = None,
        **scipy_kwargs,
    ) -> None:
        """Run SciPy ``method="SLSQP"`` with bound constraints.

        This wrapper keeps the Objective in bounded physical space and forwards
        box bounds to SciPy's sequential least-squares solver.

        Args:
            problem_objective: Objective to mutate in place with evaluation logs.
            init_params: Initial point in bounded space. If None, sampled via
                :meth:`Objective.random_params_bounded`.
            random_seed: Seed used when sampling ``init_params``.
            ftol: Precision target for SLSQP's stopping tests.
            maxiter: Maximum number of SLSQP iterations.
            tol: Top-level ``scipy.optimize.minimize`` tolerance.
            **scipy_kwargs: Additional SciPy SLSQP option entries forwarded via
                ``options=...``.
        """
        self._run_scipy_minimize(
            problem_objective,
            init_params,
            random_seed,
            tol,
            {"ftol": ftol, "maxiter": maxiter, **scipy_kwargs},
        )
