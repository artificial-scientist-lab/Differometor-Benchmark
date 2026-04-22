"""SciPy L-BFGS-B optimizer."""

from __future__ import annotations

from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based.scipy._common import (
    SciPyConfig,
    ScipyMinimizeAlgorithm,
)
from dfbench.core.objective import Objective


class LBFGSB(ScipyMinimizeAlgorithm):
    """SciPy L-BFGS-B in unbounded sigmoid space."""

    algorithm_str = "lbfgsb"
    scipy_config = SciPyConfig(method="L-BFGS-B", unbounded=True, use_bounds=False)

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        gtol: float = 1e-5,
        maxiter: int | None = None,
        maxcor: int = 10,
        maxls: int = 20,
        tol: float | None = None,
        **scipy_kwargs,
    ) -> None:
        """Run SciPy ``method="L-BFGS-B"`` in unbounded sigmoid space.

        This wrapper optimizes the Objective's sigmoid-space parameterization,
        so box bounds from the underlying problem are not passed to SciPy.

        Args:
            problem_objective: Objective to mutate in place with evaluation logs.
            init_params: Initial point in unbounded space. If None, sampled via
                :meth:`Objective.random_params_unbounded`.
            random_seed: Seed used when sampling ``init_params``.
            gtol: Stop when the projected gradient norm reaches this tolerance.
            maxiter: Maximum number of L-BFGS-B iterations.
            maxcor: Number of correction pairs kept in the limited-memory matrix.
            maxls: Maximum line-search steps per iteration.
            tol: Top-level ``scipy.optimize.minimize`` tolerance.
            **scipy_kwargs: Additional SciPy L-BFGS-B option entries forwarded
                via ``options=...``.
        """
        self._run_scipy_minimize(
            problem_objective,
            init_params,
            random_seed,
            tol,
            {
                "gtol": gtol,
                "maxiter": maxiter,
                "maxcor": maxcor,
                "maxls": maxls,
                **scipy_kwargs,
            },
        )
