"""SciPy nonlinear conjugate-gradient optimizer."""

from __future__ import annotations

from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based._scipy_common import (
    SciPyConfig,
    ScipyMinimizeAlgorithm,
)
from dfbench.core.objective import Objective


class NonlinearCG(ScipyMinimizeAlgorithm):
    """SciPy ``method="CG"`` exposed as nonlinear conjugate gradient."""

    algorithm_str = "nonlinear_cg"
    scipy_config = SciPyConfig(method="CG", unbounded=True, use_bounds=False)

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        gtol: float = 1e-5,
        maxiter: int | None = None,
        tol: float | None = None,
        **scipy_kwargs,
    ) -> None:
        """Run SciPy nonlinear conjugate gradient (``method="CG"``).

        The optimization is performed in the Objective's unbounded sigmoid
        space, so problem box bounds are not enforced inside SciPy.

        Args:
            problem_objective: Objective to mutate in place with evaluation logs.
            init_params: Initial point in unbounded space. If None, sampled via
                :meth:`Objective.random_params_unbounded`.
            random_seed: Seed used when sampling ``init_params``.
            gtol: Stop when the gradient norm falls below this tolerance.
            maxiter: Maximum number of nonlinear CG iterations.
            tol: Top-level ``scipy.optimize.minimize`` tolerance.
            **scipy_kwargs: Additional SciPy CG option entries forwarded via
                ``options=...``.
        """
        self._run_scipy_minimize(
            problem_objective,
            init_params,
            random_seed,
            tol,
            {"gtol": gtol, "maxiter": maxiter, **scipy_kwargs},
        )
