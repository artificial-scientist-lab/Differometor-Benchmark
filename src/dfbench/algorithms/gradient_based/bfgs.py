"""SciPy BFGS optimizer."""

from __future__ import annotations

from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based._scipy_common import (
    SciPyConfig,
    ScipyMinimizeAlgorithm,
)
from dfbench.core.objective import Objective


class BFGS(ScipyMinimizeAlgorithm):
    """SciPy BFGS in unbounded sigmoid space."""

    algorithm_str = "bfgs"
    scipy_config = SciPyConfig(method="BFGS", unbounded=True, use_bounds=False)

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        gtol: float = 1e-5,
        maxiter: int | None = None,
        xrtol: float = 0.0,
        tol: float | None = None,
        **scipy_kwargs,
    ) -> None:
        """Run SciPy ``method="BFGS"`` in unbounded space.

        This wrapper optimizes the Objective's sigmoid-space parameterization,
        so box bounds from the underlying problem are not passed to SciPy.

        Args:
            problem_objective: Objective to mutate in place with evaluation logs.
            init_params: Initial point in unbounded space. If None, sampled via
                :meth:`Objective.random_params_unbounded`.
            random_seed: Seed used when sampling ``init_params``.
            gtol: Stop when the gradient norm falls below this tolerance.
            maxiter: Maximum number of BFGS iterations.
            xrtol: Relative step-size tolerance used by SciPy's BFGS stopping test.
            tol: Top-level ``scipy.optimize.minimize`` tolerance.
            **scipy_kwargs: Additional SciPy BFGS option entries forwarded via
                ``options=...``.
        """
        self._run_scipy_minimize(
            problem_objective,
            init_params,
            random_seed,
            tol,
            {"gtol": gtol, "maxiter": maxiter, "xrtol": xrtol, **scipy_kwargs},
        )
