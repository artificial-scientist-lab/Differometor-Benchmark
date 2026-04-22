"""SciPy Newton-CG optimizer."""

from __future__ import annotations

from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based.scipy._common import (
    SciPyConfig,
    ScipyMinimizeAlgorithm,
)
from dfbench.core.objective import Objective


class NewtonCG(ScipyMinimizeAlgorithm):
    """SciPy Newton-CG using JAX gradients and Hessian-vector products."""

    algorithm_str = "newton_cg"
    scipy_config = SciPyConfig(
        method="Newton-CG",
        unbounded=True,
        use_bounds=False,
        use_hessp=True,
        cache_hessp=True,
    )

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        xtol: float = 1e-5,
        maxiter: int | None = None,
        tol: float | None = None,
        **scipy_kwargs,
    ) -> None:
        """Run SciPy ``method="Newton-CG"`` with JAX Hessian-vector products.

        This wrapper works in dfbench's unbounded sigmoid space and supplies
        exact gradients plus Hessian-vector products to SciPy.

        Args:
            problem_objective: Objective to mutate in place with evaluation logs.
            init_params: Initial point in unbounded space. If None, sampled via
                :meth:`Objective.random_params_unbounded`.
            random_seed: Seed used when sampling ``init_params``.
            xtol: Average relative error tolerance for the solution vector.
            maxiter: Maximum number of Newton-CG iterations.
            tol: Top-level ``scipy.optimize.minimize`` tolerance.
            **scipy_kwargs: Additional SciPy Newton-CG option entries forwarded
                via ``options=...``.
        """
        self._run_scipy_minimize(
            problem_objective,
            init_params,
            random_seed,
            tol,
            {"xtol": xtol, "maxiter": maxiter, **scipy_kwargs},
        )
