"""SciPy trust-krylov optimizer."""

from __future__ import annotations

from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based.scipy._common import (
    SciPyConfig,
    ScipyMinimizeAlgorithm,
)
from dfbench.core.objective import Objective


class TrustKrylov(ScipyMinimizeAlgorithm):
    """SciPy trust-krylov using JAX Hessian-vector products."""

    algorithm_str = "trust_krylov"
    scipy_config = SciPyConfig(
        method="trust-krylov",
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
        gtol: float = 1e-5,
        maxiter: int | None = None,
        initial_trust_radius: float = 1.0,
        inexact: bool | None = None,
        tol: float | None = None,
        **scipy_kwargs,
    ) -> None:
        """Run SciPy ``method="trust-krylov"`` with JAX Hessian-vector products.

        This large-scale trust-region wrapper operates in the Objective's
        unbounded sigmoid space.

        Args:
            problem_objective: Objective to mutate in place with evaluation logs.
            init_params: Initial point in unbounded space. If None, sampled via
                :meth:`Objective.random_params_unbounded`.
            random_seed: Seed used when sampling ``init_params``.
            gtol: Stop when the gradient norm falls below this tolerance.
            maxiter: Maximum number of trust-region iterations.
            initial_trust_radius: Initial trust-region radius.
            inexact: Whether to solve the Krylov subproblem in inexact mode.
            tol: Top-level ``scipy.optimize.minimize`` tolerance.
            **scipy_kwargs: Additional SciPy trust-krylov option entries
                forwarded via ``options=...``.
        """
        self._run_scipy_minimize(
            problem_objective,
            init_params,
            random_seed,
            tol,
            {
                "gtol": gtol,
                "maxiter": maxiter,
                "initial_trust_radius": initial_trust_radius,
                "inexact": inexact,
                **scipy_kwargs,
            },
        )
