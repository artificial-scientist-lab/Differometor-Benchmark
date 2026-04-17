"""SciPy COBYLA optimizer."""

from __future__ import annotations

from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based.scipy._common import (
    SciPyConfig,
    ScipyMinimizeAlgorithm,
)
from dfbench.core.objective import Objective


class COBYLA(ScipyMinimizeAlgorithm):
    """SciPy COBYLA with physical bounds represented in bounded space."""

    algorithm_str = "cobyla"
    scipy_config = SciPyConfig(
        method="COBYLA",
        unbounded=False,
        use_bounds=True,
        use_jac=False,
    )

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        rhobeg: float = 1.0,
        tol: float = 1e-4,
        catol: float = 1e-8,
        maxiter: int | None = None,
        **scipy_kwargs,
    ) -> None:
        """Run derivative-free SciPy ``method="COBYLA"`` with box bounds.

        This wrapper stays in bounded physical space and relies on SciPy's
        bound handling for COBYLA, without supplying gradients.

        Args:
            problem_objective: Objective to mutate in place with evaluation logs.
            init_params: Initial point in bounded space. If None, sampled via
                :meth:`Objective.random_params_bounded`.
            random_seed: Seed used when sampling ``init_params``.
            rhobeg: Initial size of the linear-approximation trust region.
            tol: Final accuracy target for the trust-region radius.
            catol: Absolute tolerance for constraint violations.
            maxiter: Maximum number of objective evaluations / iterations.
            **scipy_kwargs: Additional SciPy COBYLA option entries forwarded via
                ``options=...``.
        """
        self._run_scipy_minimize(
            problem_objective,
            init_params,
            random_seed,
            None,
            {
                "rhobeg": rhobeg,
                "tol": tol,
                "catol": catol,
                "maxiter": maxiter,
                **scipy_kwargs,
            },
        )
