"""SciPy COBYQA optimizer."""

from __future__ import annotations

from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based.scipy._common import (
    SciPyConfig,
    ScipyMinimizeAlgorithm,
)
from dfbench.core.objective import Objective


class COBYQA(ScipyMinimizeAlgorithm):
    """SciPy COBYQA in bounded physical space."""

    algorithm_str = "cobyqa"
    scipy_config = SciPyConfig(
        method="COBYQA",
        unbounded=False,
        use_bounds=True,
        use_jac=False,
    )

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        initial_tr_radius: float = 1.0,
        final_tr_radius: float = 1e-6,
        maxiter: int | None = None,
        maxfev: int | None = None,
        **scipy_kwargs,
    ) -> None:
        """Run derivative-free SciPy ``method="COBYQA"`` with box bounds.

        COBYQA works directly in bounded physical space and does not request
        gradients from the Objective.

        Args:
            problem_objective: Objective to mutate in place with evaluation logs.
            init_params: Initial point in bounded space. If None, sampled via
                :meth:`Objective.random_params_bounded`.
            random_seed: Seed used when sampling ``init_params``.
            initial_tr_radius: Initial trust-region radius.
            final_tr_radius: Target final trust-region radius.
            maxiter: Maximum number of COBYQA iterations.
            maxfev: Maximum number of objective evaluations.
            **scipy_kwargs: Additional SciPy COBYQA option entries forwarded via
                ``options=...``.
        """
        self._run_scipy_minimize(
            problem_objective,
            init_params,
            random_seed,
            None,
            {
                "initial_tr_radius": initial_tr_radius,
                "final_tr_radius": final_tr_radius,
                "maxiter": maxiter,
                "maxfev": maxfev,
                **scipy_kwargs,
            },
        )
