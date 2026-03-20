"""SciPy TNC optimizer."""

from __future__ import annotations

from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based._scipy_common import (
    SciPyConfig,
    ScipyMinimizeAlgorithm,
)
from dfbench.core.objective import Objective


class TNC(ScipyMinimizeAlgorithm):
    """SciPy TNC in unbounded sigmoid space."""

    algorithm_str = "tnc"
    scipy_config = SciPyConfig(method="TNC", unbounded=True, use_bounds=False)

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        gtol: float | None = None,
        ftol: float | None = None,
        xtol: float | None = None,
        maxfun: int | None = None,
        tol: float | None = None,
        **scipy_kwargs,
    ) -> None:
        """Run SciPy ``method="TNC"`` in unbounded sigmoid space.

        This wrapper optimizes the Objective's sigmoid-space parameterization,
        so box bounds from the underlying problem are not passed to SciPy.

        Args:
            problem_objective: Objective to mutate in place with evaluation logs.
            init_params: Initial point in unbounded space. If None, sampled via
                :meth:`Objective.random_params_unbounded`.
            random_seed: Seed used when sampling ``init_params``.
            gtol: Precision goal for the projected gradient.
            ftol: Precision goal for the objective value.
            xtol: Precision goal for the solution vector.
            maxfun: Maximum number of function evaluations used by TNC.
            tol: Top-level ``scipy.optimize.minimize`` tolerance.
            **scipy_kwargs: Additional SciPy TNC option entries forwarded via
                ``options=...``.
        """
        self._run_scipy_minimize(
            problem_objective,
            init_params,
            random_seed,
            tol,
            {
                "gtol": gtol,
                "ftol": ftol,
                "xtol": xtol,
                "maxfun": maxfun,
                **scipy_kwargs,
            },
        )
