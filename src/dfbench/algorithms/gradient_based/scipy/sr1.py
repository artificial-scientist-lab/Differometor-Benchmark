"""SciPy SR1 optimizer implemented via trust-constr."""

from __future__ import annotations

from scipy.optimize import SR1 as ScipySR1
from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based.scipy._common import (
    SciPyConfig,
    ScipyMinimizeAlgorithm,
)
from dfbench.core.objective import Objective


class SR1(ScipyMinimizeAlgorithm):
    """SciPy SR1 via ``trust-constr`` in bounded physical space."""

    algorithm_str = "sr1"
    scipy_config = SciPyConfig(
        method="trust-constr",
        unbounded=False,
        use_bounds=True,
    )

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        gtol: float = 1e-6,
        xtol: float = 1e-8,
        barrier_tol: float = 1e-8,
        maxiter: int | None = None,
        initial_tr_radius: float = 1.0,
        initial_constr_penalty: float = 1.0,
        min_denominator: float = 1e-8,
        init_scale: str | float = "auto",
        tol: float | None = None,
        **scipy_kwargs,
    ) -> None:
        """Run ``trust-constr`` with SciPy's SR1 Hessian update strategy.

        The Objective remains in bounded physical space, SciPy receives the
        problem's box bounds, and second-order information is approximated by
        :class:`scipy.optimize.SR1`.

        Args:
            objective: Objective to mutate in place with evaluation logs.
            init_params: Initial point in bounded space. If None, sampled via
                :meth:`Objective.random_params_bounded`.
            random_seed: Seed used when sampling ``init_params``.
            gtol: Tolerance on the infinity norm of the Lagrangian gradient.
            xtol: Stop when the trust-region radius falls below this threshold.
            barrier_tol: Barrier-parameter termination tolerance.
            maxiter: Maximum number of trust-constr iterations.
            initial_tr_radius: Initial trust-region radius.
            initial_constr_penalty: Initial merit-function penalty weight.
            min_denominator: Safeguard threshold used by SciPy's SR1 update.
            init_scale: Initial scaling passed to :class:`scipy.optimize.SR1`.
            tol: Top-level ``scipy.optimize.minimize`` tolerance.
            **scipy_kwargs: Additional SciPy trust-constr option entries
                forwarded via ``options=...``.
        """
        self._run_scipy_minimize(
            objective,
            init_params,
            random_seed,
            tol,
            {
                "gtol": gtol,
                "xtol": xtol,
                "barrier_tol": barrier_tol,
                "maxiter": maxiter,
                "initial_tr_radius": initial_tr_radius,
                "initial_constr_penalty": initial_constr_penalty,
                **scipy_kwargs,
            },
            hessian_update_strategy=ScipySR1(
                min_denominator=min_denominator,
                init_scale=init_scale,
            ),
        )
