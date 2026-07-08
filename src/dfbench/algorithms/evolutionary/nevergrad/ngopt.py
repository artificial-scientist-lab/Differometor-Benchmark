"""Nevergrad NGOpt: automatic algorithm-selection baseline.

NGOpt is Nevergrad's built-in meta-optimizer that automatically selects and
configures an internal algorithm based on the budget, dimensionality, and
other problem characteristics. It serves as a strong library-default baseline
without manual algorithm tuning.

Operates in **bounded physical space**; bounds are forwarded to the Nevergrad
parametrization. Unbounded mode is not supported.
"""

from __future__ import annotations

import numpy as np

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective
from dfbench.algorithms.evolutionary.nevergrad._common import safe_evaluate

try:
    import nevergrad as ng
except ImportError as exc:
    raise ImportError(
        "Nevergrad is required for NGOpt. Install with: uv add 'dfbench[evolution]'"
    ) from exc


class NevergradNGOpt(OptimizationAlgorithm):
    """NGOpt automatic algorithm-selection baseline via Nevergrad.

    Delegates algorithm choice to Nevergrad's meta-selector. The wrapper
    simply forwards candidate evaluations through the Objective for fair
    benchmark accounting.

    Attributes:
        algorithm_str: Identifier string ("ng_ngopt").
        algorithm_type: EVOLUTIONARY.

    Hyperparameters exposed via ``optimize()``:
        n_restarts: Number of independent restarts.
    """

    algorithm_str: str = "ng_ngopt"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self) -> None:
        """Initialize the NGOpt wrapper. No constructor hyperparameters."""
        pass

    def optimize(
        self,
        objective: Objective,
        max_iterations: int | None = None,
        random_seed: int | None = None,
        n_restarts: int = 1,
    ) -> None:
        """Run NGOpt optimization via Nevergrad.

        Args:
            objective: Pre-configured Objective for function evaluations.
            max_iterations: Cap on total ask/tell iterations across all restarts.
                If None, runs until the Objective budget is exhausted.
            random_seed: Seed for reproducibility.
            n_restarts: Number of independent restarts. Budget is split evenly.
        """
        obj = objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        if not hasattr(problem, "bounds"):
            raise ValueError("NGOpt requires a bounded problem (problem.bounds).")

        bounds = problem.bounds
        lb = np.asarray(bounds[0], dtype=np.float64)
        ub = np.asarray(bounds[1], dtype=np.float64)
        n_params = int(problem.n_params)

        budget_per_restart = max_iterations // n_restarts if max_iterations else None

        # JIT warmup
        obj.warmup_value()

        obj.start_logging()

        for _restart in range(n_restarts):
            if obj.budget_exceeded:
                break

            rng = np.random.default_rng(random_seed + _restart)

            parametrization = ng.p.Array(
                shape=(n_params,),
                lower=lb,
                upper=ub,
            )
            # Per-coordinate mutation scale: without this, Nevergrad uses an
            # isotropic unit-Gaussian step in physical space, which is wildly
            # mis-scaled on problems like Voyager where bound widths span
            # several orders of magnitude.
            parametrization.set_mutation(sigma=(0.3 * (ub - lb)))

            optimizer = ng.optimizers.NGOpt(
                parametrization=parametrization,
                budget=budget_per_restart or 10_000_000,
                num_workers=1,
            )
            optimizer.parametrization.random_state = np.random.RandomState(
                random_seed + _restart
            )

            step = 0
            while not obj.budget_exceeded:
                if budget_per_restart is not None and step >= budget_per_restart:
                    break

                candidate = optimizer.ask()
                params_np = np.asarray(candidate.value, dtype=np.float64)

                loss, _ = safe_evaluate(obj, params_np, lb, ub, rng)

                optimizer.tell(candidate, float(loss))
                step += 1
