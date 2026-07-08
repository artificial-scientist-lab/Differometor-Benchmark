"""Nevergrad OnePlusOne (1+1)-ES baseline.

A minimal wrapper around Nevergrad's OnePlusOne algorithm. OnePlusOne is a
lightweight (1+1)-Evolution Strategy that maintains a single candidate and
perturbs it with Gaussian noise, accepting the perturbation only if it improves
the objective. It is one of the simplest derivative-free baselines and serves
as a sanity-check control for rugged landscapes.

Operates in **bounded physical space**; bounds are passed to the Nevergrad
parametrization directly. Unbounded mode is not supported.
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
        "Nevergrad is required for OnePlusOne. Install with: uv add 'dfbench[evolution]'"
    ) from exc


class NevergradOnePlusOne(OptimizationAlgorithm):
    """Lightweight (1+1)-ES baseline via Nevergrad.

    Each iteration, the solver proposes a single candidate, which is evaluated
    through the Objective wrapper for fair benchmark accounting.

    Attributes:
        algorithm_str: Identifier string ("ng_oneplusone").
        algorithm_type: EVOLUTIONARY.

    Hyperparameters exposed via ``optimize()``:
        n_restarts: Number of independent restarts (multistart).
    """

    algorithm_str: str = "ng_oneplusone"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self) -> None:
        """Initialize the OnePlusOne wrapper. No constructor hyperparameters."""
        pass

    def optimize(
        self,
        objective: Objective,
        max_iterations: int | None = None,
        random_seed: int | None = None,
        n_restarts: int = 1,
    ) -> None:
        """Run (1+1)-ES optimization via Nevergrad.

        Args:
            objective: Pre-configured Objective for function evaluations.
            max_iterations: Cap on total optimizer *ask/tell* iterations across
                all restarts. If None, runs until the Objective budget is exhausted.
            random_seed: Seed for reproducibility.
            n_restarts: Number of independent restarts. Budget is split evenly.
        """
        obj = objective

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        bounds = obj.bounds
        lb = np.asarray(bounds[0], dtype=np.float64)
        ub = np.asarray(bounds[1], dtype=np.float64)
        n_params = int(obj.n_params)

        # Budget per restart
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
            # Per-coordinate mutation scale.  Without this, Nevergrad uses an
            # isotropic unit-Gaussian step in physical space, which is wildly
            # mis-scaled on problems like Voyager where bound widths span
            # several orders of magnitude.  We also start small (5% of each
            # box width): a (1+1)-ES with a large initial step has a vanishing
            # acceptance probability in moderate-to-high dimensions and would
            # waste many evaluations just shrinking sigma via the 1/5 rule.
            parametrization.set_mutation(sigma=(0.05 * (ub - lb)))

            optimizer = ng.optimizers.OnePlusOne(
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
