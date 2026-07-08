"""Nevergrad TBPSA: noise-robust baseline.

TBPSA (Test-Based Population-Size Adaptation) is a population-based
derivative-free optimizer from Nevergrad designed for noisy objectives.
It dynamically adapts its population size and is a good control for
landscapes with evaluation noise.

Operates in **bounded physical space**; bounds are passed to Nevergrad
parametrization directly. Unbounded mode is not supported.

This wrapper exposes ``num_evaluations`` (number of repeated evaluations
per candidate for noise averaging) as a first-class hyperparameter.
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
        "Nevergrad is required for TBPSA. Install with: uv add 'dfbench[evolution]'"
    ) from exc


class NevergradTBPSA(OptimizationAlgorithm):
    """TBPSA noise-robust baseline via Nevergrad.

    Candidates are proposed by the TBPSA optimizer. Each candidate is
    evaluated ``num_evaluations`` times through the Objective and the
    averaged loss is reported back. This makes TBPSA suitable as a
    noise-aware control.

    Attributes:
        algorithm_str: Identifier string ("ng_tbpsa").
        algorithm_type: EVOLUTIONARY.

    Hyperparameters exposed via ``optimize()``:
        n_restarts: Number of independent restarts.
        num_evaluations: Repeated evaluations per candidate for averaging.
    """

    algorithm_str: str = "ng_tbpsa"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self) -> None:
        """Initialize the TBPSA wrapper. No constructor hyperparameters."""
        pass

    def optimize(
        self,
        objective: Objective,
        max_iterations: int | None = None,
        random_seed: int | None = None,
        n_restarts: int = 1,
        num_evaluations: int = 1,
    ) -> None:
        """Run TBPSA optimization via Nevergrad.

        Args:
            objective: Pre-configured Objective for function evaluations.
            max_iterations: Cap on total ask/tell iterations across all restarts.
                If None, runs until the Objective budget is exhausted.
            random_seed: Seed for reproducibility.
            n_restarts: Number of independent restarts. Budget is split evenly.
            num_evaluations: Number of repeated evaluations per candidate.
                The averaged loss is reported to the optimizer. Each evaluation
                counts against the Objective budget. Set >1 for noisy problems.
        """
        obj = objective

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        bounds = obj.bounds
        lb = np.asarray(bounds[0], dtype=np.float64)
        ub = np.asarray(bounds[1], dtype=np.float64)
        n_params = int(obj.n_params)

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

            optimizer = ng.optimizers.TBPSA(
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

                # Repeated evaluations for noise averaging; each evaluation
                # is NaN-guarded independently.
                losses = []
                for _ in range(num_evaluations):
                    if obj.budget_exceeded:
                        break
                    loss, _ = safe_evaluate(obj, params_np, lb, ub, rng)
                    losses.append(float(loss))

                avg_loss = float(np.mean(losses)) if losses else float("inf")
                optimizer.tell(candidate, avg_loss)
                step += 1
