from __future__ import annotations

import optax
from jaxtyping import Array, Float

from learned_optimization.research.general_lopt import prefab

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


class VeLO(OptimizationAlgorithm):
    """VeLO Optimizer from the paper:
    https://arxiv.org/pdf/2211.09760

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("adam_gd").
        algorithm_type (AlgorithmType): Type classification (GRADIENT_BASED).

    Note:
        This algorithm uses the Objective's unbounded optimization mode which applies
        sigmoid bounding internally, allowing the optimizer to search in (-∞, +∞) space.
    """

    algorithm_str: str = "velo"
    algorithm_type: AlgorithmType = AlgorithmType.LEARNED

    def __init__(self) -> None:
        """Initialize VeLO optimizer."""
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        patience: int | None = None,
    ) -> None:
        """Run VeLO using `Objective` for logging.

        Args:
            objective: The Objective instance wrapping the problem.
            init_params: Initial parameters. If None, initialize randomly (using random_seed).
            random_seed: Seed for init param generation.
        """
        random_seed, _ = self.prepare(
            objective, unbounded=True, random_seed=random_seed
        )
        num_steps = objective.evals_left
        if num_steps is None:
            raise ValueError("VeLO requires an Objective with max_evals set.")
        optimizer = prefab.optax_lopt(num_steps=num_steps)

        if init_params is None:
            params = objective.random_params_unbounded()
        else:
            params = init_params

        optimizer_state = optimizer.init(params)

        # Warm-up JIT
        objective.warmup_value_and_grad()

        objective.start_logging()

        while not objective.budget_exceeded:
            loss, grads = objective.value_and_grad(
                params
            )  # Use value_and_grad, else the loss is not logged!

            # Early stopping: patience check using Objective's improvement tracker
            if patience is not None and objective.evals_since_improvement > patience:
                break

            updates, optimizer_state = optimizer.update(
                grads, optimizer_state, params, extra_args={"loss": loss}
            )
            params = optax.apply_updates(params, updates)
