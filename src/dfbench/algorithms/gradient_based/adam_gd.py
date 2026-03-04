import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


class AdamGD(OptimizationAlgorithm):
    """Adam Gradient Descent optimization algorithm.

    Implements gradient-based optimization using the Adam optimizer from Optax.
    Includes gradient clipping and early stopping based on patience.

    All history tracking, printing, and checkpointing is handled by the
    `Objective` wrapper. The algorithm loop is minimal.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("adam_gd").
        algorithm_type (AlgorithmType): Type classification (GRADIENT_BASED).

    Note:
        This algorithm uses the Objective's unbounded optimization mode which applies
        sigmoid bounding internally, allowing the optimizer to search in (-∞, +∞) space.
    """

    algorithm_str: str = "adam_gd"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        """Initialize Adam Gradient Descent optimizer."""
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        patience: int | None = None,
        learning_rate: float = 0.1,
        **adam_kwargs,
    ) -> None:
        """Run Adam using `Objective` for logging.

        Each iteration performs exactly one ``value_and_grad`` call, so the
        evaluation budget on the Objective (``max_evals``) directly controls
        the number of gradient steps.

        Args:
            problem_objective: The Objective instance wrapping the problem.
            init_params: Initial parameters. If None, initialize randomly (using random_seed).
            random_seed: Seed for init param generation.
            learning_rate: Adam learning rate.
            patience: Stop after this many iterations without improvement.
            **adam_kwargs: Passed to optax.adam().
        """
        obj = problem_objective

        random_seed, _ = self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_unbounded()
        else:
            params = init_params

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(params)

        # Warm-up JIT
        _ = obj.value_and_grad(params)

        obj.start_logging()

        while not obj.budget_exceeded:
            loss, grads = obj.value_and_grad(
                params
            )  # Use value_and_grad, else the loss is not logged!

            # Early stopping: patience check using Objective's improvement tracker
            if patience is not None and obj.evals_since_improvement > patience:
                break

            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)
