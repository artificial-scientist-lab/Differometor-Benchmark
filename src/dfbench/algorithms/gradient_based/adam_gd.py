import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from differometor.utils import sigmoid_bounding
from jaxtyping import Array, Float
from wandb import Run

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
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
        learning_rate: float = 0.1,
        patience: int = 1000,
        save_path: Path | None = None,
        wandb_run: Run | None = None,
        use_arcsinh_transform: bool = False,
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
            save_path: Path to save run data.
            wandb_run: Optional Weights & Biases run for logging.
            use_arcsinh_transform: If True, optimize arcsinh(loss) instead of the raw loss. This compresses large loss values and can improve gradient behaviour.
            **adam_kwargs: Passed to optax.adam().
        """
        obj = problem_objective
        problem = obj.problem

        _, key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            unconstrained_params = obj.random_params_unbounded(rng_key=key)
        else:
            unconstrained_params = init_params
        best_unconstrained_params = unconstrained_params

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(unconstrained_params)

        # Set up transformed objective when requested
        if use_arcsinh_transform:
            raw_value_fn = problem.sigmoid_objective_function

            def opt_value_fn(params):
                return jnp.arcsinh(raw_value_fn(params))

            opt_value_and_grad_fn = jax.jit(jax.value_and_grad(opt_value_fn))

        # Warm-up JIT
        if use_arcsinh_transform:
            opt_loss, _ = opt_value_and_grad_fn(unconstrained_params)
            initial_loss = jnp.sinh(opt_loss)
        else:
            initial_loss, _ = obj.value_and_grad(unconstrained_params)  # Warm-up JIT

        obj.start_logging()

        i = 1
        best_loss = initial_loss
        while not obj.budget_exceeded:
            iter_start_time = time.time()
            if use_arcsinh_transform:
                opt_loss, grads = opt_value_and_grad_fn(unconstrained_params)
                loss = jnp.sinh(opt_loss)
                obj.log_evaluation(unconstrained_params, loss, grads)
            else:
                loss, grads = obj.value_and_grad(
                    unconstrained_params
                )  # Use value_and_grad, else the loss is not logged!

            if loss < best_loss:
                best_loss = loss
                best_unconstrained_params = unconstrained_params

            # Early stopping: patience check using Objective's improvement tracker
            if obj.evals_since_improvement > patience:
                break

            updates, optimizer_state = optimizer.update(
                grads, optimizer_state, unconstrained_params
            )
            unconstrained_params = optax.apply_updates(unconstrained_params, updates)
            iter_time = time.time() - iter_start_time

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "loss": float(loss),
                        "best_loss": float(best_loss),
                        "iter_time": iter_time,
                        "time_elapsed": obj.time_elapsed,
                        "evals_since_improvement": obj.evals_since_improvement,
                    },
                    step=i,
                )
            i += 1

        best_constrained_params = sigmoid_bounding(
            best_unconstrained_params, problem.bounds
        )

        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "final/best_loss": float(best_loss),
                    "final/total_evals": obj.eval_count,
                    "final/total_time": obj.time_elapsed,
                    "final/improvement_count": obj.improvement_count,
                },
            )

        if save_path is not None:
            save_path = Path(str(save_path) + ".npz")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                save_path,
                best_constrained_params=np.array(best_constrained_params),
                best_unconstrained_params=np.array(best_unconstrained_params),
                best_loss=np.array(best_loss),
            )
