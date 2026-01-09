import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from collections import deque
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from differometor.utils import sigmoid_bounding
from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)


class AdamGD(OptimizationAlgorithm):
    """Adam Gradient Descent optimization algorithm.

    Implements gradient-based optimization using the Adam optimizer from Optax.
    Includes gradient clipping and early stopping based on patience.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("adam").
        algorithm_type (AlgorithmType): Type classification (GRADIENT_BASED).
        _problem (ContinuousProblem): The optimization problem instance.
        _grad_fn (Callable): JIT-compiled gradient function for the objective.

    Note:
        This algorithm uses `problem.sigmoid_objective_function` which expects
        unbounded parameters. The sigmoid bounding is applied internally by the
        objective function, allowing the optimizer to search in (-∞, +∞) space.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = AdamGD(problem)
        >>> best_params, history, losses, wall_indices = optimizer.optimize(
        ...     learning_rate=0.1,
        ...     max_iterations=10000,
        ...     patience=500,
        ... )
    """

    algorithm_str: str = "adam_gd"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(self, problem: ContinuousProblem) -> None:
        """Initialize Adam Gradient Descent optimizer.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
        """
        self._problem = problem

        self._grad_fn = jax.jit(
            jax.value_and_grad(self._problem.sigmoid_objective_function)
        )

    @jaxtyped(typechecker=typechecker)
    def optimize(
        self,
        save_to_file: bool = True,
        init_params: Float[Array, "{self._problem.n_params}"] | None = None,
        return_best_params_history: bool = False,
        random_seed: int | None = None,
        wall_times: list[int | float] | None = None,
        learning_rate: float = 0.1,
        max_iterations: int = 50000,
        patience: int = 1000,
        **adam_kwargs,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_iters {self._problem.n_params}"] | None,
        Float[Array, "n_iters"],
        list[int] | None,
    ]:
        """Run Adam gradient descent optimization.

        Args:
            save_to_file (bool): Whether to save optimization results to file. Defaults to True.
            init_params (Float[Array, "n_params"] | None): Initial parameters. If None,
                randomly initialized in range [-10, 10]. Defaults to None.
            return_best_params_history (bool): Whether to track best parameters at each
                iteration. Defaults to False.
            random_seed (int | None): Random seed for reproducibility. Controls initial
                parameter generation when init_params is None. Defaults to None.
            wall_times (list[int | float] | None): List of wall-time checkpoints (in seconds).
                The algorithm runs until the maximum checkpoint. At each checkpoint,
                the current iteration index is recorded. If None, runs for max_iterations or
                until patience is exceeded. Defaults to None.
            learning_rate (float): Learning rate for Adam optimizer. Defaults to 0.1.
            max_iterations (int): Maximum number of optimization iterations.
                Only applies when wall_times is None. Defaults to 50,000.
            patience (int): Stop if no improvement (>1e-4) after this many iterations.
                Only applies when wall_times is None. Defaults to 1,000.
            **adam_kwargs: Additional keyword arguments passed to optax.adam().
                Common options: b1 (float, default 0.9), b2 (float, default 0.999),
                eps (float, default 1e-8), eps_root (float, default 0.0),
                nesterov (bool, default False).

        Returns:
            tuple: A 4-tuple containing:
                - best_params (Float[Array, "n_params"]): Best parameters found.
                - best_params_history (Float[Array, "n_iters n_params"] | None): History of
                  best parameters per iteration. None if return_best_params_history=False.
                - losses (Float[Array, "n_iters"]): Loss at each iteration.
                - wall_time_indices (list[int] | None): Iteration indices corresponding to
                  each wall_times checkpoint (in sorted ascending order). None if wall_times is None.
        """
        # Set random seed if provided (affects initial parameter generation)
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize parameters
        best_params: Float[Array, "{self._problem.n_params}"] = (
            jnp.array(np.random.uniform(-10, 10, self._problem.n_params))
            if init_params is None
            else init_params
        )

        # warmup the function to compile it
        _ = self._grad_fn(best_params)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(best_params)

        params, losses = best_params, []
        best_params_history = []  # later shape: (n_iterations, n_params)
        best_loss = 1e10

        # Initialize wall_time_indices tracking
        wall_time_indices: list[int] | None = None
        wall_times_remaining: deque[int | float] | None = None
        if wall_times is not None:
            wall_time_indices = []
            wall_times_remaining = deque(sorted(wall_times))
            max_wall_time = wall_times_remaining[-1]

        # Separate loops for wall-time constrained vs iteration/patience constrained
        if wall_times is not None:
            # Wall-time constrained: ignore max_iterations and patience
            start_time = time.time()
            i = 0
            while (time.time() - start_time) < max_wall_time:
                elapsed = time.time() - start_time

                # Record iteration index at wall_times checkpoints
                while wall_times_remaining and elapsed >= wall_times_remaining[0]:
                    wall_time_indices.append(i)
                    wall_times_remaining.popleft()

                loss, grads = self._grad_fn(params)

                if i % 100 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                if loss < best_loss - 1e-4:
                    best_loss, best_params = loss, params
                    print(f"Iteration {i}: New best loss = {loss}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                updates, optimizer_state = optimizer.update(
                    grads, optimizer_state, params
                )
                params = optax.apply_updates(params, updates)
                losses.append(float(loss))
                i += 1

            # Fill remaining wall_times that weren't reached with final iteration
            while wall_times_remaining:
                wall_time_indices.append(i - 1 if i > 0 else 0)
                wall_times_remaining.popleft()
        else:
            # Iteration/patience constrained
            no_improve_count = 0
            for i in range(max_iterations):
                loss, grads = self._grad_fn(params)

                if i % 500 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                if loss < best_loss - 1e-4:
                    best_loss, best_params, no_improve_count = loss, params, 0
                    print(f"Iteration {i}: New best loss = {loss}")
                else:
                    no_improve_count += 1

                if return_best_params_history:
                    best_params_history.append(best_params)

                updates, optimizer_state = optimizer.update(
                    grads, optimizer_state, params
                )
                params = optax.apply_updates(params, updates)
                losses.append(float(loss))

                if no_improve_count > patience:
                    break

        losses = jnp.array(losses)
        best_params_history = (
            jnp.array(best_params_history) if return_best_params_history else None
        )

        # Transform unbounded parameters back to bounded space
        best_params_bounded = sigmoid_bounding(best_params, self._problem.bounds)
        if return_best_params_history:
            sigmoid_bounding_v = jax.vmap(
                lambda p: sigmoid_bounding(p, self._problem.bounds)
            )
            best_params_history_bounded = sigmoid_bounding_v(best_params_history)
        else:
            best_params_history_bounded = None

        if save_to_file:
            self._problem.output_to_files(
                best_params=best_params_bounded,
                losses=losses,
                algorithm_str=self.algorithm_str,
                hyper_param_str=f"lr{learning_rate}",
            )  # TODO maybe conditionally add more hyperparameters to string

        return best_params_bounded, best_params_history_bounded, losses, wall_time_indices
