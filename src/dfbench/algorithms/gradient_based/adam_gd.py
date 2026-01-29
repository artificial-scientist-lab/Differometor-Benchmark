import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float

from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)
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
        _problem (ContinuousProblem): The optimization problem instance.

    Note:
        This algorithm uses `problem.sigmoid_objective_function` which expects
        unbounded parameters. The sigmoid bounding is applied internally by the
        objective function, allowing the optimizer to search in (-∞, +∞) space.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = AdamGD(problem)
        >>> objective = optimizer.optimize(
        ...     learning_rate=0.1,
        ...     max_iterations=10000,
        ...     patience=500,
        ... )
    """

    algorithm_str: str = "adam_gd"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(
        self,
        problem: ContinuousProblem,
        verbose: int = 0,
        save_params_history: bool = True,
    ) -> None:
        """Initialize Adam Gradient Descent optimizer.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
            verbose (int): Verbosity level (0=silent, 1+=prints). Defaults to 0.
            save_params_history: Whether to save parameter history. Defaults to True.
        """
        self._problem = problem
        self._verbose = verbose
        self._save_params_history = save_params_history

    def optimize(
        self,
        init_params: Float[Array, "{self._problem.n_params}"] | None = None,
        random_seed: int | None = None,
        max_time: float | None = None,
        learning_rate: float = 0.1,
        max_iterations: int = 50000,
        patience: int = 1000,
        verbose: int | None = None,
        print_every: int = 100,
        plot_loss: bool = False,
        save_run_to_file: bool = False,
        **adam_kwargs,
    ) -> Objective:
        """Run Adam using `Objective` for logging.

        Args:
            init_params: Initial parameters. If None, random in [-10, 10].
            random_seed: Seed for init param generation.
            max_time: Time budget in seconds. None for unlimited.
            learning_rate: Adam learning rate.
            max_iterations: Max iterations (also applied when max_time is set).
            patience: Stop after this many iterations without improvement.
            verbose: Verbosity level (0=silent, 1+=prints via Objective).
            print_every: Print summary every N evaluations.
            plot_loss: If True, call problem.output_to_files for plotting.
            save_run_to_file: If True, call obj.save_run_data for checkpointing.
            **adam_kwargs: Passed to optax.adam().

        Returns:
            The Objective instance with all logged data.
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        params = (
            jnp.array(np.random.uniform(-10, 10, self._problem.n_params))
            if init_params is None
            else init_params
        )

        obj = Objective(
            self._problem,
            unbounded=True,
            max_time=max_time,
            max_evals=max_iterations,
            save_params_history=self._save_params_history,
            print_every=print_every,
            verbose=verbose if verbose is not None else self._verbose,
            algorithm_str=self.algorithm_str,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(params)

        if self._verbose >= 1:
            print(f"Warming up JIT compilation...")
        _ = obj.value_and_grad(params)  # Warm-up JIT

        obj.start_logging()

        while not obj.budget_exceeded:

            loss, grads = obj.value_and_grad(params)

            # Early stopping: patience check using Objective's improvement tracker
            if obj.evals_since_improvement > patience:
                break

            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
            params = optax.apply_updates(params, updates)

        # Outputs
        if plot_loss:
            obj.output_to_files()
        if save_run_to_file:
            obj.save_run_data()

        return obj
