import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, Float

from dfbench.core.objective import Objective
from dfbench.core.protocols import (
    AlgorithmType,
    ContinuousProblem,
    OptimizationAlgorithm,
)
from dfbench.core.utils import inverse_sigmoid_bounding


class LBFGSGD(OptimizationAlgorithm):
    """L-BFGS optimization algorithm.

    Implements gradient-based optimization using the L-BFGS optimizer from Optax.
    Includes early stopping based on patience.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("lbfgs").
        algorithm_type (AlgorithmType): Type classification (GRADIENT_BASED).
        _problem (ContinuousProblem): The optimization problem instance.

    Note:
        This algorithm uses `problem.sigmoid_objective_function` which expects
        unbounded parameters. The sigmoid bounding is applied internally by the
        objective function, allowing the optimizer to search in (-∞, +∞) space.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = LBFGS(problem)
        >>> objective = optimizer.optimize(
        ...     max_iterations=10000,
        ...     patience=500,
        ... )
    """

    algorithm_str: str = "lbfgs_gd"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(
        self,
        problem: ContinuousProblem,
    ) -> None:
        """Initialize L-BFGS optimizer.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
        """
        self._problem = problem

    def optimize(
        self,
        init_params: Float[Array, "{self._problem.n_params}"] | None = None,
        random_seed: int | None = None,
        max_time: float | None = None,
        max_iterations: int | None = None,
        patience: int = 1000,
        verbose: int = 0,
        print_every: int = 100,
        plot_loss: bool = False,
        save_params_history: bool = True,
        save_run_to_file: bool = False,
        **lbfgs_kwargs,
    ) -> Objective:
        """Run L-BFGS using `Objective` for logging.

        Args:
            init_params: Initial parameters. If None, random in [lower, upper] bounds
                of problem.
            random_seed: Seed for init param generation.
            max_time: Time budget in seconds. None for iterations-limited only.
            max_iterations: Max iterations. None for time-limited only.
            patience: Stop after this many iterations without improvement.
            verbose: Verbosity level (0=silent, 1+=prints via Objective).
            print_every: Print summary every N evaluations.
            plot_loss: If True, call problem.output_to_files for plotting.
            save_params_history: Whether to save parameter history.
            save_run_to_file: If True, call obj.save_run_data for checkpointing.
            **lbfgs_kwargs: Passed to optax.lbfgs().

        Raises:
            ValueError: If neither `max_iterations` nor `max_time` is specified, or if
                neither `random_seed` nor `init_params` is provided.

        Returns:
            The Objective instance with all logged data.
        """
        if max_iterations is None and max_time is None:
            raise ValueError("Either `max_iterations` or `max_time` must be specified.")

        if init_params is not None:
            constrained_params = init_params
        elif random_seed is not None:
            constrained_params = jr.uniform(
                jr.PRNGKey(random_seed),
                shape=(self._problem.n_params,),
                minval=self._problem.bounds[0],
                maxval=self._problem.bounds[1],
            )
        else:
            raise ValueError("Either `random_seed` or `init_params` must be specified.")

        # Map to unconstrained space for L-BFGS
        unconstrained_params = inverse_sigmoid_bounding(
            constrained_params, self._problem.bounds
        )

        obj = Objective(
            self._problem,
            unbounded=True,
            max_time=max_time,
            max_evals=max_iterations,
            save_params_history=save_params_history,
            print_every=print_every,
            verbose=verbose,
            algorithm_str=self.algorithm_str,
        )

        optimizer = optax.lbfgs(**lbfgs_kwargs)
        optimizer_state = optimizer.init(unconstrained_params)

        value_fn = self._problem.sigmoid_objective_function
        value_and_grad_fn = jax.value_and_grad(value_fn)

        @jax.jit
        def _step(params, opt_state):
            loss, grads = value_and_grad_fn(params)
            updates, new_opt_state = optimizer.update(
                grads,
                opt_state,
                params,
                value=loss,
                grad=grads,
                value_fn=value_fn,
            )
            new_params = optax.apply_updates(params, updates)
            return jnp.asarray(new_params), new_opt_state, loss, grads

        if verbose >= 1:
            print("Warming up JIT compilation...")
        _ = _step(unconstrained_params, optimizer_state)  # Warm-up JIT

        obj.start_logging()

        while not obj.budget_exceeded:
            prior_unconstrained_params = unconstrained_params
            unconstrained_params, optimizer_state, loss, grads = _step(
                unconstrained_params, optimizer_state
            )
            obj._log_time()
            obj._log_evals(
                prior_unconstrained_params, loss, grads
            )  # Log params associated with loss
            obj._log_to_file()

            # Early stopping: patience check using Objective's improvement tracker
            if obj.evals_since_improvement > patience:
                break

        # Outputs
        if plot_loss:
            obj.output_to_files()
        if save_run_to_file:
            obj.save_run_data()

        return obj
