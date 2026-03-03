import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dfbench.core.objective import Objective
from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType


class LBFGSGD(OptimizationAlgorithm):
    """L-BFGS optimization algorithm.

    Implements gradient-based optimization using the L-BFGS optimizer from Optax.
    Includes early stopping based on patience.

    All history tracking, printing, and checkpointing is handled by the
    `Objective` wrapper. The algorithm loop is minimal.

    Because L-BFGS requires calling ``value_and_grad`` *inside* the
    JIT-compiled optimizer step (``optax.lbfgs`` needs the value function
    for its line-search), this algorithm uses ``obj.log_evaluation()``
    to record results after each JIT step rather than calling
    ``obj.value_and_grad()`` directly.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("lbfgs_gd").
        algorithm_type (AlgorithmType): Type classification (GRADIENT_BASED).

    Note:
        This algorithm uses the Objective's unbounded optimization mode which applies
        sigmoid bounding internally, allowing the optimizer to search in (-∞, +∞) space.

    Example:
        >>> problem = VoyagerProblem()
        >>> obj = Objective(problem, max_evals=10_000)
        >>> optimizer = LBFGSGD()
        >>> obj = optimizer.optimize(obj, random_seed=42, patience=500)
    """

    algorithm_str: str = "lbfgs_gd"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        """Initialize L-BFGS optimizer."""
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        patience: int | None = None,
        **lbfgs_kwargs,
    ) -> None:
        """Run L-BFGS using ``Objective`` for logging.

        The optimization step (value_and_grad + optimizer update) is
        ``jax.jit``-compiled for performance.  Because the evaluation
        happens inside the JIT boundary, results are logged *after* each
        step via ``obj.log_evaluation()``.

        Each iteration performs exactly one evaluation, so the evaluation
        budget on the Objective (``max_evals``) directly controls the
        number of L-BFGS steps.

        Args:
            problem_objective: The Objective instance wrapping the problem.
            init_params: Initial parameters. If None, initialize randomly
                (using random_seed) in unbounded space.
            random_seed: Seed for reproducibility. If None, uses system
                entropy.
            patience: Stop after this many iterations without improvement.
            **lbfgs_kwargs: Passed to ``optax.lbfgs()``.
        """
        obj = problem_objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_unbounded()
        else:
            params = init_params

        # Build the JIT-compiled step using the sigmoid objective directly,
        # since optax.lbfgs needs the raw value function for line-search.
        value_fn = problem.sigmoid_objective_function
        value_and_grad_fn = jax.value_and_grad(value_fn)

        optimizer = optax.lbfgs(**lbfgs_kwargs)
        optimizer_state = optimizer.init(params)

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

        # Warm-up JIT
        _ = _step(params, optimizer_state)

        obj.start_logging()

        while not obj.budget_exceeded:
            prior_params = params
            params, optimizer_state, loss, grads = _step(params, optimizer_state)

            # Log using the public API (replaces _log_time + _log_evals + _log_to_file)
            obj.log_evaluation(prior_params, loss, grads)

            # Early stopping: patience check using Objective's improvement tracker
            if obj.evals_since_improvement > patience:
                break
