import jax
import jax.numpy as jnp
try:
    import optax
except ImportError as exc:
    raise ImportError(
        "optax is required for this algorithm. Install with:  uv add 'dfbench[optax]'"
    ) from exc
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

    @staticmethod
    def _linesearch_eval_count(opt_state) -> int:
        """Extract the number of internal line-search evaluations.

        Optax's LBFGS state currently stores the zoom line-search info in the
        final element of the optimizer state tuple. We scan from the back to
        keep this resilient to minor chain layout changes.
        """
        states = opt_state if isinstance(opt_state, tuple) else (opt_state,)
        for state in reversed(states):
            info = getattr(state, "info", None)
            if info is None:
                continue

            num_steps = getattr(info, "num_linesearch_steps", None)
            if num_steps is not None:
                return int(jax.device_get(num_steps))

        return 0

    def __init__(self) -> None:
        """Initialize L-BFGS optimizer."""
        pass

    def optimize(
        self,
        objective: Objective,
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

        Each iteration performs one explicit ``value_and_grad`` call plus any
        additional internal line-search probes used by Optax. The explicit
        evaluation is logged with the true ``loss`` and ``grad``; the internal
        probes are replay-logged using the same values so the Objective's time
        and evaluation histories reflect the real call count closely enough for
        downstream time-based analysis.

        Args:
            objective: The Objective instance wrapping the problem.
            init_params: Initial parameters. If None, initialize randomly
                (using random_seed) in unbounded space.
            random_seed: Seed for reproducibility. If None, uses system
                entropy.
            patience: Stop after this many iterations without improvement.
            **lbfgs_kwargs: Passed to ``optax.lbfgs()``.
        """
        obj = objective

        random_seed, _ = self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_unbounded()
        else:
            params = init_params

        # Build the JIT-compiled step using an unlogged value function,
        # since optax.lbfgs needs a raw callable for line-search.
        value_fn = obj.value_function(unbounded=True)
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
        warmup_optimizer_state = optimizer.init(obj.random_params_unbounded())
        _, warmup_optimizer_state, _, _ = _step(
            obj.random_params_unbounded(), warmup_optimizer_state
        )
        _ = _step(obj.random_params_unbounded(), warmup_optimizer_state)

        obj.start_logging()

        while not obj.budget_exceeded:
            prior_params = params
            params, optimizer_state, loss, grads = _step(params, optimizer_state)

            # Log the visible value_and_grad call first.
            obj.log_evaluation(prior_params, loss, grads)

            # Replay Optax's internal line-search probes so eval/time accounting
            # matches the true compute spent in each LBFGS iteration.
            # NOTE: this is not 100% accurate.
            extra_evals = self._linesearch_eval_count(optimizer_state)
            for _ in range(extra_evals):
                if obj.budget_exceeded:
                    break
                obj.log_evaluation(prior_params, loss, grads)

            # Early stopping: patience check using Objective's improvement tracker
            if patience is not None and obj.evals_since_improvement > patience:
                break
