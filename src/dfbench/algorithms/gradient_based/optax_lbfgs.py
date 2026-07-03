"""L-BFGS optimizer (Optax) — canonical implementation for this batch.

This wraps ``optax.lbfgs()`` following the same JIT-compiled pattern as
the existing ``LBFGSGD`` class but registered under the ``optax_*`` naming
scheme for consistency with the rest of the Optax batch.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


class OptaxLBFGS(OptimizationAlgorithm):
    """L-BFGS optimizer via ``optax.lbfgs``.

    Uses second-order curvature information for faster convergence on
    smooth landscapes.

    Because ``optax.lbfgs`` needs the raw value function for its internal
    line-search, this algorithm JIT-compiles the full optimisation step and
    uses ``obj.log_evaluation()`` to record results (instead of calling
    ``obj.value_and_grad()`` directly).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        patience, and any kwargs accepted by ``optax.lbfgs()``.
    """

    algorithm_str: str = "optax_lbfgs"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    @staticmethod
    def _linesearch_eval_count(opt_state) -> int:
        """Extract internal line-search evaluation count from LBFGS state."""
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

        Each iteration performs one explicit ``value_and_grad`` call plus
        any additional internal line-search probes used by Optax.

        Args:
            objective: Pre-configured Objective.
            init_params: Starting point.  ``None`` → random unbounded.
            random_seed: Seed for reproducibility.
            patience: Early-stop after this many evals without improvement.
            **lbfgs_kwargs: Forwarded to ``optax.lbfgs()``.
        """
        obj = objective

        self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_unbounded()
        else:
            params = init_params

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
            new_params = jnp.asarray(optax.apply_updates(params, updates))
            return new_params, new_opt_state, loss, grads

        # JIT warmup
        warmup_state = optimizer.init(obj.random_params_unbounded())
        _, warmup_state, _, _ = _step(obj.random_params_unbounded(), warmup_state)
        _ = _step(obj.random_params_unbounded(), warmup_state)

        obj.start_logging()

        while not obj.budget_exceeded:
            prior_params = params
            params, optimizer_state, loss, grads = _step(params, optimizer_state)

            obj.log_evaluation(prior_params, loss, grads)

            # Replay internal line-search probes for eval accounting
            extra_evals = self._linesearch_eval_count(optimizer_state)
            for _ in range(extra_evals):
                if obj.budget_exceeded:
                    break
                obj.log_evaluation(prior_params, loss, grads)

            if patience is not None and obj.evals_since_improvement > patience:
                break
