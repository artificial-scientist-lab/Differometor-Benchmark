"""Lookahead optimizer wrapper (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)
from dfbench.core.objective import Objective


class OptaxLookahead(OptaxAlgorithm):
    """Lookahead optimizer via ``optax.lookahead``.

    Wraps a fast inner optimizer with slow-weight averaging
    (Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back", 2019).
    This is *not* a standalone optimizer — it wraps a user-configurable
    base Optax optimizer (default: Adam).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate (for the inner optimizer), sync_period (k),
        slow_step_size (alpha), inner_optimizer_name,
        grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_lookahead"

    _INNER_MAP = {
        "adam": optax.adam,
        "adamw": optax.adamw,
        "sgd": optax.sgd,
        "rmsprop": optax.rmsprop,
        "lion": optax.lion,
    }

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        inner_name = kw.get("inner_optimizer_name", "adam")
        inner_fn = self._INNER_MAP.get(inner_name)
        if inner_fn is None:
            raise ValueError(
                f"Unknown inner optimizer '{inner_name}'. "
                f"Choose from {list(self._INNER_MAP)}"
            )
        inner = inner_fn(learning_rate)
        if grad_clip_norm is not None:
            inner = optax.chain(optax.clip_by_global_norm(grad_clip_norm), inner)

        return optax.lookahead(
            inner,
            sync_period=kw.get("sync_period", 6),
            slow_step_size=kw.get("slow_step_size", 0.5),
        )

    def optimize(
        self,
        problem_objective: Objective,
        init_params=None,
        random_seed=None,
        patience=None,
        learning_rate=0.1,
        grad_clip_norm=1.0,
        **kwargs,
    ):
        """Lookahead loop — uses LookaheadParams wrapper for init."""
        obj = problem_objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_unbounded() * (1 + 1e-8)
        else:
            params = init_params

        optimizer = self._make_optimizer(
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            **kwargs,
        )

        # Lookahead expects a LookaheadParams named tuple (fast, slow)
        la_params = optax.LookaheadParams.init_synced(params)
        opt_state = optimizer.init(la_params)

        # JIT warmup
        _ = obj.value_and_grad(params)

        obj.start_logging()

        while not obj.budget_exceeded:
            loss, grads = obj.value_and_grad(la_params.fast)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            # Lookahead expects plain gradients, not LookaheadParams
            updates, opt_state = optimizer.update(grads, opt_state, la_params)
            la_params = optax.apply_updates(la_params, updates)
