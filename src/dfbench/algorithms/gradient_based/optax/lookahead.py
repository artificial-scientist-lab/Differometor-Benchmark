"""Lookahead optimizer wrapper (Optax)."""

import jax
import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    _is_nonfinite,
    _MAX_NAN_STREAK,
)
from dfbench.core.objective import Objective


class OptaxLookahead(OptaxAlgorithm):
    """Lookahead optimizer via ``optax.lookahead``.

    Wraps a fast inner optimizer with slow-weight averaging
    (Zhang et al., "Lookahead Optimizer: k steps forward, 1 step back", 2019).
    This is *not* a standalone optimizer; it wraps a user-configurable
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
        objective: Objective,
        init_params=None,
        random_seed=None,
        patience=None,
        learning_rate=0.1,
        grad_clip_norm=1.0,
        **kwargs,
    ):
        """Lookahead loop: uses LookaheadParams wrapper for init."""
        obj = objective
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
        obj.warmup_value_and_grad()

        obj.start_logging()

        nan_streak = 0
        rng_key = jax.random.PRNGKey(random_seed if random_seed is not None else 0)

        while not obj.budget_exceeded:
            loss, grads = obj.value_and_grad(la_params.fast)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            if _is_nonfinite(loss, grads):
                nan_streak += 1
                rng_key, sub_key = jax.random.split(rng_key)

                if nan_streak > _MAX_NAN_STREAK:
                    best = obj.best_params
                    if best is not None:
                        params = (
                            best
                            + jax.random.normal(sub_key, best.shape) * learning_rate
                        )
                    else:
                        params = obj.random_params_unbounded()
                    la_params = optax.LookaheadParams.init_synced(params)
                    opt_state = optimizer.init(la_params)
                    nan_streak = 0
                else:
                    scale = learning_rate * (2 ** min(nan_streak, 8))
                    fast = (
                        la_params.fast
                        + jax.random.normal(sub_key, la_params.fast.shape) * scale
                    )
                    la_params = la_params._replace(fast=fast)
                continue

            nan_streak = 0
            # Lookahead expects plain gradients, not LookaheadParams
            updates, opt_state = optimizer.update(grads, opt_state, la_params)
            la_params = optax.apply_updates(la_params, updates)
