"""Shared base class for Optax-based first-order gradient optimizers.

All Optax algorithms in this batch inherit from `OptaxAlgorithm` which
provides the boilerplate: optimizer creation, optional gradient clipping,
optional learning-rate warmup schedule, and a standard train loop that calls
``obj.value_and_grad`` once per iteration.

Algorithms that need a custom loop (e.g. LBFGS, SAM, Lookahead) can
override ``optimize`` but still reuse the helpers.
"""

from __future__ import annotations

import jax
import optax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


def build_optimizer(
    base: optax.GradientTransformation,
    *,
    grad_clip_norm: float | None = 1.0,
    warmup_steps: int = 0,
    peak_lr: float | None = None,
) -> optax.GradientTransformation:
    """Chain optional gradient clipping and LR warmup around a base optimizer.

    Args:
        base: The core Optax GradientTransformation (e.g. ``optax.adam(lr)``).
        grad_clip_norm: If not None, prepend ``clip_by_global_norm``.
        warmup_steps: If > 0, prepend a linear warmup schedule that scales
            the learning rate from 0 to ``peak_lr`` over this many steps.
        peak_lr: Used only when ``warmup_steps > 0``.

    Returns:
        A chained ``GradientTransformation``.
    """
    parts: list[optax.GradientTransformation] = []

    if grad_clip_norm is not None:
        parts.append(optax.clip_by_global_norm(grad_clip_norm))

    if warmup_steps > 0 and peak_lr is not None:
        schedule = optax.linear_schedule(
            init_value=0.0, end_value=1.0, transition_steps=warmup_steps
        )
        parts.append(optax.scale_by_schedule(schedule))

    parts.append(base)
    return optax.chain(*parts) if len(parts) > 1 else base


# Maximum consecutive NaN/Inf evaluations before falling back to the
# best-known point (or a fresh random start).
_MAX_NAN_STREAK: int = 20

# NaN perturbation base scale.  Starts miniscule (1e-10) and doubles each
# consecutive miss, mirroring the escalation used elsewhere in dfbench.
_NAN_PERTURB_BASE: float = 1e-10


def _is_nonfinite(loss, grads) -> bool:
    """Return True if loss or any gradient entry is NaN or Inf."""
    return bool(not jnp.isfinite(loss) or not jnp.all(jnp.isfinite(grads)))


class OptaxAlgorithm(OptimizationAlgorithm):
    """Thin base class for single-step Optax optimizers.

    Subclasses only need to set ``algorithm_str`` and implement
    ``_make_optimizer`` which returns an ``optax.GradientTransformation``.

    The standard ``optimize`` loop is:

    1. ``prepare(obj, unbounded=True)``
    2. init params (random unbounded)
    3. JIT warmup ``value_and_grad``
    4. ``obj.start_logging()``
    5. ``while not obj.budget_exceeded``: one ``value_and_grad`` + update

    When the objective returns non-finite (NaN/Inf) values, the optimizer
    state is left untouched and the parameters are randomly perturbed with
    an escalating scale.  After ``_MAX_NAN_STREAK`` consecutive failures
    the loop falls back to the best-known point (or a fresh random start)
    and reinitialises the optimizer state.

    Algorithms that need extra logic (e.g. two-step SAM, Lookahead slow
    weights) should override ``optimize`` entirely.
    """

    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    # -- subclass hook -------------------------------------------------------

    def _make_optimizer(
        self,
        learning_rate: float,
        grad_clip_norm: float | None,
        **kwargs,
    ) -> optax.GradientTransformation:
        """Return the fully-assembled Optax optimizer.

        Override in every concrete subclass.
        """
        raise NotImplementedError

    # -- standard loop -------------------------------------------------------

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        patience: int | None = None,
        learning_rate: float = 0.1,
        grad_clip_norm: float | None = 1.0,
        **kwargs,
    ) -> None:
        """Run a standard single-evaluation-per-step Optax loop.

        Args:
            objective: Pre-configured Objective.
            init_params: Starting point.  ``None`` -> random unbounded.
            random_seed: Seed for reproducibility.
            patience: Early-stop after this many evals without improvement.
            learning_rate: Passed to ``_make_optimizer``.
            grad_clip_norm: Max global gradient norm (None to disable).
            **kwargs: Forwarded to ``_make_optimizer``.
        """
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
        opt_state = optimizer.init(params)

        # JIT warmup
        obj.warmup_value_and_grad()

        obj.start_logging()

        nan_streak = 0
        rng_key = jax.random.PRNGKey(random_seed if random_seed is not None else 0)

        while not obj.budget_exceeded:
            loss, grads = obj.value_and_grad(params)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            # --- NaN / Inf guard ------------------------------------------
            if _is_nonfinite(loss, grads):
                nan_streak += 1
                rng_key, sub_key = jax.random.split(rng_key)

                if nan_streak > _MAX_NAN_STREAK:
                    # Fallback: jump to best-known point or fresh random start
                    best = obj.best_params
                    if best is not None:
                        params = (
                            best
                            + jax.random.normal(sub_key, best.shape) * _NAN_PERTURB_BASE
                        )
                    else:
                        params = obj.random_params_unbounded()
                    opt_state = optimizer.init(params)
                    nan_streak = 0
                else:
                    # Escalating perturbation: starts at 1e-10 and doubles
                    # each consecutive miss (capped at 2**30 ≈ 1e9 growth).
                    scale = _NAN_PERTURB_BASE * (2 ** min(nan_streak, 30))
                    params = params + jax.random.normal(sub_key, params.shape) * scale
                continue
            # --------------------------------------------------------------

            nan_streak = 0
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
