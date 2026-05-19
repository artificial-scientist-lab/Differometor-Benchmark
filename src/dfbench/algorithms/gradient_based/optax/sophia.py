"""Sophia optimizer (Optax-compatible local wrapper).

Optax 0.2.4 does not ship ``sophia`` in the main namespace. This module
provides a thin local implementation that follows the Optax
``GradientTransformation`` API so it plugs directly into the standard
``OptaxAlgorithm`` loop.

Reference: Liu et al., "Sophia: A Scalable Stochastic Second-Order
Optimizer for Language Model Pre-training", 2023.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)
from dfbench.core.algorithm import AlgorithmType
from dfbench.core.objective import Objective


# ── Sophia GradientTransformation (local) ────────────────────────────


class _SophiaState:
    """Plain container for Sophia optimiser state."""

    __slots__ = ("mu", "hessian_diag", "count")

    def __init__(self, mu, hessian_diag, count):
        self.mu = mu
        self.hessian_diag = hessian_diag
        self.count = count


def sophia(
    learning_rate: float = 1e-3,
    b1: float = 0.965,
    b2: float = 0.99,
    eps: float = 1e-8,
    gamma: float = 0.01,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Sophia-H style optimizer as an Optax GradientTransformation.

    This implementation approximates Sophia using a diagonal Hessian EMA
    without requiring an explicit Hessian-vector product (the Hessian diagonal
    is updated from the squared gradient, similar to Sophia-G / Adafactor).

    Args:
        learning_rate: Base learning rate.
        b1: Decay rate for the first moment.
        b2: Decay rate for the Hessian diagonal EMA.
        eps: Small constant for numerical stability.
        gamma: Clipping threshold — updates are clipped to [-1/gamma, 1/gamma].
        weight_decay: Decoupled weight decay coefficient.
    """

    def init_fn(params):
        mu = jnp.zeros_like(params)
        h = jnp.zeros_like(params)
        return _SophiaState(mu=mu, hessian_diag=h, count=jnp.array(0, dtype=jnp.int32))

    def update_fn(updates, state, params=None):
        mu = b1 * state.mu + (1 - b1) * updates
        # Approximate diagonal Hessian via squared gradient (Sophia-G style)
        h = b2 * state.hessian_diag + (1 - b2) * updates**2
        count = state.count + 1

        # Clipped Newton-like step
        update = mu / jnp.maximum(gamma * h, eps)
        update = jnp.clip(update, -1.0 / gamma, 1.0 / gamma)

        # Decoupled weight decay
        if weight_decay > 0.0 and params is not None:
            update = update + weight_decay * params

        update = -learning_rate * update
        new_state = _SophiaState(mu=mu, hessian_diag=h, count=count)
        return update, new_state

    return optax.GradientTransformation(init_fn, update_fn)


# ── Algorithm class ──────────────────────────────────────────────────


class OptaxSophia(OptaxAlgorithm):
    """Sophia optimizer (local Optax-compatible implementation).

    A lightweight second-order optimizer that uses a diagonal Hessian
    estimate clipped to prevent extreme steps.  Because optax 0.2.4
    does not include ``sophia``, a local wrapper is used.
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, eps, gamma, weight_decay,
        grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_sophia"

    def _make_optimizer(self, learning_rate=1e-3, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            sophia(
                learning_rate,
                b1=kw.get("b1", 0.965),
                b2=kw.get("b2", 0.99),
                eps=kw.get("eps", 1e-8),
                gamma=kw.get("gamma", 0.01),
                weight_decay=kw.get("weight_decay", 0.0),
            ),
            grad_clip_norm=grad_clip_norm,
        )

    def optimize(
        self,
        objective: Objective,
        init_params=None,
        random_seed=None,
        patience=None,
        learning_rate: float = 1e-3,
        grad_clip_norm=1.0,
        **kwargs,
    ) -> None:
        """Sophia loop — defaults to learning_rate=1e-3."""
        super().optimize(
            objective,
            init_params=init_params,
            random_seed=random_seed,
            patience=patience,
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            **kwargs,
        )
