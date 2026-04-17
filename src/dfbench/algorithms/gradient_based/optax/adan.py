"""Adan optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxAdan(OptaxAlgorithm):
    """Adan optimizer via Optax.

    Adaptive Nesterov Momentum Algorithm (Xie et al., 2023).
    Uses a Nesterov-style momentum estimation.
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, b3, eps, weight_decay, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_adan"

    def _make_optimizer(self, learning_rate=0.01, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.adan(
                learning_rate,
                b1=kw.get("b1", 0.98),
                b2=kw.get("b2", 0.92),
                b3=kw.get("b3", 0.99),
                eps=kw.get("eps", 1e-8),
                weight_decay=kw.get("weight_decay", 0.0),
            ),
            grad_clip_norm=grad_clip_norm,
        )
