"""NadamW optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxNadamW(OptaxAlgorithm):
    """NadamW (Nesterov Adam with decoupled weight decay) optimizer via Optax.

    Nadam with decoupled weight decay.
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, eps, weight_decay, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_nadamw"

    def _make_optimizer(self, learning_rate=0.01, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.nadamw(
                learning_rate,
                b1=kw.get("b1", 0.9),
                b2=kw.get("b2", 0.999),
                eps=kw.get("eps", 1e-8),
                weight_decay=kw.get("weight_decay", 1e-4),
            ),
            grad_clip_norm=grad_clip_norm,
        )
