"""LAMB optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxLAMB(OptaxAlgorithm):
    """LAMB (Layer-wise Adaptive Moments) optimizer via Optax.

    Trust-ratio scaled Adam for large-batch training (You et al., 2020).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, eps, weight_decay, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_lamb"

    def _make_optimizer(self, learning_rate=0.01, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.lamb(
                learning_rate,
                b1=kw.get("b1", 0.9),
                b2=kw.get("b2", 0.999),
                eps=kw.get("eps", 1e-6),
                weight_decay=kw.get("weight_decay", 0.0),
            ),
            grad_clip_norm=grad_clip_norm,
        )
