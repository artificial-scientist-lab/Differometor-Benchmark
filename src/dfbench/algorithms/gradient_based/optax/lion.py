"""Lion optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxLion(OptaxAlgorithm):
    """Lion (EvoLved Sign Momentum) optimizer via Optax.

    Discovered through program search (Chen et al., 2023).  Uses only the
    sign of the momentum, making updates uniform in magnitude.
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, weight_decay, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_lion"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.lion(
                learning_rate,
                b1=kw.get("b1", 0.9),
                b2=kw.get("b2", 0.99),
                weight_decay=kw.get("weight_decay", 0.0),
            ),
            grad_clip_norm=grad_clip_norm,
        )
