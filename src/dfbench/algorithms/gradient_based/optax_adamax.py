"""AdaMax optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based._optax_common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxAdaMax(OptaxAlgorithm):
    """AdaMax optimizer via Optax.

    Variant of Adam based on the infinity norm (Kingma & Ba, 2015).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, eps, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_adamax"

    def _make_optimizer(self, learning_rate=0.01, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.adamax(
                learning_rate,
                b1=kw.get("b1", 0.9),
                b2=kw.get("b2", 0.999),
                eps=kw.get("eps", 1e-8),
            ),
            grad_clip_norm=grad_clip_norm,
        )
