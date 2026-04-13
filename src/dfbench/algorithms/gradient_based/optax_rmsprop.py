"""RMSProp optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based._optax_common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxRMSProp(OptaxAlgorithm):
    """RMSProp optimizer via Optax.

    Divides the gradient by a running average of its squared magnitude
    (Hinton, unpublished, 2012).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, decay, eps, momentum, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_rmsprop"

    def _make_optimizer(self, learning_rate=0.01, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.rmsprop(
                learning_rate,
                decay=kw.get("decay", 0.9),
                eps=kw.get("eps", 1e-8),
                momentum=kw.get("momentum", None),
            ),
            grad_clip_norm=grad_clip_norm,
        )
