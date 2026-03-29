"""OAdam (Optimistic Adam) optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based._optax_common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxOAdam(OptaxAlgorithm):
    """Optimistic Adam via ``optax.optimistic_adam``.

    Combines the optimistic gradient update with Adam's adaptive
    learning rates (Daskalakis et al., 2018).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, eps, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_oadam"

    def _make_optimizer(self, learning_rate=0.01, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.optimistic_adam(
                learning_rate,
                b1=kw.get("b1", 0.9),
                b2=kw.get("b2", 0.999),
                eps=kw.get("eps", 1e-8),
            ),
            grad_clip_norm=grad_clip_norm,
        )
