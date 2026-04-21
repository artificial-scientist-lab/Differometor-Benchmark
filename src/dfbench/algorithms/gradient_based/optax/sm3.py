"""SM3 optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxSM3(OptaxAlgorithm):
    """SM3 optimizer via Optax.

    Memory-efficient adaptive optimizer that maintains only O(d) instead
    of O(d^2) second-moment information (Anil et al., 2019).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_sm3"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.sm3(learning_rate),
            grad_clip_norm=grad_clip_norm,
        )
