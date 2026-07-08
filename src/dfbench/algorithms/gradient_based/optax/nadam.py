"""Nadam optimizer (Optax)."""

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
    optax,
)


class OptaxNadam(OptaxAlgorithm):
    """Nadam (Nesterov Adam) optimizer via Optax.

    Incorporates Nesterov momentum into Adam (Dozat, 2016).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, eps, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_nadam"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.nadam(
                learning_rate,
                b1=kw.get("b1", 0.9),
                b2=kw.get("b2", 0.999),
                eps=kw.get("eps", 1e-8),
            ),
            grad_clip_norm=grad_clip_norm,
        )
