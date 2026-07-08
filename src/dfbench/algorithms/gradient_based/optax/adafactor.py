"""Adafactor optimizer (Optax)."""


from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
    optax,
)


class OptaxAdafactor(OptaxAlgorithm):
    """Adafactor optimizer via Optax.

    Memory-efficient adaptive optimizer that factorises the second-moment
    estimate (Shazeer & Stern, 2018).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, decay_rate, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_adafactor"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.adafactor(
                learning_rate=learning_rate,
                decay_rate=kw.get("decay_rate", 0.8),
            ),
            grad_clip_norm=grad_clip_norm,
        )
