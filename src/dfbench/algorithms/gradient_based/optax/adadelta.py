"""AdaDelta optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxAdaDelta(OptaxAlgorithm):
    """AdaDelta optimizer via Optax.

    Extension of AdaGrad that reduces its aggressive, monotonically-decreasing
    learning rate (Zeiler, 2012).  Operates in unbounded space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, rho, eps, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_adadelta"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.adadelta(
                learning_rate,
                rho=kw.get("rho", 0.9),
                eps=kw.get("eps", 1e-6),
            ),
            grad_clip_norm=grad_clip_norm,
        )
