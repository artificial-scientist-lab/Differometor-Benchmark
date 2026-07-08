"""Yogi optimizer (Optax)."""

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
    optax,
)


class OptaxYogi(OptaxAlgorithm):
    """Yogi optimizer via Optax.

    Addresses the non-convergence issue of Adam by using an additive
    second-moment update instead of an exponential moving average
    (Zaheer et al., 2018).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, eps, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_yogi"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.yogi(
                learning_rate,
                b1=kw.get("b1", 0.9),
                b2=kw.get("b2", 0.999),
                eps=kw.get("eps", 1e-3),
            ),
            grad_clip_norm=grad_clip_norm,
        )
