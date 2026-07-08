"""AdaGrad optimizer (Optax)."""


from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
    optax,
)


class OptaxAdaGrad(OptaxAlgorithm):
    """AdaGrad optimizer via Optax.

    Adapts per-parameter learning rates using accumulated squared gradients
    (Duchi et al., 2011).  Operates in unbounded space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, eps, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_adagrad"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.adagrad(
                learning_rate,
                eps=kw.get("eps", 1e-8),
            ),
            grad_clip_norm=grad_clip_norm,
        )
