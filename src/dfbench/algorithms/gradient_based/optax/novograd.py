"""NovoGrad optimizer (Optax)."""


from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
    optax,
)


class OptaxNovoGrad(OptaxAlgorithm):
    """NovoGrad optimizer via Optax.

    A layer-wise adaptive rate optimizer with low memory footprint
    (Ginsburg et al., 2019).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, eps, weight_decay,
        grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_novograd"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.novograd(
                learning_rate,
                b1=kw.get("b1", 0.95),
                b2=kw.get("b2", 0.98),
                eps=kw.get("eps", 1e-8),
                weight_decay=kw.get("weight_decay", 0.0),
            ),
            grad_clip_norm=grad_clip_norm,
        )
