"""OGD (Optimistic Gradient Descent) optimizer (Optax)."""


from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
    optax,
)


class OptaxOGD(OptaxAlgorithm):
    """Optimistic Gradient Descent via ``optax.optimistic_gradient_descent``.

    Uses the optimistic update rule which predicts the next gradient
    from the difference of successive gradients (Rakhlin & Sridharan, 2013).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_ogd"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.optimistic_gradient_descent(learning_rate),
            grad_clip_norm=grad_clip_norm,
        )
