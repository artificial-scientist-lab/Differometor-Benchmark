"""RProp optimizer (Optax)."""


from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
    optax,
)


class OptaxRProp(OptaxAlgorithm):
    """RProp (Resilient Backpropagation) optimizer via Optax.

    Uses only the sign of the gradient and maintains per-parameter step sizes
    (Riedmiller & Braun, 1993).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, eta_minus, eta_plus, min_step_size, max_step_size,
        grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_rprop"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.rprop(
                learning_rate,
                eta_minus=kw.get("eta_minus", 0.5),
                eta_plus=kw.get("eta_plus", 1.2),
                min_step_size=kw.get("min_step_size", 1e-6),
                max_step_size=kw.get("max_step_size", 50.0),
            ),
            grad_clip_norm=grad_clip_norm,
        )
