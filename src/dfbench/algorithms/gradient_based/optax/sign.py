"""SignSGD and Signum optimizers (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxSignSGD(OptaxAlgorithm):
    """SignSGD optimizer via ``optax.sign_sgd``.

    Uses only the sign of the gradient for updates (Bernstein et al., 2018).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_sign_sgd"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.sign_sgd(learning_rate),
            grad_clip_norm=grad_clip_norm,
        )


class OptaxSignum(OptaxAlgorithm):
    """Signum optimizer (SignSGD with momentum) via Optax.

    Applies sign of the momentum buffer rather than the raw gradient
    (Bernstein et al., 2018).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, momentum, grad_clip_norm, patience.

    Note:
        Optax does not have a dedicated ``signum``; this is composed as
        ``chain(trace(momentum), scale_by_sign(), scale(-lr))``.
    """

    algorithm_str: str = "optax_signum"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        momentum = kw.get("momentum", 0.9)
        base = optax.chain(
            optax.trace(decay=momentum, nesterov=False),
            optax.scale_by_sign(),
            optax.scale(-learning_rate),
        )
        if grad_clip_norm is not None:
            return optax.chain(optax.clip_by_global_norm(grad_clip_norm), base)
        return base
