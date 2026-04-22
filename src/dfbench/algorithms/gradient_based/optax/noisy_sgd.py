"""NoisySGD optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxNoisySGD(OptaxAlgorithm):
    """Noisy SGD optimizer via Optax.

    SGD with additive Gaussian noise whose variance decays over time
    (``optax.noisy_sgd``).  The noise can help escape sharp local minima.
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, eta (noise scale), gamma (noise decay),
        grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_noisy_sgd"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.noisy_sgd(
                learning_rate,
                eta=kw.get("eta", 0.01),
                gamma=kw.get("gamma", 0.55),
            ),
            grad_clip_norm=grad_clip_norm,
        )
