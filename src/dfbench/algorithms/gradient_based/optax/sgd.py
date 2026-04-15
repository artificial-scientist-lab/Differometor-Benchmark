"""SGD, SGDM, NAG optimizers (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxSGD(OptaxAlgorithm):
    """Plain Stochastic Gradient Descent via Optax.

    Vanilla SGD without momentum.
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_sgd"

    def _make_optimizer(self, learning_rate=0.01, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.sgd(learning_rate, momentum=0.0, nesterov=False),
            grad_clip_norm=grad_clip_norm,
        )


class OptaxSGDM(OptaxAlgorithm):
    """SGD with momentum via Optax.

    Classic momentum SGD (Polyak, 1964).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, momentum, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_sgdm"

    def _make_optimizer(self, learning_rate=0.01, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.sgd(
                learning_rate,
                momentum=kw.get("momentum", 0.9),
                nesterov=False,
            ),
            grad_clip_norm=grad_clip_norm,
        )


class OptaxNAG(OptaxAlgorithm):
    """Nesterov Accelerated Gradient via Optax.

    SGD with Nesterov momentum look-ahead (Nesterov, 1983).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, momentum, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_nag"

    def _make_optimizer(self, learning_rate=0.01, grad_clip_norm=1.0, **kw):
        return build_optimizer(
            optax.sgd(
                learning_rate,
                momentum=kw.get("momentum", 0.9),
                nesterov=True,
            ),
            grad_clip_norm=grad_clip_norm,
        )
