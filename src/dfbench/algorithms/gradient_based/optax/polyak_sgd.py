"""PolyakSGD optimizer (Optax)."""

import optax

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)


class OptaxPolyakSGD(OptaxAlgorithm):
    """Polyak Step-Size SGD via Optax.

    Uses the Polyak step-size rule: step = (f(x) - f*) / ||g||^2,
    where f* is the (estimated) optimal value (``optax.polyak_sgd``).
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        max_learning_rate, f_min (target minimum), grad_clip_norm, patience.

    Note:
        ``polyak_sgd`` requires passing the current loss value to each
        ``optimizer.update`` call. This algorithm overrides the standard
        loop to supply the loss.
    """

    algorithm_str: str = "optax_polyak_sgd"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        parts = []
        if grad_clip_norm is not None:
            parts.append(optax.clip_by_global_norm(grad_clip_norm))
        parts.append(
            optax.polyak_sgd(
                max_learning_rate=kw.get("max_learning_rate", learning_rate),
                f_min=kw.get("f_min", 0.0),
            )
        )
        return optax.chain(*parts) if len(parts) > 1 else parts[0]

    def optimize(
        self,
        problem_objective,
        init_params=None,
        random_seed=None,
        patience=None,
        learning_rate=0.1,
        grad_clip_norm=1.0,
        **kwargs,
    ):
        """Polyak SGD loop — passes loss to ``optimizer.update``."""
        obj = problem_objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_unbounded() * (1 + 1e-8)
        else:
            params = init_params

        optimizer = self._make_optimizer(
            learning_rate=learning_rate,
            grad_clip_norm=grad_clip_norm,
            **kwargs,
        )
        opt_state = optimizer.init(params)

        # JIT warmup
        _ = obj.value_and_grad(params)

        obj.start_logging()

        while not obj.budget_exceeded:
            loss, grads = obj.value_and_grad(params)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            updates, opt_state = optimizer.update(
                grads, opt_state, params, value=loss
            )
            params = optax.apply_updates(params, updates)
