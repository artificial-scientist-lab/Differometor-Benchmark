"""SAM (Sharpness-Aware Minimization) optimizer (Optax contrib)."""

import optax
import optax.contrib

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    build_optimizer,
)
from dfbench.core.algorithm import AlgorithmType
from dfbench.core.objective import Objective


class OptaxSAM(OptaxAlgorithm):
    """Sharpness-Aware Minimization via ``optax.contrib.sam``.

    SAM seeks parameters that lie in uniformly-low-loss neighbourhoods
    (Foret et al., 2021).  Each logical SAM step internally alternates
    between an adversarial perturbation (ascent) and the true parameter
    update (descent).  With ``sync_period=2`` (default) the optimizer
    expects two ``update`` calls per logical step: the first is the
    adversarial step, the second is the real update.

    Because each SAM iteration needs *two* ``value_and_grad`` calls,
    this algorithm overrides the standard loop.
    Operates in unbounded (sigmoid-transformed) space by default.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, rho (adversarial step-size), sync_period,
        grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_sam"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        rho = kw.get("rho", 0.05)
        sync_period = kw.get("sync_period", 2)
        base = optax.adam(learning_rate)
        adv = optax.chain(optax.contrib.normalize(), optax.sgd(rho))
        sam_opt = optax.contrib.sam(
            base, adv, sync_period=sync_period, reset_state=True,
        )
        if grad_clip_norm is not None:
            return optax.chain(optax.clip_by_global_norm(grad_clip_norm), sam_opt)
        return sam_opt

    def optimize(
        self,
        problem_objective: Objective,
        init_params=None,
        random_seed=None,
        patience=None,
        learning_rate=0.1,
        grad_clip_norm=1.0,
        **kwargs,
    ):
        """SAM loop — two gradient evaluations per logical step."""
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
            # Adversarial step (perturbation)
            loss, grads = obj.value_and_grad(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            if obj.budget_exceeded:
                break

            # True descent step
            loss2, grads2 = obj.value_and_grad(params)
            updates2, opt_state = optimizer.update(grads2, opt_state, params)
            params = optax.apply_updates(params, updates2)

            if patience is not None and obj.evals_since_improvement > patience:
                break
