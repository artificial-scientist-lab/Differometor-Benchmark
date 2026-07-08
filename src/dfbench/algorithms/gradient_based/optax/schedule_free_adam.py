"""Schedule-Free Adam optimizer (Optax contrib)."""

from dfbench.algorithms.gradient_based.optax._common import (
    OptaxAlgorithm,
    optax,
)


class OptaxScheduleFreeAdam(OptaxAlgorithm):
    """Schedule-Free AdamW optimizer via ``optax.contrib.schedule_free_adamw``.

    Eliminates the need for a learning-rate schedule by incorporating a
    schedule-free mechanism (Defazio et al., 2024).
    Operates in unbounded (sigmoid-transformed) space by default.

    At evaluation time, the user should call
    ``optax.contrib.schedule_free_eval_params(opt_state, params)``
    to get the evaluation-ready parameters. Within the benchmark loop
    we evaluate the *fast* (training) params for simplicity; the
    ``schedule_free_eval_params`` call is used only for the final
    reported loss.

    Hyperparameters exposed through ``optimize()``:
        learning_rate, b1, b2, eps, weight_decay,
        warmup_steps, grad_clip_norm, patience.
    """

    algorithm_str: str = "optax_schedule_free_adam"

    def _make_optimizer(self, learning_rate=0.1, grad_clip_norm=1.0, **kw):
        base = optax.contrib.schedule_free_adamw(
            learning_rate,
            b1=kw.get("b1", 0.9),
            b2=kw.get("b2", 0.999),
            eps=kw.get("eps", 1e-8),
            weight_decay=kw.get("weight_decay", 0.0),
            warmup_steps=kw.get("warmup_steps", 0),
        )
        if grad_clip_norm is not None:
            return optax.chain(optax.clip_by_global_norm(grad_clip_norm), base)
        return base
