"""Exercise scalar Objective logging against the real constrained Voyager model.

This smoke script covers standard value-bearing calls, derivative-only calls,
explicit aux calls, warmups, save-token selection, and history alignment. It
uses a small frequency grid so the real Differometor simulation remains
practical on CPU.

Run from the repository root:

    uv run python scripts/smoke_constrained_voyager_scalar_logging.py
"""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem


AUX_HISTORY_NAMES = (
    "sensitivity_loss_history",
    "penalty_history",
    "is_feasible_history",
    "violations_history",
    "power_hard_history",
    "power_soft_history",
    "power_detector_history",
)


def _assert_finite(*values) -> None:
    for value in values:
        assert bool(jnp.all(jnp.isfinite(jnp.asarray(value))))


def _points(problem: ConstrainedVoyagerProblem):
    lower, upper = jnp.asarray(problem.bounds)
    span = upper - lower
    return (
        lower + 0.50 * span,
        lower + 0.45 * span,
        lower + 0.55 * span,
    )


def run(n_frequencies: int) -> None:
    problem = ConstrainedVoyagerProblem(n_frequencies=n_frequencies)
    midpoint, left, right = _points(problem)
    objective = Objective(
        problem,
        max_evals=20,
        save=["grad", "eval_type", "aux"],
    )

    # Compile every scalar first-order family without recording anything.
    objective.warmup_value()
    objective.warmup_grad()
    objective.warmup_value_and_grad()
    objective.warmup_value_aux()
    objective.warmup_value_and_grad_aux()
    jax.effects_barrier()
    assert objective.eval_count == 0
    assert objective.log_call_count == 0
    assert objective.loss_history == []

    objective.start_logging()

    loss = objective.value(midpoint)
    value, grad = objective.value_and_grad(left)
    grad_only = objective.grad(right)
    aux_loss, aux = objective.value_aux(left)
    aux_value, aux_grad, aux_from_grad = objective.value_and_grad_aux(right)
    callable_loss = objective(midpoint)

    _assert_finite(loss, value, grad, grad_only, aux_loss, aux_value, aux_grad)
    _assert_finite(
        *jax.tree_util.tree_leaves(aux),
        *jax.tree_util.tree_leaves(aux_from_grad),
    )
    assert loss.ndim == 0
    assert value.ndim == 0
    assert grad.shape == (problem.n_params,)
    assert grad_only.shape == (problem.n_params,)
    assert aux_grad.shape == (problem.n_params,)
    np.testing.assert_allclose(
        float(aux_loss),
        float(aux["sensitivity_loss"] + aux["penalty"]),
    )
    np.testing.assert_allclose(
        float(aux_value),
        float(aux_from_grad["sensitivity_loss"] + aux_from_grad["penalty"]),
    )

    expected_aux_presence = [True, True, False, True, True, True]
    for name in AUX_HISTORY_NAMES:
        history = getattr(objective, name)
        assert len(history) == len(expected_aux_presence)
        assert [entry is not None for entry in history] == expected_aux_presence

    assert objective.eval_count == 6
    assert objective.log_call_count == 6
    assert len(objective.loss_history) == 6
    assert len(objective.params_history) == 6
    assert len(objective.grad_history) == 6
    assert len(objective.time_steps) == 6
    assert objective.eval_type_counts == {1: 3, 2: 1, 3: 2}
    assert bool(jnp.isnan(objective.loss_history[2]))
    assert objective.grad_history[0] is None
    assert objective.grad_history[1] is not None
    assert objective.grad_history[2] is not None
    assert objective.grad_history[3] is None
    assert objective.grad_history[4] is not None
    assert objective.grad_history[5] is None
    assert objective.best_loss is not None
    assert objective.best_is_feasible is not None
    assert callable_loss.ndim == 0

    print(
        {
            "script": "scalar_logging",
            "n_frequencies": n_frequencies,
            "n_params": problem.n_params,
            "eval_count": objective.eval_count,
            "log_call_count": objective.log_call_count,
            "eval_type_counts": objective.eval_type_counts,
            "best_loss": float(objective.best_loss),
            "best_is_feasible": objective.best_is_feasible,
            "status": "ok",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-frequencies",
        type=int,
        default=8,
        help="Frequency points for the real Voyager simulation (default: 8).",
    )
    args = parser.parse_args()
    run(args.n_frequencies)


if __name__ == "__main__":
    main()
