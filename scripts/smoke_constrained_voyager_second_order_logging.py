"""Exercise every second-order logging path on constrained Voyager.

This is intentionally isolated because cold Hessian compilation is expensive
on CPU. It covers scalar and singleton-batched Hessians, combined
value/gradient/Hessian calls, their public ``batched_*`` aliases, warmups,
full-batch histories, and the derivative-only aux ``None`` contract.

Run from the repository root:

    uv run python scripts/smoke_constrained_voyager_second_order_logging.py
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


def run(n_frequencies: int, hessian_batch_size: int) -> None:
    problem = ConstrainedVoyagerProblem(n_frequencies=n_frequencies)
    lower, upper = jnp.asarray(problem.bounds)
    midpoint = lower + 0.5 * (upper - lower)
    singleton = midpoint[None, :]
    objective = Objective(
        problem,
        max_evals=20,
        hessian_batch_size=hessian_batch_size,
        save=[
            "grad",
            "hessian",
            "batched_loss",
            "batched_grad",
            "batched_hessian",
            "batched_aux",
            "eval_type",
        ],
        save_batched_params_history=True,
    )

    objective.warmup_hessian()
    objective.warmup_value_grad_and_hessian()
    objective.warmup_vmap_hessian(batch_size=1)
    objective.warmup_vmap_value_grad_and_hessian(batch_size=1)
    jax.effects_barrier()
    assert objective.eval_count == 0
    assert objective.log_call_count == 0
    assert objective.loss_history == []

    objective.start_logging()
    hessian = objective.hessian(midpoint)
    value, grad, combined_hessian = objective.value_grad_and_hessian(midpoint)
    batched_hessians = objective.vmap_hessian(singleton)
    batched_values, batched_grads, batched_combined_hessians = (
        objective.vmap_value_grad_and_hessian(singleton)
    )
    alias_hessians = objective.batched_hessian(singleton)
    alias_values, alias_grads, alias_combined_hessians = (
        objective.batched_value_grad_and_hessian(singleton)
    )

    _assert_finite(
        hessian,
        value,
        grad,
        combined_hessian,
        batched_hessians,
        batched_values,
        batched_grads,
        batched_combined_hessians,
        alias_hessians,
        alias_values,
        alias_grads,
        alias_combined_hessians,
    )
    assert hessian.shape == (problem.n_params, problem.n_params)
    assert grad.shape == (problem.n_params,)
    assert combined_hessian.shape == (problem.n_params, problem.n_params)
    assert batched_hessians.shape == (1, problem.n_params, problem.n_params)
    assert batched_values.shape == (1,)
    assert batched_grads.shape == (1, problem.n_params)
    assert batched_combined_hessians.shape == (
        1,
        problem.n_params,
        problem.n_params,
    )
    assert alias_hessians.shape == (1, problem.n_params, problem.n_params)
    assert alias_values.shape == (1,)
    assert alias_grads.shape == (1, problem.n_params)
    assert alias_combined_hessians.shape == (
        1,
        problem.n_params,
        problem.n_params,
    )

    expected_aux_presence = [False, True, False, True, False, True]
    for name in AUX_HISTORY_NAMES:
        history = getattr(objective, name)
        assert len(history) == 6
        assert [entry is not None for entry in history] == expected_aux_presence
    assert np.asarray(objective.sensitivity_loss_history[1]).shape == ()
    assert np.asarray(objective.sensitivity_loss_history[3]).shape == (1,)
    assert np.asarray(objective.violations_history[3]).shape[0] == 1

    expected_hessian_shapes = [
        (problem.n_params, problem.n_params),
        (problem.n_params, problem.n_params),
        (1, problem.n_params, problem.n_params),
        (1, problem.n_params, problem.n_params),
        (1, problem.n_params, problem.n_params),
        (1, problem.n_params, problem.n_params),
    ]
    assert [np.asarray(entry).shape for entry in objective.hessian_history] == (
        expected_hessian_shapes
    )
    assert [entry is not None for entry in objective.grad_history] == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
    assert bool(jnp.isnan(objective.loss_history[0]))
    assert bool(jnp.all(jnp.isnan(objective.loss_history[2])))
    assert bool(jnp.all(jnp.isnan(objective.loss_history[4])))
    assert np.asarray(objective.params_history[0]).shape == (problem.n_params,)
    assert np.asarray(objective.params_history[2]).shape == (1, problem.n_params)
    assert objective.eval_count == 6
    assert objective.log_call_count == 6
    assert objective.eval_type_counts == {8: 1, 11: 1, 12: 2, 15: 2}
    assert objective.best_loss is not None
    assert objective.best_is_feasible is not None

    print(
        {
            "script": "second_order_logging",
            "n_frequencies": n_frequencies,
            "n_params": problem.n_params,
            "hessian_batch_size": hessian_batch_size,
            "eval_count": objective.eval_count,
            "log_call_count": objective.log_call_count,
            "eval_type_counts": objective.eval_type_counts,
            "status": "ok",
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-frequencies",
        type=int,
        default=1,
        help="Frequency points; 1 keeps the 48x48 Hessian smoke practical on CPU.",
    )
    parser.add_argument(
        "--hessian-batch-size",
        type=int,
        default=1,
        help="Hessian columns computed together (default: 1, lowest memory).",
    )
    args = parser.parse_args()
    run(args.n_frequencies, args.hessian_batch_size)


if __name__ == "__main__":
    main()
