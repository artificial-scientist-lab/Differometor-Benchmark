"""Exercise batched aux logging against the real constrained Voyager model.

The script verifies reduced aux, full-batch aux, singleton-batch semantics,
standard and explicit batched methods, derivative-only placeholders, and the
public ``batched_*`` aliases.

Run from the repository root:

    uv run python scripts/smoke_constrained_voyager_batched_logging.py
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


def _batch(problem: ConstrainedVoyagerProblem):
    lower, upper = jnp.asarray(problem.bounds)
    span = upper - lower
    return jnp.stack(
        [
            lower + 0.45 * span,
            lower + 0.55 * span,
        ]
    )


def _stored_aux(objective: Objective, index: int) -> dict:
    return {
        "sensitivity_loss": objective.sensitivity_loss_history[index],
        "penalty": objective.penalty_history[index],
        "is_feasible": objective.is_feasible_history[index],
        "violations": objective.violations_history[index],
        "power_values": {
            "hard": objective.power_hard_history[index],
            "soft": objective.power_soft_history[index],
            "detector": objective.power_detector_history[index],
        },
    }


def _assert_tree_allclose(actual, expected) -> None:
    actual_leaves, actual_tree = jax.tree_util.tree_flatten(actual)
    expected_leaves, expected_tree = jax.tree_util.tree_flatten(expected)
    assert actual_tree == expected_tree
    for actual_leaf, expected_leaf in zip(actual_leaves, expected_leaves, strict=True):
        np.testing.assert_allclose(
            np.asarray(actual_leaf),
            np.asarray(expected_leaf),
        )


def _assert_aux_batch_sizes(objective: Objective, sizes: list[int | None]) -> None:
    for name in AUX_HISTORY_NAMES:
        history = getattr(objective, name)
        assert len(history) == len(sizes)
        for entry, size in zip(history, sizes, strict=True):
            if size is None:
                assert entry is None
            else:
                assert entry is not None
                assert np.asarray(entry).shape[0] == size


def run(n_frequencies: int) -> None:
    problem = ConstrainedVoyagerProblem(n_frequencies=n_frequencies)
    batch = _batch(problem)
    singleton = batch[:1]

    reduced = Objective(
        problem,
        max_evals=20,
        save=["grad", "eval_type", "aux"],
    )
    reduced.warmup_vmap_value_aux(batch_size=2)
    reduced.warmup_vmap_value_and_grad(batch_size=2)
    reduced.warmup_vmap_grad(batch_size=2)
    jax.effects_barrier()
    assert reduced.eval_count == 0
    assert reduced.loss_history == []

    reduced.start_logging()
    losses, returned_aux = reduced.vmap_value_aux(batch)
    values, grads = reduced.vmap_value_and_grad(batch)
    grads_only = reduced.vmap_grad(batch)
    singleton_losses, singleton_aux = reduced.vmap_value_aux(singleton)

    assert losses.shape == (2,)
    assert values.shape == (2,)
    assert grads.shape == (2, problem.n_params)
    assert grads_only.shape == (2, problem.n_params)
    assert singleton_losses.shape == (1,)
    best_index = int(jnp.nanargmin(losses))
    expected_reduced_aux = jax.tree_util.tree_map(
        lambda leaf: leaf[best_index],
        returned_aux,
    )
    _assert_tree_allclose(_stored_aux(reduced, 0), expected_reduced_aux)
    expected_singleton_aux = jax.tree_util.tree_map(
        lambda leaf: leaf[0],
        singleton_aux,
    )
    _assert_tree_allclose(_stored_aux(reduced, 3), expected_singleton_aux)
    np.testing.assert_allclose(
        np.asarray(reduced.params_history[0]),
        np.asarray(batch[best_index]),
    )
    assert np.asarray(reduced.loss_history[0]).shape == ()
    assert np.asarray(reduced.params_history[3]).shape == (problem.n_params,)
    for name in AUX_HISTORY_NAMES:
        history = getattr(reduced, name)
        assert [entry is not None for entry in history] == [True, True, False, True]
    assert reduced.eval_count == 7
    assert reduced.log_call_count == 4
    assert reduced.eval_type_counts == {5: 2, 6: 1, 7: 1}

    full = Objective(
        problem,
        max_evals=20,
        save=[
            "grad",
            "batched_loss",
            "batched_grad",
            "batched_aux",
            "eval_type",
        ],
        save_batched_params_history=True,
    )
    full.warmup_vmap_value(batch_size=2)
    full.warmup_vmap_value_aux(batch_size=1)
    full.warmup_vmap_value_and_grad_aux(batch_size=2)
    jax.effects_barrier()
    assert full.eval_count == 0

    full.start_logging()
    full_losses = full.vmap_value(batch)
    explicit_singleton_losses, explicit_singleton_aux = full.vmap_value_aux(singleton)
    explicit_values, explicit_grads, explicit_aux = full.vmap_value_and_grad_aux(batch)
    alias_values, alias_grads = full.batched_value_and_grad(singleton)
    alias_grad_only = full.batched_grad(singleton)

    assert full_losses.shape == (2,)
    assert explicit_singleton_losses.shape == (1,)
    assert explicit_singleton_aux["violations"].shape[0] == 1
    assert explicit_values.shape == (2,)
    assert explicit_grads.shape == (2, problem.n_params)
    assert explicit_aux["violations"].shape[0] == 2
    assert alias_values.shape == (1,)
    assert alias_grads.shape == (1, problem.n_params)
    assert alias_grad_only.shape == (1, problem.n_params)
    _assert_aux_batch_sizes(full, [2, 1, 2, 1, None])
    assert [np.asarray(entry).shape for entry in full.loss_history] == [
        (2,),
        (1,),
        (2,),
        (1,),
        (1,),
    ]
    assert all(np.asarray(entry).ndim == 2 for entry in full.params_history)
    assert full.grad_history[0] is None
    assert full.grad_history[1] is None
    assert np.asarray(full.grad_history[2]).shape == (2, problem.n_params)
    assert np.asarray(full.grad_history[3]).shape == (1, problem.n_params)
    assert np.asarray(full.grad_history[4]).shape == (1, problem.n_params)
    assert full.eval_count == 7
    assert full.log_call_count == 5
    assert full.eval_type_counts == {5: 2, 6: 1, 7: 2}

    print(
        {
            "script": "batched_logging",
            "n_frequencies": n_frequencies,
            "n_params": problem.n_params,
            "reduced": {
                "eval_count": reduced.eval_count,
                "log_call_count": reduced.log_call_count,
                "best_batch_index": reduced.best_batch_index,
            },
            "full": {
                "eval_count": full.eval_count,
                "log_call_count": full.log_call_count,
                "aux_batch_sizes": [2, 1, 2, 1, None],
            },
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
