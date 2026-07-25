"""Exercise manual logging, budgets, and storage on constrained Voyager.

The script uses real constrained-Voyager aux data to verify unlogged raw
functions, manual aux/None alignment, disabled histories, penalty rebinding,
atomic budget rejection, periodic checkpoints, NPZ/JSON round-trips, resume,
reset, and unbounded-space logging.

Run from the repository root:

    uv run python scripts/smoke_constrained_voyager_manual_storage.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

from dfbench import Objective
from dfbench.problems import ConstrainedVoyagerProblem, zero_penalty


AUX_HISTORY_NAMES = (
    "sensitivity_loss_history",
    "penalty_history",
    "is_feasible_history",
    "violations_history",
    "power_hard_history",
    "power_soft_history",
    "power_detector_history",
)


def _assert_mixed_aux_history(objective: Objective) -> None:
    for name in AUX_HISTORY_NAMES:
        history = getattr(objective, name)
        assert len(history) == 2
        assert history[0] is not None
        assert history[1] is None


def _assert_aux_roundtrip(actual: Objective, expected: Objective) -> None:
    for name in AUX_HISTORY_NAMES:
        actual_history = getattr(actual, name)
        expected_history = getattr(expected, name)
        assert len(actual_history) == len(expected_history)
        for actual_entry, expected_entry in zip(
            actual_history, expected_history, strict=True
        ):
            if expected_entry is None:
                assert actual_entry is None
            else:
                np.testing.assert_allclose(
                    np.asarray(actual_entry),
                    np.asarray(expected_entry),
                )


def _matching_objective(
    problem: ConstrainedVoyagerProblem,
    checkpoint_dir: Path,
    checkpoint_format: str,
) -> Objective:
    return Objective(
        problem,
        save=["grad", "eval_type", "aux"],
        checkpoint_dir=checkpoint_dir,
        checkpoint_format=checkpoint_format,
    )


def run(n_frequencies: int) -> None:
    problem = ConstrainedVoyagerProblem(n_frequencies=n_frequencies)
    lower, upper = jnp.asarray(problem.bounds)
    midpoint = lower + 0.5 * (upper - lower)

    with TemporaryDirectory(prefix="dfbench-voyager-logging-") as tmp:
        checkpoint_dir = Path(tmp)
        objective = Objective(
            problem,
            save=["grad", "eval_type", "aux"],
            save_to_file_every=1,
            checkpoint_dir=checkpoint_dir,
        )
        objective.set_penalty_fn(zero_penalty)

        raw_aux_fn = objective.value_function_aux()
        assert raw_aux_fn is not None
        raw_value_and_grad = jax.jit(jax.value_and_grad(raw_aux_fn, has_aux=True))
        (loss, aux), grad = raw_value_and_grad(midpoint)
        jax.effects_barrier()
        assert objective.eval_count == 0
        assert objective.log_call_count == 0
        assert objective.loss_history == []
        assert float(aux["penalty"]) == 0.0
        assert bool(jnp.all(aux["violations"] == 0.0))

        objective.start_logging()
        objective.log_evaluation(midpoint, loss, grad, aux=aux)
        objective.log_evaluation(midpoint, loss, grad)
        _assert_mixed_aux_history(objective)
        assert objective.eval_count == 2
        assert objective.log_call_count == 2
        assert objective.eval_type_counts == {3: 2}
        assert len(objective.loss_history) == 2
        assert len(objective.grad_history) == 2
        assert len(objective.params_history) == 2
        assert len(objective.time_steps) == 2

        try:
            objective.set_penalty_fn(zero_penalty)
        except RuntimeError as exc:
            assert "before start_logging" in str(exc)
        else:
            raise AssertionError("set_penalty_fn unexpectedly succeeded after logging")

        periodic_paths = list(checkpoint_dir.rglob("*.npz"))
        assert len(periodic_paths) == 1
        npz_path = objective.save_run_data(algorithm_name="voyager_manual_npz")
        assert npz_path.exists()
        assert npz_path.suffix == ".npz"

        loaded_npz = _matching_objective(problem, checkpoint_dir, "npz")
        loaded_npz.load_run_data(npz_path)
        _assert_aux_roundtrip(loaded_npz, objective)
        assert loaded_npz.eval_count == objective.eval_count
        assert loaded_npz.log_call_count == objective.log_call_count
        assert loaded_npz.eval_type_counts == objective.eval_type_counts
        assert loaded_npz.best_eval_index == objective.best_eval_index
        assert loaded_npz.best_is_feasible == objective.best_is_feasible
        loaded_npz.start_logging()
        loaded_npz.log_evaluation(midpoint, loss, grad, aux=aux)
        assert loaded_npz.eval_count == 3
        assert loaded_npz.log_call_count == 3

        json_objective = _matching_objective(problem, checkpoint_dir, "json")
        json_objective.start_logging()
        json_objective.log_evaluation(midpoint, loss, grad, aux=aux)
        json_objective.log_evaluation(midpoint, loss, grad)
        json_path = json_objective.save_run_data(algorithm_name="voyager_manual_json")
        assert json_path.exists()
        assert json_path.suffix == ".json"
        loaded_json = _matching_objective(problem, checkpoint_dir, "json")
        loaded_json.load_run_data(json_path)
        _assert_aux_roundtrip(loaded_json, json_objective)
        assert loaded_json.eval_count == 2
        assert loaded_json.log_call_count == 2
        assert loaded_json.eval_type_counts == {3: 2}

        # A whole manual batch that does not fit is counted but rejected
        # atomically from every standard and aux history.
        stacked_aux = jax.tree_util.tree_map(
            lambda leaf: jnp.stack([leaf, leaf]),
            aux,
        )
        rejected = Objective(
            problem,
            max_evals=1,
            save=["batched_loss", "batched_aux"],
        )
        rejected.start_logging()
        rejected.log_evaluation(
            params=jnp.stack([midpoint, midpoint]),
            loss=jnp.stack([loss, loss]),
            aux=stacked_aux,
        )
        assert rejected.eval_count == 2
        assert rejected.log_call_count == 0
        assert rejected.budget_exceeded
        assert rejected.loss_history == []
        assert rejected.params_history == []
        assert rejected.time_steps == []
        assert all(getattr(rejected, name) == [] for name in AUX_HISTORY_NAMES)

        timed_out = Objective(problem, max_time=0.0, save=["aux"])
        timed_out.start_logging()
        timed_out.log_evaluation(midpoint, loss, aux=aux)
        assert timed_out.time_exceeded
        assert timed_out.eval_count == 0
        assert timed_out.log_call_count == 0
        assert timed_out.loss_history == []

        # Explicit aux remains available without save tokens, but nothing is
        # allocated in aux histories.
        no_aux_storage = Objective(problem)
        no_aux_storage.start_logging()
        returned_loss, returned_aux = no_aux_storage.value_aux(midpoint)
        assert bool(jnp.isfinite(returned_loss))
        assert returned_aux.keys() == aux.keys()
        assert all(getattr(no_aux_storage, name) == [] for name in AUX_HISTORY_NAMES)

        disabled_standard_histories = Objective(
            problem,
            save_time_steps=False,
            save_params_history=False,
            save=["is_feasible"],
        )
        disabled_standard_histories.start_logging()
        disabled_standard_histories.log_evaluation(midpoint, loss, aux=aux)
        assert len(disabled_standard_histories.loss_history) == 1
        assert len(disabled_standard_histories.is_feasible_history) == 1
        assert disabled_standard_histories.params_history == []
        assert disabled_standard_histories.time_steps == []

        unbounded = Objective(problem, unbounded=True, save=["is_feasible"])
        raw_midpoint = jnp.zeros(problem.n_params)
        unbounded.start_logging()
        unbounded_loss = unbounded.value(raw_midpoint)
        assert bool(jnp.isfinite(unbounded_loss))
        assert len(unbounded.is_feasible_history) == 1
        np.testing.assert_allclose(
            np.asarray(unbounded.params_history[0]),
            np.asarray(raw_midpoint),
        )

        loaded_json.reset()
        assert loaded_json.eval_count == 0
        assert loaded_json.log_call_count == 0
        assert loaded_json.loss_history == []
        assert all(getattr(loaded_json, name) == [] for name in AUX_HISTORY_NAMES)

        print(
            {
                "script": "manual_storage",
                "n_frequencies": n_frequencies,
                "n_params": problem.n_params,
                "npz_roundtrip": True,
                "json_roundtrip": True,
                "periodic_checkpoint": True,
                "atomic_budget_rejection": True,
                "unbounded_logging": True,
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
