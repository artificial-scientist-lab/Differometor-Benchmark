"""Objective penalty-function switching tests.

Mirrors ``test_objective_space_mode.py``: ``set_penalty_fn`` must update
the problem's penalty function, re-trace the JIT-compiled objective, and
rebind the Objective's cached evaluation callables — all before
``start_logging()``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from dfbench.core.objective import Objective
from dfbench.problems.base_problem import (
    OpticalSetupProblem,
    relu_penalty,
    squashed_relu_penalty,
    zero_penalty,
)


# ── Helper: concrete OpticalSetupProblem with a real penalty path ─────


class _PenaltyStubProblem(OpticalSetupProblem):
    """Problem whose objective includes a power-penalty term.

    The penalty is driven by a fixed ``powers`` array so that switching
    ``_power_penalty_fn`` produces an observable change in the loss.
    """

    _supports_power_penalty = True

    def __init__(self) -> None:
        super().__init__(name="penalty_stub", n_frequencies=4)
        # Fixed "powers": three groups of one element each.
        # Values chosen above the respective thresholds so the penalty
        # is non-zero for the squashed/relu presets.
        self._powers = [
            jnp.array([[1.0e7]]),   # hard-side group (> HARD threshold)
            jnp.array([[1.0e4]]),   # soft-side group (> SOFT threshold)
            jnp.array([[1.0]]),     # detector group (> DETECTOR threshold)
        ]
        self._build_objective_function()

    def _build_objective_function(self) -> None:
        import jax

        powers = self._powers

        @jax.jit
        def objective_function(optimized_parameters):
            violations = self._compute_power_violations(powers)
            penalty = jnp.sum(violations)
            sensitivity_loss = jnp.sum(optimized_parameters**2)
            return sensitivity_loss + penalty

        @jax.jit
        def objective_function_aux(optimized_parameters):
            violations = self._compute_power_violations(powers)
            penalty = jnp.sum(violations)
            sensitivity_loss = jnp.sum(optimized_parameters**2)
            aux = self._build_aux(powers, sensitivity_loss, penalty, violations)
            return sensitivity_loss + penalty, aux

        self.objective_function = objective_function
        self.objective_function_aux = objective_function_aux

    @property
    def bounds(self):
        return jnp.array([[-1.0, -2.0], [1.0, 2.0]])

    @property
    def optimization_pairs(self):
        return [("comp", "param_a"), ("comp", "param_b")]

    def calculate_sensitivity(self, optimized_parameters):
        return jnp.ones(4)

    def to_spec(self) -> dict:
        spec = self._base_spec()
        spec["type"] = "_PenaltyStubProblem"
        return spec


# ======================================================================
# set_penalty_fn
# ======================================================================


@pytest.fixture()
def problem():
    return _PenaltyStubProblem()


class TestSetPenaltyFn:

    def test_default_penalty_fn_is_squashed_relu(self, problem):
        assert problem.power_penalty_fn is squashed_relu_penalty

    def test_objective_penalty_fn_passthrough(self, problem):
        obj = Objective(problem)
        assert obj.penalty_fn is squashed_relu_penalty

    def test_set_penalty_fn_takes_effect(self, problem):
        """Switching to zero_penalty removes the penalty term from the loss."""
        obj = Objective(problem)
        params = jnp.array([0.0, 0.0])  # zero base loss → loss == penalty

        default_loss = float(obj.value(params))
        assert default_loss > 0.0  # squashed_relu produces a penalty

        obj.set_penalty_fn(zero_penalty)
        zero_loss = float(obj.value(params))
        assert zero_loss == pytest.approx(0.0)

    def test_set_penalty_fn_rebinds_grad(self, problem):
        """Grad callables are rebound after set_penalty_fn.

        The penalty is constant w.r.t. params (powers are fixed), so the
        grad is ``2*params`` regardless of the penalty preset. We verify
        the rebind happened by checking the grad matches the zero-penalty
        analytical value after the switch.
        """
        obj = Objective(problem)
        params = jnp.array([0.5, -0.5])

        obj.set_penalty_fn(zero_penalty)
        grad_zero = np.array(obj.grad(params))

        np.testing.assert_allclose(grad_zero, np.array([1.0, -1.0]), atol=1e-6)

    def test_set_penalty_fn_property_reflects_change(self, problem):
        obj = Objective(problem)
        obj.set_penalty_fn(relu_penalty)
        assert obj.penalty_fn is relu_penalty
        assert obj.problem.power_penalty_fn is relu_penalty

    def test_set_penalty_fn_after_start_logging_raises(self, problem):
        obj = Objective(problem)
        obj.start_logging()

        with pytest.raises(RuntimeError, match="set_penalty_fn"):
            obj.set_penalty_fn(zero_penalty)

    def test_set_penalty_fn_composes_with_set_space_mode(self, problem):
        """Order independence: both rebinds apply before start_logging."""
        obj = Objective(problem, unbounded=False)
        params = jnp.array([0.0, 0.0])

        loss_bounded = float(obj.value(params))

        obj.set_penalty_fn(zero_penalty)
        obj.set_space_mode(True)
        loss_composed = float(obj.value(params))

        # In unbounded space the params are sigmoid-mapped away from the
        # bounds, and the penalty is now zero — value must differ.
        assert loss_composed != pytest.approx(loss_bounded)
        assert obj.unbounded is True
        assert obj.penalty_fn is zero_penalty

    def test_set_penalty_fn_relu_differs_from_squashed(self, problem):
        """relu_penalty and squashed_relu_penalty give different magnitudes."""
        obj = Objective(problem)
        params = jnp.array([0.0, 0.0])

        obj.set_penalty_fn(squashed_relu_penalty)
        squashed_loss = float(obj.value(params))

        obj.set_penalty_fn(relu_penalty)
        relu_loss = float(obj.value(params))

        # relu is unbounded; squashed saturates below 1 per element.
        assert relu_loss > squashed_loss


# ======================================================================
# Problems without penalty support
# ======================================================================


class TestPenaltyFnOnUnsupportedProblem:
    """The mock QuadraticProblem has no power-penalty path.

    ``Objective.penalty_fn`` returns ``None`` and ``set_penalty_fn``
    raises a clear ``RuntimeError`` rather than silently no-op'ing on a
    problem that does not fulfil the penalty contract.
    """

    def test_penalty_fn_is_none_for_plain_problem(self, mock_problem):
        obj = Objective(mock_problem)
        assert obj.penalty_fn is None

    def test_set_penalty_fn_raises_on_unsupported_problem(self, mock_problem):
        obj = Objective(mock_problem)
        with pytest.raises(RuntimeError, match="does not opt into"):
            obj.set_penalty_fn(zero_penalty)


# ======================================================================
# Opt-in flag guards (Layer 1)
# ======================================================================


class TestPenaltyOptInFlag:
    """``_supports_power_penalty`` is the opt-in marker for the penalty contract.

    Problems that inherit ``OpticalSetupProblem.set_penalty_fn`` but have no
    power-constraint path (VoyagerProblem, VoyagerTuningProblem) must still
    reject the call rather than silently rebuilding. The flag is the single
    source of truth both ``OpticalSetupProblem.set_penalty_fn`` and
    ``Objective.set_penalty_fn`` check.
    """

    def test_voyager_problem_rejects_set_penalty_fn(self):
        from dfbench.problems import VoyagerProblem

        problem = VoyagerProblem(n_frequencies=8)
        assert problem._supports_power_penalty is False
        with pytest.raises(RuntimeError, match="no power-constraint path"):
            problem.set_penalty_fn(zero_penalty)

    def test_voyager_problem_via_objective_rejects_set_penalty_fn(self):
        from dfbench.problems import VoyagerProblem

        obj = Objective(VoyagerProblem(n_frequencies=8))
        with pytest.raises(RuntimeError, match="does not opt into"):
            obj.set_penalty_fn(zero_penalty)

    def test_constrained_voyager_supports_penalty(self):
        from dfbench.problems import ConstrainedVoyagerProblem

        problem = ConstrainedVoyagerProblem(n_frequencies=8)
        assert problem._supports_power_penalty is True
        problem.set_penalty_fn(zero_penalty)  # no raise

    def test_uifo_supports_penalty(self):
        from dfbench.problems import UIFOProblem

        problem = UIFOProblem(size=2, n_frequencies=8, topology_seed=0)
        assert problem._supports_power_penalty is True
        problem.set_penalty_fn(zero_penalty)  # no raise


# ======================================================================
# power_thresholds property
# ======================================================================


class TestPowerThresholds:
    def test_stub_problem_thresholds(self, problem):
        t = problem.power_thresholds
        assert set(t.keys()) == {"hard", "soft", "detector"}
        assert t["hard"] > t["soft"] > t["detector"]

    def test_objective_thresholds_passthrough(self, problem):
        obj = Objective(problem)
        assert obj.power_thresholds == problem.power_thresholds

    def test_thresholds_none_for_unsupported_problem(self, mock_problem):
        obj = Objective(mock_problem)
        assert obj.power_thresholds is None

    def test_thresholds_none_for_voyager_problem(self):
        from dfbench.problems import VoyagerProblem

        obj = Objective(VoyagerProblem(n_frequencies=8))
        assert obj.power_thresholds is None


# ======================================================================
# objective_function_aux (Layer 0)
# ======================================================================


class TestObjectiveFunctionAux:
    """The aux objective returns ``(loss, aux)`` with a pytree aux dict.

    ``aux`` carries the loss decomposition, the physical feasibility flag,
    the per-constraint violations, and the raw per-group powers. These tests
    pin the schema and the feasibility semantics (physical, not penalty-based)
    without depending on a full optical simulation.
    """

    def test_aux_schema(self, problem):
        loss, aux = problem.objective_function_aux(jnp.array([0.0, 0.0]))
        assert set(aux.keys()) == {
            "sensitivity_loss",
            "penalty",
            "is_feasible",
            "violations",
            "power_values",
        }
        assert set(aux["power_values"].keys()) == {"hard", "soft", "detector"}
        assert loss == pytest.approx(
            float(aux["sensitivity_loss"] + aux["penalty"])
        )

    def test_is_feasible_false_when_power_exceeds_threshold(self, problem):
        # The stub's fixed powers all exceed their thresholds.
        _, aux = problem.objective_function_aux(jnp.array([0.0, 0.0]))
        assert bool(aux["is_feasible"]) is False

    def test_is_feasible_true_when_power_under_threshold(self):
        # Build a stub with all powers below their thresholds.
        problem = _PenaltyStubProblem()
        problem._powers = [
            jnp.array([[1.0]]),    # well below HARD (3.5e6)
            jnp.array([[1.0]]),    # well below SOFT (2e3)
            jnp.array([[1e-3]]),   # well below DETECTOR (1e-2)
        ]
        problem._build_objective_function()
        _, aux = problem.objective_function_aux(jnp.array([0.0, 0.0]))
        assert bool(aux["is_feasible"]) is True

    def test_aux_loss_matches_plain_objective(self, problem):
        params = jnp.array([0.3, -0.2])
        plain = float(problem.objective_function(params))
        loss, _ = problem.objective_function_aux(params)
        assert float(loss) == pytest.approx(plain, rel=1e-6)

    def test_aux_survives_set_penalty_fn_rebuild(self, problem):
        problem.set_penalty_fn(relu_penalty)
        loss, aux = problem.objective_function_aux(jnp.array([0.0, 0.0]))
        assert float(aux["penalty"]) > 0.0
        assert float(loss) == pytest.approx(
            float(aux["sensitivity_loss"] + aux["penalty"])
        )

# ======================================================================
# Objective-level aux methods (Layer 2)
# ======================================================================


class TestObjectiveValueAux:
    def test_value_aux_returns_loss_and_aux(self, problem):
        obj = Objective(problem)
        params = jnp.array([0.0, 0.0])
        loss, aux = obj.value_aux(params)
        assert float(loss) == pytest.approx(
            float(aux["sensitivity_loss"] + aux["penalty"])
        )
        assert set(aux.keys()) == {
            "sensitivity_loss",
            "penalty",
            "is_feasible",
            "violations",
            "power_values",
        }

    def test_value_aux_logs_loss(self, problem):
        obj = Objective(problem)
        obj.start_logging()
        loss, _ = obj.value_aux(jnp.array([0.0, 0.0]))
        assert obj.eval_count == 1
        assert obj.current_loss is not None

    def test_value_aux_raises_on_unsupported_problem(self, mock_problem):
        obj = Objective(mock_problem)
        with pytest.raises(RuntimeError, match="does not expose"):
            obj.value_aux(jnp.array([0.0, 0.0]))

    def test_value_and_grad_aux(self, problem):
        obj = Objective(problem)
        params = jnp.array([0.5, -0.5])
        value, grad, aux = obj.value_and_grad_aux(params)
        # Penalty is constant w.r.t. params (fixed powers), so grad == 2*params.
        np.testing.assert_allclose(np.array(grad), np.array([1.0, -1.0]), atol=1e-6)
        assert float(value) == pytest.approx(
            float(aux["sensitivity_loss"] + aux["penalty"])
        )

    def test_value_and_grad_aux_raises_on_unsupported(self, mock_problem):
        obj = Objective(mock_problem)
        with pytest.raises(RuntimeError, match="does not expose"):
            obj.value_and_grad_aux(jnp.array([0.0, 0.0]))

    def test_vmap_value_aux_batched_aux_pytree(self, problem):
        obj = Objective(problem)
        params = jnp.array([[0.0, 0.0], [0.3, -0.2], [1.0, 1.0]])
        losses, aux = obj.vmap_value_aux(params)
        assert losses.shape == (3,)
        # Every leaf gains a leading batch dim.
        assert aux["is_feasible"].shape == (3,)
        assert aux["sensitivity_loss"].shape == (3,)
        assert aux["penalty"].shape == (3,)
        assert aux["violations"].shape[0] == 3
        assert aux["power_values"]["hard"].shape[0] == 3
        assert aux["power_values"]["soft"].shape[0] == 3
        assert aux["power_values"]["detector"].shape[0] == 3

    def test_vmap_value_aux_raises_on_unsupported(self, mock_problem):
        obj = Objective(mock_problem)
        params = jnp.zeros((2, mock_problem._n_params))
        with pytest.raises(RuntimeError, match="does not expose"):
            obj.vmap_value_aux(params)

    def test_vmap_value_and_grad_aux(self, problem):
        obj = Objective(problem)
        params = jnp.array([[0.5, -0.5], [0.0, 0.0]])
        values, grads, aux = obj.vmap_value_and_grad_aux(params)
        assert values.shape == (2,)
        assert grads.shape == (2, 2)
        assert aux["is_feasible"].shape == (2,)

    def test_warmup_value_aux_skips_unsupported(self, mock_problem):
        obj = Objective(mock_problem)
        # Should be a no-op, not raise.
        obj.warmup_value_aux()
        obj.warmup_value_and_grad_aux()
        obj.warmup_vmap_value_aux()
        obj.warmup_vmap_value_and_grad_aux()

    def test_warmup_value_aux_runs_on_supported(self, problem):
        obj = Objective(problem)
        obj.warmup_value_aux()
        obj.warmup_value_and_grad_aux()
        obj.warmup_vmap_value_aux(batch_size=2)
        obj.warmup_vmap_value_and_grad_aux(batch_size=2)


# ======================================================================
# Save tokens and aux histories (Layer 3)
# ======================================================================


class TestSaveConfigAuxTokens:
    def test_aux_alias_expands_to_all_five(self):
        from dfbench.core.storage import SaveConfig

        cfg = SaveConfig.from_flags(save=["aux"])
        assert cfg.sensitivity_loss
        assert cfg.penalty
        assert cfg.is_feasible
        assert cfg.power_values
        assert cfg.violations
        assert not cfg.batched_sensitivity_loss

    def test_batched_aux_alias_expands_to_all_five_batched(self):
        from dfbench.core.storage import SaveConfig

        cfg = SaveConfig.from_flags(save=["batched_aux"])
        assert cfg.batched_sensitivity_loss
        assert cfg.batched_penalty
        assert cfg.batched_is_feasible
        assert cfg.batched_power_values
        assert cfg.batched_violations
        assert not cfg.sensitivity_loss  # alias only flips batched variants

    def test_per_field_batched_token(self):
        from dfbench.core.storage import SaveConfig

        cfg = SaveConfig.from_flags(save=["batched_is_feasible", "penalty"])
        assert cfg.batched_is_feasible
        assert cfg.penalty
        assert not cfg.is_feasible
        assert not cfg.batched_penalty

    def test_unknown_token_raises(self):
        from dfbench.core.storage import SaveConfig

        with pytest.raises(ValueError, match="Unknown save token"):
            SaveConfig.from_flags(save=["nonsense"])

    def test_save_config_roundtrip(self):
        from dfbench.core.storage import SaveConfig

        cfg = SaveConfig.from_flags(save=["aux", "batched_is_feasible"])
        d = cfg.to_dict()
        cfg2 = SaveConfig.from_dict(d)
        assert cfg.mismatch(cfg2) == []


class TestAuxHistories:
    def test_histories_empty_by_default(self, problem):
        obj = Objective(problem)
        obj.start_logging()
        obj.value_aux(jnp.array([0.0, 0.0]))
        # No aux save tokens enabled -> aux histories stay empty.
        assert obj.sensitivity_loss_history == []
        assert obj.is_feasible_history == []
        assert obj.violations_history == []
        assert obj.power_hard_history == []

    def test_is_feasible_token_records_history(self, problem):
        obj = Objective(problem, save=["is_feasible"])
        obj.start_logging()
        obj.value_aux(jnp.array([0.0, 0.0]))
        assert len(obj.is_feasible_history) == 1
        assert obj.is_feasible_history[0] is not None
        # Other aux histories stay empty.
        assert obj.sensitivity_loss_history == []
        assert obj.violations_history == []

    def test_aux_alias_records_all_histories(self, problem):
        obj = Objective(problem, save=["aux"])
        obj.start_logging()
        obj.value_aux(jnp.array([0.0, 0.0]))
        assert len(obj.sensitivity_loss_history) == 1
        assert len(obj.penalty_history) == 1
        assert len(obj.is_feasible_history) == 1
        assert len(obj.violations_history) == 1
        assert len(obj.power_hard_history) == 1
        assert len(obj.power_soft_history) == 1
        assert len(obj.power_detector_history) == 1

    def test_batched_is_feasible_stores_full_batch(self, problem):
        obj = Objective(problem, save=["batched_is_feasible"])
        obj.start_logging()
        params = jnp.array([[0.0, 0.0], [0.3, -0.2], [1.0, 1.0]])
        obj.vmap_value_aux(params)
        assert len(obj.is_feasible_history) == 1
        entry = obj.is_feasible_history[0]
        assert entry.shape == (3,)

    def test_reduced_is_feasible_picks_best_loss_point(self, problem):
        obj = Objective(problem, save=["is_feasible"])
        obj.start_logging()
        # Three points; the one with lowest loss is [0,0] (penalty constant,
        # base loss = sum(params**2) = 0). Its feasibility is False (powers
        # exceed thresholds), so the reduced entry must be False.
        params = jnp.array([[1.0, 1.0], [0.0, 0.0], [0.5, -0.5]])
        obj.vmap_value_aux(params)
        assert len(obj.is_feasible_history) == 1
        assert bool(obj.is_feasible_history[0]) is False

    def test_non_aux_eval_does_not_touch_aux_histories(self, problem):
        obj = Objective(problem, save=["aux"])
        obj.start_logging()
        obj.value(jnp.array([0.0, 0.0]))
        assert obj.sensitivity_loss_history == []
        assert obj.is_feasible_history == []


# ======================================================================
# best_is_feasible property (Layer 2)
# ======================================================================


class TestBestIsFeasible:
    def test_none_when_token_disabled(self, problem):
        obj = Objective(problem)  # no aux tokens
        obj.start_logging()
        obj.value_aux(jnp.array([0.0, 0.0]))
        assert obj.best_is_feasible is None

    def test_none_when_no_evals(self, problem):
        obj = Objective(problem, save=["is_feasible"])
        assert obj.best_is_feasible is None

    def test_false_when_best_point_infeasible(self, problem):
        obj = Objective(problem, save=["is_feasible"])
        obj.start_logging()
        # The stub's fixed powers all exceed thresholds, so every point is
        # infeasible; best loss is at [0,0].
        obj.value_aux(jnp.array([0.0, 0.0]))
        assert obj.best_is_feasible is False

    def test_true_when_best_point_feasible(self):
        problem = _PenaltyStubProblem()
        problem._powers = [
            jnp.array([[1.0]]),
            jnp.array([[1.0]]),
            jnp.array([[1e-3]]),
        ]
        problem._build_objective_function()
        obj = Objective(problem, save=["is_feasible"])
        obj.start_logging()
        obj.value_aux(jnp.array([0.0, 0.0]))
        assert obj.best_is_feasible is True

    def test_batched_best_uses_batched_is_feasible(self, problem):
        obj = Objective(problem, save=["batched_is_feasible"])
        obj.start_logging()
        params = jnp.array([[1.0, 1.0], [0.0, 0.0], [0.5, -0.5]])
        obj.vmap_value_aux(params)
        # Best loss is at index 1 ([0,0]); all points infeasible here.
        assert obj.best_batch_index == 1
        assert obj.best_is_feasible is False


# ======================================================================
# Checkpoint roundtrip with aux histories (Layer 3)
# ======================================================================


class TestAuxCheckpointRoundtrip:
    def test_npz_roundtrip_preserves_aux_histories(self, problem, tmp_path):
        obj = Objective(
            problem, save=["aux"], checkpoint_dir=str(tmp_path)
        )
        obj.start_logging()
        obj.value_aux(jnp.array([0.0, 0.0]))
        obj.value_aux(jnp.array([0.3, -0.2]))
        path = obj.save_run_data(algorithm_name="aux_test")

        obj2 = Objective(problem, save=["aux"], checkpoint_dir=str(tmp_path))
        obj2.load_run_data(path)
        assert len(obj2.sensitivity_loss_history) == 2
        assert len(obj2.is_feasible_history) == 2
        assert len(obj2.power_hard_history) == 2
        assert obj2.best_is_feasible is False
        assert obj2.best_eval_index == obj.best_eval_index

    def test_json_roundtrip_preserves_aux_histories(self, problem, tmp_path):
        obj = Objective(
            problem,
            save=["aux"],
            checkpoint_format="json",
            checkpoint_dir=str(tmp_path),
        )
        obj.start_logging()
        obj.value_aux(jnp.array([0.0, 0.0]))
        path = obj.save_run_data(algorithm_name="aux_json")

        obj2 = Objective(
            problem,
            save=["aux"],
            checkpoint_format="json",
            checkpoint_dir=str(tmp_path),
        )
        obj2.load_run_data(path)
        assert len(obj2.is_feasible_history) == 1
        assert len(obj2.power_detector_history) == 1
        assert obj2.best_eval_index == obj.best_eval_index

    def test_legacy_checkpoint_loads_with_empty_aux(self, problem, tmp_path):
        # Save without aux tokens, then load into an Objective that has aux
        # tokens enabled. Aux histories should be empty after load.
        obj = Objective(problem, checkpoint_dir=str(tmp_path))
        obj.start_logging()
        obj.value(jnp.array([0.0, 0.0]))
        path = obj.save_run_data(algorithm_name="no_aux")

        obj2 = Objective(problem, save=["aux"], checkpoint_dir=str(tmp_path))
        obj2.load_run_data(path)
        assert obj2.sensitivity_loss_history == []
        assert obj2.is_feasible_history == []
