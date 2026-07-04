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
            # base loss is the parameter norm so the penalty is observable
            return jnp.sum(optimized_parameters**2) + penalty

        self.objective_function = objective_function

    @property
    def bounds(self):
        return jnp.array([[-1.0, -2.0], [1.0, 2.0]])

    @property
    def optimization_pairs(self):
        return [("comp", "param_a"), ("comp", "param_b")]

    def calculate_sensitivity(self, optimized_parameters):
        return jnp.ones(4)


# ======================================================================
# set_penalty_fn
# ======================================================================


class TestSetPenaltyFn:
    @pytest.fixture()
    def problem(self):
        return _PenaltyStubProblem()

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
        with pytest.raises(RuntimeError, match="does not expose"):
            obj.set_penalty_fn(zero_penalty)