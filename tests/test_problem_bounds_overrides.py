"""Section 4 (partial) — Problem bounds overrides and penalty functions.

Tests 4.10–4.11, 4.15, 4.21–4.26 — these do NOT require Differometor simulation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from dfbench.problems.base_problem import (
    OpticalSetupProblem,
    relu_penalty,
    squashed_relu_penalty,
    zero_penalty,
)


# ── Helper: minimal concrete subclass for testing ─────────────────────


class _StubProblem(OpticalSetupProblem):
    """Minimal concrete subclass for unit-testing base-class helpers."""

    def __init__(self):
        super().__init__(name="stub", n_frequencies=10)

    @property
    def bounds(self):
        return jnp.array([[-1.0, -2.0], [1.0, 2.0]])

    @property
    def optimization_pairs(self):
        return [("comp", "param_a"), ("comp", "param_b")]

    def objective_function(self, params):
        return jnp.sum(params**2)

    def calculate_sensitivity(self, optimized_parameters):
        return jnp.ones(10)


# ======================================================================
# _apply_property_bounds_overrides (4.21–4.23)
# ======================================================================


class TestApplyPropertyBoundsOverrides:
    @pytest.fixture()
    def problem(self):
        return _StubProblem()

    def test_narrowing_accepted(self, problem):
        """4.21 Narrowing overrides are accepted."""
        default = {"param_a": [-1.0, 1.0], "param_b": [-2.0, 2.0]}
        result = problem._apply_property_bounds_overrides(
            default, bounds_overrides={"param_a": (-0.5, 0.5)}
        )
        assert result["param_a"] == [-0.5, 0.5]
        assert result["param_b"] == [-2.0, 2.0]  # unchanged

    def test_widening_rejected(self, problem):
        """4.21b Widening raises ValueError (unless allow_widen=True)."""
        default = {"param_a": [-1.0, 1.0]}
        with pytest.raises(ValueError, match="narrow"):
            problem._apply_property_bounds_overrides(
                default, bounds_overrides={"param_a": (-5.0, 5.0)}
            )

    def test_widening_with_allow(self, problem):
        """4.21c Widening is accepted when allow_widen=True."""
        default = {"param_a": [-1.0, 1.0]}
        result = problem._apply_property_bounds_overrides(
            default, bounds_overrides={"param_a": (-5.0, 5.0)}, allow_widen=True
        )
        assert result["param_a"] == [-5.0, 5.0]

    def test_lower_ge_upper_error(self, problem):
        """4.22 lower >= upper raises ValueError."""
        default = {"param_a": [-1.0, 1.0]}
        with pytest.raises(ValueError, match="must be <"):
            problem._apply_property_bounds_overrides(
                default, bounds_overrides={"param_a": (0.5, 0.5)}
            )

    def test_unknown_property_error(self, problem):
        """4.23 Unknown property raises ValueError."""
        default = {"param_a": [-1.0, 1.0]}
        with pytest.raises(ValueError, match="Unknown"):
            problem._apply_property_bounds_overrides(
                default, bounds_overrides={"nonexistent": (-0.5, 0.5)}
            )


# ======================================================================
# Penalty functions (4.24–4.26)
# ======================================================================


class TestSquashedReluPenalty:
    def test_below_threshold(self):
        """4.24a Returns 0 when value < threshold."""
        result = squashed_relu_penalty(jnp.array(0.5), 1.0)
        assert float(result) == pytest.approx(0.0)

    def test_above_threshold(self):
        """4.24b Returns > 0 when value > threshold."""
        result = squashed_relu_penalty(jnp.array(2.0), 1.0)
        assert float(result) > 0

    def test_bounded_below_one(self):
        """4.24c Result is bounded in [0, 1)."""
        result = squashed_relu_penalty(jnp.array(1000.0), 1.0)
        assert 0.0 <= float(result) < 1.0


class TestReluPenalty:
    def test_below_threshold(self):
        """4.25a Returns 0 when value < threshold."""
        result = relu_penalty(jnp.array(0.5), 1.0)
        assert float(result) == pytest.approx(0.0)

    def test_above_threshold(self):
        """4.25b Positive linear above threshold."""
        result = relu_penalty(jnp.array(2.0), 1.0)
        assert float(result) == pytest.approx(1.0)  # 2/1 - 1 = 1

    def test_linear(self):
        """4.25c Linearly increasing."""
        r1 = relu_penalty(jnp.array(3.0), 1.0)
        r2 = relu_penalty(jnp.array(5.0), 1.0)
        assert float(r2) > float(r1)


class TestZeroPenalty:
    def test_always_zero(self):
        """4.26 Always returns zero."""
        result = zero_penalty(jnp.array(100.0), 1.0)
        assert float(result) == 0.0

    def test_array_input(self):
        """4.26b Works with arrays."""
        result = zero_penalty(jnp.array([1.0, 2.0, 3.0]), 1.0)
        np.testing.assert_array_equal(np.array(result), np.zeros(3))
