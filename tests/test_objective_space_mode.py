"""Objective space-mode switching tests."""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from dfbench.core.objective import Objective


class TestObjectiveSpaceMode:
    def test_constructor_custom_mapping_requires_inverse(self, mock_problem):
        with pytest.raises(ValueError, match="requires both"):
            Objective(
                mock_problem,
                unbounded=True,
                unit_mapping=lambda x: x,
            )

    def test_set_space_mode_rebinds_evaluation_functions(self, mock_problem):
        params = jnp.array([1.0, -1.0])

        bounded_obj = Objective(mock_problem, unbounded=False)
        bounded_obj.start_logging()
        bounded_loss = float(bounded_obj.value(params))

        obj = Objective(mock_problem, unbounded=False)
        obj.set_space_mode(True)
        obj.start_logging()
        unbounded_loss = float(obj.value(params))

        assert obj.unbounded is True
        assert bounded_loss != pytest.approx(unbounded_loss)

    def test_set_space_mode_after_start_logging_raises(self, mock_problem):
        obj = Objective(mock_problem, unbounded=False)
        obj.start_logging()

        with pytest.raises(RuntimeError, match="set_space_mode"):
            obj.set_space_mode(True)

    def test_set_space_mode_custom_mapping_requires_inverse(self, mock_problem):
        obj = Objective(mock_problem, unbounded=False)

        with pytest.raises(ValueError, match="requires both"):
            obj.set_space_mode(
                True,
                unit_mapping=lambda x: x,
            )

    def test_set_space_mode_custom_mapping_rebinds_objective(self, mock_problem):
        params = jnp.array([0.0, 0.0])

        default_obj = Objective(mock_problem, unbounded=True)
        default_obj.start_logging()
        default_loss = float(default_obj.value(params))

        def forward(x):
            # deterministic alternative [0,1] mapping for test visibility
            return jnp.full_like(x, 0.9)

        def inverse(x):
            return x

        obj = Objective(mock_problem, unbounded=True)
        obj.set_space_mode(
            True,
            unit_mapping=forward,
            inverse_unit_mapping=inverse,
        )
        obj.start_logging()
        mapped_loss = float(obj.value(params))

        assert mapped_loss != pytest.approx(default_loss)

    def test_constructor_custom_mapping_is_used_for_bounded_view(self, mock_problem):
        def forward(x):
            return jax.nn.sigmoid(x)

        def inverse(x):
            x = jnp.clip(x, 1e-7, 1.0 - 1e-7)
            return jnp.log(x / (1.0 - x))

        obj = Objective(
            mock_problem,
            unbounded=True,
            unit_mapping=forward,
            inverse_unit_mapping=inverse,
        )
        obj.start_logging()
        params = jnp.array([0.4, -0.3])
        _ = obj.value(params)

        # Custom mapping maps to [0,1]; Objective scales to bounds
        lower, upper = mock_problem.bounds
        expected = np.array(lower + (upper - lower) * forward(params))
        actual = np.array(obj.best_params_bounded)
        np.testing.assert_allclose(actual, expected, atol=1e-6)
