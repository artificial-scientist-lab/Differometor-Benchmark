"""Algorithm-specific tests for the native-JAX custom/hybrid algorithm batch.

The shared protocol checks (smoke, algorithm_str, bounds, etc.) are covered
by the REGISTRY-driven tests in ``test_algorithms_uniform.py``. Only knobs
that the uniform suite cannot express belong here.
"""

from __future__ import annotations

import pytest

from dfbench.algorithms.gradient_based.custom_jax import ARCJAX
from dfbench.core.objective import Objective


class TestARCJAX:
    def test_arc_not_implemented(self, mock_problem):
        """ARCJAX is intentionally disabled and must raise NotImplementedError."""
        algo = ARCJAX()
        obj = Objective(mock_problem, max_evals=10, max_time=60)
        with pytest.raises(NotImplementedError):
            algo.optimize(obj, random_seed=42)
