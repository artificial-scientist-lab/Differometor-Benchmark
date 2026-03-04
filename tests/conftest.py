"""Shared fixtures for dfbench tests.

The mock problem provides a simple quadratic objective that doesn't require
Differometor or GPU simulation, making it suitable for fast CI tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dfbench.core.problem import ContinuousProblem
from dfbench.core.objective import Objective


# ---------------------------------------------------------------------------
# Mock problem: simple quadratic  f(x) = sum(x^2)
# ---------------------------------------------------------------------------


class QuadraticProblem(ContinuousProblem):
    """2-parameter quadratic problem for unit tests.

    objective_function(x) = sum(x^2); minimum at origin.
    sigmoid_objective_function applies sigmoid_bounding internally.
    """

    def __init__(self, n_params: int = 2) -> None:
        super().__init__()
        self.name = "quadratic_mock"
        self._n_params = n_params
        lo = -5.0 * jnp.ones(n_params)
        hi = 5.0 * jnp.ones(n_params)
        self._bounds = jnp.stack([lo, hi])

        # Plain bounded objective: expects params in [-5, 5]
        self.objective_function = lambda x: jnp.sum(x**2)

        # Unbounded objective: inverse-sigmoid first, then evaluate
        from differometor.utils import sigmoid_bounding

        def _sigmoid_obj(x):
            bounded = sigmoid_bounding(x, self._bounds)
            return jnp.sum(bounded**2)

        self.sigmoid_objective_function = _sigmoid_obj

    @property
    def bounds(self):
        return self._bounds

    @property
    def optimization_pairs(self):
        return [(f"comp_{i}", "param") for i in range(self._n_params)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_problem():
    """2-param quadratic mock problem."""
    return QuadraticProblem(n_params=2)


@pytest.fixture
def mock_problem_5d():
    """5-param quadratic mock problem (for batch / diversity tests)."""
    return QuadraticProblem(n_params=5)


@pytest.fixture
def seeded_obj(mock_problem):
    """Objective wrapping a mock problem, seeded and ready for logging."""
    obj = Objective(
        mock_problem,
        max_evals=200,
        max_time=60.0,
        save_grad_history=True,
        save_params_history=True,
        save_time_steps=True,
    )
    obj.set_seed(42)
    return obj


@pytest.fixture
def seeded_obj_unbounded(mock_problem):
    """Unbounded-mode Objective wrapping a mock problem."""
    obj = Objective(
        mock_problem,
        unbounded=True,
        max_evals=200,
        max_time=60.0,
        save_grad_history=True,
        save_params_history=True,
        save_time_steps=True,
    )
    obj.set_seed(42)
    return obj
