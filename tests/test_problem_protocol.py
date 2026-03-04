"""Section 3 — Problem Protocol (dfbench.core.problem.ContinuousProblem).

Tests 3.1–3.3: abstract base class behaviour.
"""

import pytest

from dfbench.core.problem import ContinuousProblem


class TestContinuousProblemABC:
    """3.1–3.3"""

    def test_cannot_instantiate_directly(self):
        """3.1 ContinuousProblem is abstract — cannot instantiate."""
        with pytest.raises(TypeError):
            ContinuousProblem()

    def test_minimal_subclass_requires_methods(self):
        """3.2 Concrete subclass must implement bounds, optimization_pairs,
        objective_function, and sigmoid_objective_function.
        """

        class Incomplete(ContinuousProblem):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_n_params_equals_len_optimization_pairs(self, mock_problem):
        """3.3 n_params == len(optimization_pairs)."""
        assert mock_problem.n_params == len(mock_problem.optimization_pairs)
