"""Tests for MADS / OrthoMADS direct-search algorithms (OMADS wrapper).

Covers:
 - algorithm_str / algorithm_type metadata
 - bounded mode enforcement
 - smoke run (eval_count > 0, best_loss not None, loss_history non-empty)
 - time_steps monotonicity
 - reproducibility with seed
 - best_params_bounded within problem bounds
"""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.core.algorithm import AlgorithmType
from dfbench.core.objective import Objective

from dfbench.algorithms.direct_search.omads_mads import OmadsMADS, OmadsOrthoMADS

DIRECT_SEARCH_ALGORITHMS = [OmadsMADS, OmadsOrthoMADS]


# ======================================================================
# Metadata
# ======================================================================


class TestDirectSearchMetadata:
    @pytest.mark.parametrize("cls", DIRECT_SEARCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_str(self, cls):
        """algorithm_str is a non-empty string."""
        algo = cls()
        assert isinstance(algo.algorithm_str, str) and len(algo.algorithm_str) > 0

    @pytest.mark.parametrize("cls", DIRECT_SEARCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_type(self, cls):
        """algorithm_type is EVOLUTIONARY (derivative-free category)."""
        algo = cls()
        assert algo.algorithm_type == AlgorithmType.EVOLUTIONARY


# ======================================================================
# Smoke tests — basic correctness
# ======================================================================


class TestDirectSearchSmoke:
    @pytest.mark.parametrize("cls", DIRECT_SEARCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_produces_evals(self, cls, mock_problem):
        """After optimize(), eval_count > 0."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0

    @pytest.mark.parametrize("cls", DIRECT_SEARCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_best_loss_not_none(self, cls, mock_problem):
        """After optimize(), best_loss is not None."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.best_loss is not None

    @pytest.mark.parametrize("cls", DIRECT_SEARCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_loss_history_non_empty(self, cls, mock_problem):
        """After optimize(), loss_history is non-empty."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert len(obj.loss_history) > 0

    @pytest.mark.parametrize("cls", DIRECT_SEARCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_time_steps_monotonic(self, cls, mock_problem):
        """time_steps monotonically non-decreasing."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        ts = obj.time_steps
        assert len(ts) > 0
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1]


# ======================================================================
# Bounded-mode enforcement
# ======================================================================


class TestDirectSearchBounded:
    @pytest.mark.parametrize("cls", DIRECT_SEARCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_bounded_mode(self, cls, mock_problem):
        """prepare() sets obj.unbounded = False."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.unbounded is False

    @pytest.mark.parametrize("cls", DIRECT_SEARCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_best_params_bounded(self, cls, mock_problem):
        """best_params_bounded is within problem bounds."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        bp = obj.best_params_bounded
        bounds = mock_problem.bounds
        assert np.all(np.array(bp) >= np.array(bounds[0]) - 1e-6)
        assert np.all(np.array(bp) <= np.array(bounds[1]) + 1e-6)


# ======================================================================
# Reproducibility
# ======================================================================


class TestDirectSearchReproducibility:
    @pytest.mark.parametrize("cls", DIRECT_SEARCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_reproducibility(self, cls, mock_problem):
        """Two runs with same seed produce identical loss_history."""
        algo1 = cls()
        obj1 = Objective(mock_problem, max_evals=20, max_time=60)
        algo1.optimize(obj1, random_seed=42)

        algo2 = cls()
        obj2 = Objective(mock_problem, max_evals=20, max_time=60)
        algo2.optimize(obj2, random_seed=42)

        np.testing.assert_allclose(
            [float(v) for v in obj1.loss_history],
            [float(v) for v in obj2.loss_history],
            atol=1e-5,
        )
