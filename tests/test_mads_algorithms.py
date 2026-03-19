"""Tests for MADS and OrthoMADS direct-search algorithms.

Covers:
  - Smoke tests: MADS and OrthoMADS run without error on a mock problem.
  - Shared parametrised tests: eval_count, best_loss, bounded params, timing.
  - Algorithm-specific tests: algorithm_str, algorithm_type, bounded mode.
  - Import and API surface tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.core.algorithm import AlgorithmType
from dfbench.core.objective import Objective

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from dfbench.algorithms.direct_search.mads import MADS
from dfbench.algorithms.direct_search.orthomads import OrthoMADS

MADS_ALGORITHMS = [MADS, OrthoMADS]


# ---------------------------------------------------------------------------
# Shared parametrised tests
# ---------------------------------------------------------------------------


class TestCommonChecks:
    """Shared checks that apply to both MADS and OrthoMADS."""

    @pytest.mark.parametrize("cls", MADS_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_str_nonempty(self, cls):
        """algorithm_str is a non-empty string."""
        algo = cls()
        assert isinstance(algo.algorithm_str, str) and len(algo.algorithm_str) > 0

    @pytest.mark.parametrize("cls", MADS_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_type(self, cls):
        """algorithm_type is DIRECT_SEARCH."""
        algo = cls()
        assert algo.algorithm_type == AlgorithmType.DIRECT_SEARCH

    @pytest.mark.parametrize("cls", MADS_ALGORITHMS, ids=lambda c: c.__name__)
    def test_optimize_produces_evals(self, cls, mock_problem):
        """After optimize(), eval_count > 0."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0

    @pytest.mark.parametrize("cls", MADS_ALGORITHMS, ids=lambda c: c.__name__)
    def test_best_loss_not_none(self, cls, mock_problem):
        """After optimize(), best_loss is not None."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.best_loss is not None

    @pytest.mark.parametrize("cls", MADS_ALGORITHMS, ids=lambda c: c.__name__)
    def test_loss_history_nonempty(self, cls, mock_problem):
        """After optimize(), loss_history is non-empty."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert len(obj.loss_history) > 0

    @pytest.mark.parametrize("cls", MADS_ALGORITHMS, ids=lambda c: c.__name__)
    def test_time_steps_monotonic(self, cls, mock_problem):
        """time_steps is monotonically non-decreasing."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        ts = obj.time_steps
        assert len(ts) > 0
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1]

    @pytest.mark.parametrize("cls", MADS_ALGORITHMS, ids=lambda c: c.__name__)
    def test_bounded_mode(self, cls, mock_problem):
        """prepare() sets obj.unbounded = False (bounded physical space)."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.unbounded is False

    @pytest.mark.parametrize("cls", MADS_ALGORITHMS, ids=lambda c: c.__name__)
    def test_best_params_within_bounds(self, cls, mock_problem):
        """best_params_bounded is within the problem bounds."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        bp = np.array(obj.best_params_bounded)
        lo = np.array(mock_problem.bounds[0])
        hi = np.array(mock_problem.bounds[1])
        assert np.all(bp >= lo - 1e-6)
        assert np.all(bp <= hi + 1e-6)

    @pytest.mark.parametrize("cls", MADS_ALGORITHMS, ids=lambda c: c.__name__)
    def test_eval_budget_respected(self, cls, mock_problem):
        """eval_count does not significantly exceed max_evals."""
        max_evals = 20
        algo = cls()
        obj = Objective(mock_problem, max_evals=max_evals, max_time=60)
        algo.optimize(obj, random_seed=42)
        # OMADS may overshoot by up to one poll set (2*n_params candidates),
        # so we allow a small tolerance.
        assert obj.eval_count <= max_evals + 2 * mock_problem.n_params

    @pytest.mark.parametrize("cls", MADS_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_str_on_objective(self, cls, mock_problem):
        """obj.algorithm_str is set to the algorithm's identifier."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=10, max_time=60)
        algo.optimize(obj, random_seed=0)
        assert obj.algorithm_str == algo.algorithm_str


# ---------------------------------------------------------------------------
# Algorithm-specific tests
# ---------------------------------------------------------------------------


class TestMADS:
    """Tests specific to the MADS algorithm."""

    def test_algorithm_str_is_mads(self):
        """MADS.algorithm_str == 'mads'."""
        assert MADS.algorithm_str == "mads"

    def test_default_hyperparams(self):
        """Default poll_size_init and min_poll_size are conservative."""
        algo = MADS()
        assert algo.poll_size_init == 1.0
        assert algo.min_poll_size == 1e-9

    def test_hyperparams_override_via_init(self):
        """Constructor hyperparameters are stored on the instance."""
        algo = MADS(poll_size_init=0.5, min_poll_size=1e-6)
        assert algo.poll_size_init == 0.5
        assert algo.min_poll_size == 1e-6

    def test_optimize_with_small_poll_size(self, mock_problem):
        """MADS runs and logs evals with a smaller initial poll size."""
        algo = MADS(poll_size_init=0.1)
        obj = Objective(mock_problem, max_evals=20, max_time=60)
        algo.optimize(obj, random_seed=7)
        assert obj.eval_count > 0

    def test_optimize_poll_size_override(self, mock_problem):
        """poll_size_init passed to optimize() overrides __init__ value."""
        algo = MADS(poll_size_init=2.0)
        obj = Objective(mock_problem, max_evals=20, max_time=60)
        # Override to a more conservative value
        algo.optimize(obj, random_seed=7, poll_size_init=0.5)
        assert obj.eval_count > 0


class TestOrthoMADS:
    """Tests specific to the OrthoMADS algorithm."""

    def test_algorithm_str_is_orthomads(self):
        """OrthoMADS.algorithm_str == 'orthomads'."""
        assert OrthoMADS.algorithm_str == "orthomads"

    def test_default_hyperparams(self):
        """Default poll_size_init and min_poll_size are conservative."""
        algo = OrthoMADS()
        assert algo.poll_size_init == 1.0
        assert algo.min_poll_size == 1e-9

    def test_hyperparams_override_via_init(self):
        """Constructor hyperparameters are stored on the instance."""
        algo = OrthoMADS(poll_size_init=0.5, min_poll_size=1e-6)
        assert algo.poll_size_init == 0.5
        assert algo.min_poll_size == 1e-6

    def test_optimize_with_small_poll_size(self, mock_problem):
        """OrthoMADS runs and logs evals with a smaller initial poll size."""
        algo = OrthoMADS(poll_size_init=0.1)
        obj = Objective(mock_problem, max_evals=20, max_time=60)
        algo.optimize(obj, random_seed=7)
        assert obj.eval_count > 0

    def test_optimize_poll_size_override(self, mock_problem):
        """poll_size_init passed to optimize() overrides __init__ value."""
        algo = OrthoMADS(poll_size_init=2.0)
        obj = Objective(mock_problem, max_evals=20, max_time=60)
        # Override to a more conservative value
        algo.optimize(obj, random_seed=7, poll_size_init=0.5)
        assert obj.eval_count > 0


# ---------------------------------------------------------------------------
# API surface / import tests
# ---------------------------------------------------------------------------


class TestImports:
    """Verify that MADS and OrthoMADS are importable from the top-level package."""

    def test_import_from_algorithms_package(self):
        """MADS and OrthoMADS are importable from dfbench.algorithms."""
        from dfbench.algorithms import MADS, OrthoMADS  # noqa: F401

    def test_import_from_direct_search(self):
        """MADS and OrthoMADS are importable from direct_search sub-package."""
        from dfbench.algorithms.direct_search import MADS, OrthoMADS  # noqa: F401

    def test_direct_search_algorithm_type_enum_member(self):
        """AlgorithmType has DIRECT_SEARCH member."""
        assert hasattr(AlgorithmType, "DIRECT_SEARCH")
        assert AlgorithmType.DIRECT_SEARCH.value == "direct_search"
