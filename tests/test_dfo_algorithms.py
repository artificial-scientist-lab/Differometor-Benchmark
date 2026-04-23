"""Section 7b — Derivative-free algorithm unit tests with mock problem.

Smoke tests and parametrized checks for PDFO (UOBYQA, NEWUOA, LINCOA)
and Py-BOBYQA algorithms.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective

from dfbench.algorithms import (
    PDFOUOBYQA,
    PDFONEWUOA,
    PDFOLINCOA,
    PyBOBYQA,
)

DFO_ALGORITHMS = [PDFOUOBYQA, PDFONEWUOA, PDFOLINCOA, PyBOBYQA]

# Algorithms that handle box bounds natively (LINCOA, PyBOBYQA)
BOUNDED_DFO = [PDFOLINCOA, PyBOBYQA]

# Algorithms that are unconstrained under the hood (clip-based)
UNCONSTRAINED_DFO = [PDFOUOBYQA, PDFONEWUOA]


# ======================================================================
# Common DFO checks
# ======================================================================


class TestDFOCommon:
    @pytest.mark.parametrize("cls", DFO_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_str_nonempty(self, cls):
        """algorithm_str is a non-empty string."""
        algo = cls()
        assert isinstance(algo.algorithm_str, str) and len(algo.algorithm_str) > 0

    @pytest.mark.parametrize("cls", DFO_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_type_evolutionary(self, cls):
        """algorithm_type is EVOLUTIONARY (DFO methods are population-free but bounded)."""
        assert cls.algorithm_type == AlgorithmType.EVOLUTIONARY

    @pytest.mark.parametrize("cls", DFO_ALGORITHMS, ids=lambda c: c.__name__)
    def test_optimize_produces_evals(self, cls, mock_problem):
        """After optimize(), eval_count > 0."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0

    @pytest.mark.parametrize("cls", DFO_ALGORITHMS, ids=lambda c: c.__name__)
    def test_best_loss_not_none(self, cls, mock_problem):
        """After optimize(), best_loss is not None."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.best_loss is not None

    @pytest.mark.parametrize("cls", DFO_ALGORITHMS, ids=lambda c: c.__name__)
    def test_loss_history_non_empty(self, cls, mock_problem):
        """After optimize(), loss_history is non-empty."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert len(obj.loss_history) > 0

    @pytest.mark.parametrize("cls", DFO_ALGORITHMS, ids=lambda c: c.__name__)
    def test_time_steps_monotonic(self, cls, mock_problem):
        """time_steps monotonically non-decreasing."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        ts = obj.time_steps
        assert len(ts) > 0
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1]

    @pytest.mark.parametrize("cls", DFO_ALGORITHMS, ids=lambda c: c.__name__)
    def test_bounded_mode(self, cls, mock_problem):
        """DFO algorithms operate in bounded mode (unbounded=False)."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.unbounded is False

    @pytest.mark.parametrize("cls", DFO_ALGORITHMS, ids=lambda c: c.__name__)
    def test_best_params_within_bounds(self, cls, mock_problem):
        """best_params_bounded is within problem bounds."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        bp = obj.best_params_bounded
        bounds = mock_problem.bounds
        assert np.all(np.array(bp) >= np.array(bounds[0]) - 1e-6)
        assert np.all(np.array(bp) <= np.array(bounds[1]) + 1e-6)

    @pytest.mark.parametrize("cls", DFO_ALGORITHMS, ids=lambda c: c.__name__)
    def test_budget_respected(self, cls, mock_problem):
        """eval_count does not grossly exceed max_evals."""
        max_evals = 20
        algo = cls()
        obj = Objective(mock_problem, max_evals=max_evals, max_time=60)
        algo.optimize(obj, random_seed=42)
        # Allow some overshoot because DFO solvers may batch-evaluate
        # but the Objective stops logging.
        assert obj.eval_count <= max_evals + 5


# ======================================================================
# PDFO-specific tests
# ======================================================================


class TestPDFO:
    def test_uobyqa_missing_package(self, mock_problem, monkeypatch):
        """UOBYQA raises ImportError when pdfo is not available."""
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "pdfo":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        algo = PDFOUOBYQA()
        obj = Objective(mock_problem, max_evals=10, max_time=10)
        with pytest.raises(ImportError, match="PDFO"):
            algo.optimize(obj, random_seed=42)

    def test_newuoa_custom_npt(self, mock_problem):
        """NEWUOA accepts a custom npt parameter."""
        n = mock_problem.bounds.shape[1]
        npt = n + 2  # minimum allowed
        algo = PDFONEWUOA(npt=npt)
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0

    def test_lincoa_with_bounds(self, mock_problem):
        """LINCOA respects box bounds natively."""
        algo = PDFOLINCOA()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        bp = obj.best_params_bounded
        bounds = mock_problem.bounds
        assert np.all(np.array(bp) >= np.array(bounds[0]) - 1e-6)
        assert np.all(np.array(bp) <= np.array(bounds[1]) + 1e-6)

    @pytest.mark.parametrize(
        "cls", [PDFOUOBYQA, PDFONEWUOA, PDFOLINCOA], ids=lambda c: c.__name__
    )
    def test_custom_radius(self, cls, mock_problem):
        """Custom radius_init/radius_final are respected without error."""
        algo = cls(radius_init=0.5, radius_final=1e-4)
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0

    @pytest.mark.parametrize(
        "cls", [PDFOUOBYQA, PDFONEWUOA, PDFOLINCOA], ids=lambda c: c.__name__
    )
    def test_multistart(self, cls, mock_problem):
        """Multistart with n_restarts=2 produces more evals than n_restarts=1."""
        algo1 = cls(n_restarts=1)
        obj1 = Objective(mock_problem, max_evals=50, max_time=60)
        algo1.optimize(obj1, random_seed=42)

        algo2 = cls(n_restarts=2)
        obj2 = Objective(mock_problem, max_evals=50, max_time=60)
        algo2.optimize(obj2, random_seed=42)

        # With 2 restarts and enough budget, should do at least as many evals
        assert obj2.eval_count >= obj1.eval_count


# ======================================================================
# Py-BOBYQA-specific tests
# ======================================================================


class TestPyBOBYQA:
    def test_missing_package(self, mock_problem, monkeypatch):
        """PyBOBYQA raises ImportError when pybobyqa is not available."""
        import builtins
        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "pybobyqa":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)
        algo = PyBOBYQA()
        obj = Objective(mock_problem, max_evals=10, max_time=10)
        with pytest.raises(ImportError, match="Py-BOBYQA"):
            algo.optimize(obj, random_seed=42)

    def test_seek_global_minimum(self, mock_problem):
        """seek_global_minimum=True does not crash."""
        algo = PyBOBYQA(seek_global_minimum=True)
        obj = Objective(mock_problem, max_evals=40, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0

    def test_noisy_objective_mode(self, mock_problem):
        """objfun_has_noise=True does not crash."""
        algo = PyBOBYQA(objfun_has_noise=True)
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0

    def test_custom_npt(self, mock_problem):
        """Custom npt does not crash."""
        n = mock_problem.bounds.shape[1]
        algo = PyBOBYQA(npt=n + 1)  # minimum
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0

    def test_multistart(self, mock_problem):
        """n_restarts=2 runs more evaluations than n_restarts=1."""
        algo1 = PyBOBYQA(n_restarts=1)
        obj1 = Objective(mock_problem, max_evals=50, max_time=60)
        algo1.optimize(obj1, random_seed=42)

        algo2 = PyBOBYQA(n_restarts=2)
        obj2 = Objective(mock_problem, max_evals=50, max_time=60)
        algo2.optimize(obj2, random_seed=42)

        assert obj2.eval_count >= obj1.eval_count

    def test_scaling_within_bounds(self, mock_problem):
        """scaling_within_bounds=True and False both work."""
        for scaling in (True, False):
            algo = PyBOBYQA(scaling_within_bounds=scaling)
            obj = Objective(mock_problem, max_evals=20, max_time=60)
            algo.optimize(obj, random_seed=42)
            assert obj.eval_count > 0
