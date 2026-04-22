"""Algorithm-specific tests for the BO / surrogate batch.

The shared protocol checks (smoke, algorithm_str, bounds, etc.) are covered
by the REGISTRY-driven tests in ``test_algorithms_uniform.py``. Only knobs
that the uniform suite cannot express belong here.

External-package algorithms (HEBO, SMAC, Ax) are skipped when their
dependencies are not installed.
"""

from __future__ import annotations

import importlib

import pytest

from dfbench.algorithms.surrogate_based.turbo_lbfgs import TuRBOLBFGS
from dfbench.core.objective import Objective

_has_ax = importlib.util.find_spec("ax") is not None
_has_hebo = importlib.util.find_spec("hebo") is not None
_has_smac = importlib.util.find_spec("smac") is not None

if _has_ax:
    from dfbench.algorithms.surrogate_based.ax_saasbo import AxSAASBO
if _has_hebo:
    from dfbench.algorithms.surrogate_based.hebo_bo import HEBO
if _has_smac:
    from dfbench.algorithms.surrogate_based.smac_bo import SMAC


@pytest.mark.skipif(not _has_ax, reason="ax-platform not installed")
class TestAxSAASBO:
    def test_smoke(self, mock_problem):
        algo = AxSAASBO()
        obj = Objective(mock_problem, max_evals=30, max_time=120)
        algo.optimize(
            obj,
            random_seed=42,
            n_initial=5,
            max_iterations=2,
            num_warmup=8,
            num_samples=4,
        )
        assert obj.eval_count > 0
        assert obj.best_loss is not None


@pytest.mark.skipif(not _has_hebo, reason="HEBO not installed")
class TestHEBO:
    def test_smoke(self, mock_problem):
        algo = HEBO()
        obj = Objective(mock_problem, max_evals=30, max_time=120)
        algo.optimize(
            obj,
            random_seed=42,
            batch_size=1,
            max_iterations=10,
        )
        assert obj.eval_count > 0
        assert obj.best_loss is not None


@pytest.mark.skipif(not _has_smac, reason="SMAC3 not installed")
class TestSMAC:
    def test_smoke(self, mock_problem):
        algo = SMAC()
        obj = Objective(mock_problem, max_evals=30, max_time=120)
        algo.optimize(
            obj,
            random_seed=42,
            max_iterations=5,
            n_initial=5,
        )
        assert obj.eval_count > 0
        assert obj.best_loss is not None


class TestTuRBOLBFGS:
    def test_two_phases(self, mock_problem):
        """TuRBO->L-BFGS runs both phases without entering unbounded mode."""
        algo = TuRBOLBFGS()
        obj = Objective(mock_problem, max_evals=50, max_time=120)
        algo.optimize(
            obj,
            random_seed=42,
            turbo_iterations=3,
            n_initial=5,
            lbfgs_patience=5,
        )
        assert obj.eval_count > 0
        assert obj.unbounded is False

    def test_eval_count_reasonable(self, mock_problem):
        """TuRBO phase + L-BFGS phase should produce at least n_initial evals."""
        algo = TuRBOLBFGS()
        obj = Objective(mock_problem, max_evals=50, max_time=120)
        algo.optimize(
            obj,
            random_seed=42,
            turbo_iterations=2,
            n_initial=5,
            lbfgs_patience=3,
        )
        assert obj.eval_count >= 5
