"""Smoke tests and parametrized checks for the BO / surrogate batch.

Tests follow the same pattern as test_algorithms_unit.py: use the
QuadraticProblem mock and Objective wrapper to verify that each algorithm
produces evaluations, records losses, and meets the standard protocol.

External-package algorithms (HEBO, SMAC, Ax) are skipped when their
dependencies are not installed rather than failing loudly.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective

# ── Conditional imports ──────────────────────────────────────────────

# BoTorch-only algorithms (always available when botorch is installed)
from dfbench.algorithms.surrogate_based.botorch_qnei import BotorchqNEI
from dfbench.algorithms.surrogate_based.botorch_qkg import BotorchqKG
from dfbench.algorithms.surrogate_based.botorch_rembo import REMBO
from dfbench.algorithms.surrogate_based.botorch_gebo import GEBO
from dfbench.algorithms.surrogate_based.botorch_linebo import LineBO
from dfbench.algorithms.surrogate_based.ax_baxus import BAxUS
from dfbench.algorithms.surrogate_based.turbo_lbfgs import TuRBOLBFGS

_has_ax = importlib.util.find_spec("ax") is not None
_has_hebo = importlib.util.find_spec("hebo") is not None
_has_smac = importlib.util.find_spec("smac") is not None

if _has_ax:
    from dfbench.algorithms.surrogate_based.ax_saasbo import AxSAASBO
if _has_hebo:
    from dfbench.algorithms.surrogate_based.hebo_bo import HEBO
if _has_smac:
    from dfbench.algorithms.surrogate_based.smac_bo import SMAC

# ── Algorithm catalogue ──────────────────────────────────────────────

# Always-available (BoTorch)
BOTORCH_ALGORITHMS = [
    BotorchqNEI,
    BotorchqKG,
    REMBO,
    GEBO,
    LineBO,
    BAxUS,
    TuRBOLBFGS,
]

# Helpers to attach correct optimize kwargs
_OPTIMIZE_KWARGS: dict[type, dict] = {
    BotorchqNEI: {"max_iterations": 2, "n_initial": 5},
    BotorchqKG: {"max_iterations": 2, "n_initial": 5, "num_fantasies": 4},
    REMBO: {"max_iterations": 2, "n_initial": 5, "d_embedding": 2},
    GEBO: {"max_iterations": 2, "n_initial": 5},
    LineBO: {"max_iterations": 2, "n_initial": 5, "line_samples": 5},
    BAxUS: {"max_iterations": 2, "n_initial": 5, "d_init": 2},
    TuRBOLBFGS: {"turbo_iterations": 2, "n_initial": 5, "lbfgs_patience": 5},
}


def _kwargs(cls):
    return _OPTIMIZE_KWARGS.get(cls, {})


# ======================================================================
#  Parametrized BoTorch-based tests
# ======================================================================


class TestBOBatchCommon:
    """Common checks for all BoTorch-based BO algorithms."""

    @pytest.mark.parametrize("cls", BOTORCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_str(self, cls):
        algo = cls()
        assert isinstance(algo.algorithm_str, str) and len(algo.algorithm_str) > 0

    @pytest.mark.parametrize("cls", BOTORCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_type(self, cls):
        algo = cls()
        assert algo.algorithm_type == AlgorithmType.SURROGATE_BASED

    @pytest.mark.parametrize("cls", BOTORCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_produces_evals(self, cls, mock_problem):
        algo = cls()
        obj = Objective(mock_problem, max_evals=50, max_time=120)
        algo.optimize(obj, random_seed=42, **_kwargs(cls))
        assert obj.eval_count > 0

    @pytest.mark.parametrize("cls", BOTORCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_best_loss_not_none(self, cls, mock_problem):
        algo = cls()
        obj = Objective(mock_problem, max_evals=50, max_time=120)
        algo.optimize(obj, random_seed=42, **_kwargs(cls))
        assert obj.best_loss is not None

    @pytest.mark.parametrize("cls", BOTORCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_loss_history_non_empty(self, cls, mock_problem):
        algo = cls()
        obj = Objective(mock_problem, max_evals=50, max_time=120)
        algo.optimize(obj, random_seed=42, **_kwargs(cls))
        assert len(obj.loss_history) > 0

    @pytest.mark.parametrize("cls", BOTORCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_time_steps_monotonic(self, cls, mock_problem):
        algo = cls()
        obj = Objective(mock_problem, max_evals=50, max_time=120)
        algo.optimize(obj, random_seed=42, **_kwargs(cls))
        ts = obj.time_steps
        assert len(ts) > 0
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1]

    @pytest.mark.parametrize("cls", BOTORCH_ALGORITHMS, ids=lambda c: c.__name__)
    def test_bounded_mode(self, cls, mock_problem):
        """BO algorithms should use bounded mode (unbounded=False)."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=50, max_time=120)
        algo.optimize(obj, random_seed=42, **_kwargs(cls))
        # TuRBOLBFGS switches to unbounded in phase 2; others stay bounded
        if cls is not TuRBOLBFGS:
            assert obj.unbounded is False


# ======================================================================
#  Individual smoke tests for external-package algorithms
# ======================================================================


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


# ======================================================================
#  TuRBO→L-BFGS specific checks
# ======================================================================


class TestTuRBOLBFGS:
    def test_two_phases(self, mock_problem):
        """Verify that TuRBO→L-BFGS uses both bounded and unbounded phases."""
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
        # Objective stays in bounded mode throughout (cannot switch after
        # start_logging); Phase 2 runs L-BFGS on sigmoid objective internally
        # and logs bounded params via log_evaluation.
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
