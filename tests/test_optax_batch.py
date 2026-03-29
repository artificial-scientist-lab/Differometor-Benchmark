"""Tests for the Optax algorithm batch.

Smoke tests: every algorithm runs on the mock QuadraticProblem and produces
at least one evaluation with a non-None best_loss.

Heavier tests: AdamW, AdaBelief, NoisySGD, PolyakSGD, SAM, Sophia, LBFGS
are run with a larger budget and checked for loss improvement.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.core.algorithm import AlgorithmType
from dfbench.core.objective import Objective

# ── Import all Optax algorithms ──────────────────────────────────

from dfbench.algorithms import (
    OptaxAdam,
    OptaxAdamW,
    OptaxAdaBelief,
    OptaxAdafactor,
    OptaxAMSGrad,
    OptaxAdaGrad,
    OptaxAdaDelta,
    OptaxAdaMax,
    OptaxAdaMaxW,
    OptaxAdan,
    OptaxLion,
    OptaxLAMB,
    OptaxNadam,
    OptaxNadamW,
    OptaxRMSProp,
    OptaxRProp,
    OptaxRAdam,
    OptaxSGD,
    OptaxSGDM,
    OptaxNAG,
    OptaxNoisySGD,
    OptaxPolyakSGD,
    OptaxSAM,
    OptaxSophia,
    OptaxLookahead,
    OptaxScheduleFreeAdam,
    OptaxYogi,
    OptaxNovoGrad,
    OptaxOGD,
    OptaxOAdam,
    OptaxSignSGD,
    OptaxSignum,
    OptaxSM3,
    OptaxLBFGS,
)

# ── Algorithm lists ───────────────────────────────────────────────

ALL_OPTAX = [
    OptaxAdam,
    OptaxAdamW,
    OptaxAdaBelief,
    OptaxAdafactor,
    OptaxAMSGrad,
    OptaxAdaGrad,
    OptaxAdaDelta,
    OptaxAdaMax,
    OptaxAdaMaxW,
    OptaxAdan,
    OptaxLion,
    OptaxLAMB,
    OptaxNadam,
    OptaxNadamW,
    OptaxRMSProp,
    OptaxRProp,
    OptaxRAdam,
    OptaxSGD,
    OptaxSGDM,
    OptaxNAG,
    OptaxNoisySGD,
    OptaxPolyakSGD,
    OptaxSAM,
    OptaxSophia,
    OptaxLookahead,
    OptaxScheduleFreeAdam,
    OptaxYogi,
    OptaxNovoGrad,
    OptaxOGD,
    OptaxOAdam,
    OptaxSignSGD,
    OptaxSignum,
    OptaxSM3,
    OptaxLBFGS,
]

HEAVIER_SUBSET = [
    OptaxAdamW,
    OptaxAdaBelief,
    OptaxNoisySGD,
    OptaxPolyakSGD,
    OptaxSAM,
    OptaxSophia,
    OptaxLBFGS,
]


# ======================================================================
# Shared parametrised smoke tests
# ======================================================================


class TestOptaxSmoke:
    """Smoke tests: every Optax algorithm produces evals on the mock problem."""

    @pytest.mark.parametrize("cls", ALL_OPTAX, ids=lambda c: c.__name__)
    def test_algorithm_str(self, cls):
        """algorithm_str is a non-empty string starting with 'optax_'."""
        algo = cls()
        assert isinstance(algo.algorithm_str, str)
        assert algo.algorithm_str.startswith("optax_")

    @pytest.mark.parametrize("cls", ALL_OPTAX, ids=lambda c: c.__name__)
    def test_algorithm_type(self, cls):
        """algorithm_type is GRADIENT_BASED."""
        assert cls.algorithm_type == AlgorithmType.GRADIENT_BASED

    @pytest.mark.parametrize("cls", ALL_OPTAX, ids=lambda c: c.__name__)
    def test_eval_count_positive(self, cls, mock_problem):
        """After optimize(), eval_count > 0."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0

    @pytest.mark.parametrize("cls", ALL_OPTAX, ids=lambda c: c.__name__)
    def test_best_loss_not_none(self, cls, mock_problem):
        """After optimize(), best_loss is not None."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.best_loss is not None

    @pytest.mark.parametrize("cls", ALL_OPTAX, ids=lambda c: c.__name__)
    def test_loss_history_non_empty(self, cls, mock_problem):
        """After optimize(), loss_history has entries."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert len(obj.loss_history) > 0

    @pytest.mark.parametrize("cls", ALL_OPTAX, ids=lambda c: c.__name__)
    def test_unbounded_mode(self, cls, mock_problem):
        """Optax algorithms set obj.unbounded = True."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.unbounded is True

    @pytest.mark.parametrize("cls", ALL_OPTAX, ids=lambda c: c.__name__)
    def test_best_params_bounded(self, cls, mock_problem):
        """best_params_bounded is within problem bounds."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42)
        bp = obj.best_params_bounded
        bounds = mock_problem.bounds
        assert np.all(np.array(bp) >= np.array(bounds[0]) - 1e-6)
        assert np.all(np.array(bp) <= np.array(bounds[1]) + 1e-6)


# ======================================================================
# Heavier tests (larger budget, check improvement)
# ======================================================================


class TestOptaxHeavier:
    """Run a subset of algorithms with a bigger budget and check loss drops."""

    @pytest.mark.parametrize("cls", HEAVIER_SUBSET, ids=lambda c: c.__name__)
    def test_loss_improves(self, cls, mock_problem):
        """best_loss at the end should be better than the first recorded loss."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=50, max_time=120)
        algo.optimize(obj, random_seed=42)
        history = [float(l) for l in obj.loss_history if l is not None]
        assert len(history) >= 2, "Need at least 2 losses to check improvement"
        assert min(history) < history[0], (
            f"{cls.__name__}: best loss {min(history):.6f} did not improve "
            f"over initial {history[0]:.6f}"
        )

    @pytest.mark.parametrize("cls", HEAVIER_SUBSET, ids=lambda c: c.__name__)
    def test_time_steps_monotonic(self, cls, mock_problem):
        """time_steps should be monotonically non-decreasing."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=50, max_time=120)
        algo.optimize(obj, random_seed=42)
        ts = obj.time_steps
        assert len(ts) > 0
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1]


# ======================================================================
# Algorithm-specific edge-case tests
# ======================================================================


class TestOptaxSAMSpecific:
    def test_sam_rho_parameter(self, mock_problem):
        """SAM with custom rho runs without error."""
        algo = OptaxSAM()
        obj = Objective(mock_problem, max_evals=20, max_time=60)
        algo.optimize(obj, random_seed=42, rho=0.1)
        assert obj.eval_count > 0


class TestOptaxLookaheadSpecific:
    def test_different_inner_optimizers(self, mock_problem):
        """Lookahead with adamw inner optimizer runs without error."""
        algo = OptaxLookahead()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42, inner_optimizer_name="adamw")
        assert obj.eval_count > 0

    def test_invalid_inner_optimizer(self, mock_problem):
        """Lookahead with unknown inner optimizer raises ValueError."""
        algo = OptaxLookahead()
        obj = Objective(mock_problem, max_evals=10, max_time=60)
        with pytest.raises(ValueError, match="Unknown inner optimizer"):
            algo.optimize(obj, random_seed=42, inner_optimizer_name="nonexistent")


class TestOptaxPolyakSGDSpecific:
    def test_custom_f_min(self, mock_problem):
        """PolyakSGD with custom f_min runs without error."""
        algo = OptaxPolyakSGD()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42, f_min=0.0)
        assert obj.eval_count > 0


class TestOptaxSophiaSpecific:
    def test_sophia_gamma_parameter(self, mock_problem):
        """Sophia with custom gamma runs without error."""
        algo = OptaxSophia()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42, gamma=0.05)
        assert obj.eval_count > 0
