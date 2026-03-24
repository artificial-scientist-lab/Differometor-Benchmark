"""Standalone tests for the native-JAX custom/hybrid algorithm batch.

Kept separate to avoid merge conflicts with shared test files across
parallel algorithm branches.  Will be consolidated post-merge.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.core.objective import Objective
from dfbench.algorithms.gradient_based.custom_jax import (
    SGLDJAX,
    ASAMJAX,
    AdamToLBFGSJAX,
    EntropySGDJAX,
    SGHMCJAX,
    ARCJAX,
    OGDJAX,
    OAdamJAX,
    PerturbedGDJAX,
    NoisyAdamJAX,
    GDRestartsJAX,
    GaussianSmoothingGDJAX,
)

# All classes that actually run (ARCJAX excluded — intentionally disabled)
RUNNABLE = [
    SGLDJAX,
    ASAMJAX,
    AdamToLBFGSJAX,
    EntropySGDJAX,
    SGHMCJAX,
    OGDJAX,
    OAdamJAX,
    PerturbedGDJAX,
    NoisyAdamJAX,
    GDRestartsJAX,
    GaussianSmoothingGDJAX,
]


class TestCustomJAXBatch:
    """Smoke-level tests for the 12 native-JAX algorithms."""

    @pytest.mark.parametrize("cls", RUNNABLE, ids=lambda c: c.__name__)
    def test_smoke(self, cls, mock_problem):
        """Algorithm runs and produces at least one evaluation."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=12, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0
        assert obj.best_loss is not None

    def test_arc_not_implemented(self, mock_problem):
        """ARCJAX is intentionally disabled and must raise NotImplementedError."""
        algo = ARCJAX()
        obj = Objective(mock_problem, max_evals=10, max_time=60)
        with pytest.raises(NotImplementedError):
            algo.optimize(obj, random_seed=42)

    @pytest.mark.parametrize("cls", RUNNABLE, ids=lambda c: c.__name__)
    def test_algorithm_str(self, cls):
        """Each class exposes a non-empty algorithm_str."""
        assert isinstance(cls.algorithm_str, str) and cls.algorithm_str

    @pytest.mark.parametrize("cls", RUNNABLE, ids=lambda c: c.__name__)
    def test_best_params_bounded(self, cls, mock_problem):
        """Best params stay within problem bounds."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=12, max_time=60)
        algo.optimize(obj, random_seed=42)
        bp = obj.best_params
        bounds = mock_problem.bounds
        assert bp is not None
        assert np.all(np.array(bp) >= np.array(bounds[0]) - 1e-6)
        assert np.all(np.array(bp) <= np.array(bounds[1]) + 1e-6)

    @pytest.mark.parametrize("cls", RUNNABLE + [ARCJAX], ids=lambda c: c.__name__)
    def test_import_from_top_level(self, cls):
        """Each class is importable from the top-level algorithms package."""
        import dfbench.algorithms as alg

        assert hasattr(alg, cls.__name__)
