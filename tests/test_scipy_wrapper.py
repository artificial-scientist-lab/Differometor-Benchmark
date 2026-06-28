"""SciPy adapter behavior tests."""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.algorithms import BFGS, TrustConstr, TrustNCG
from dfbench.algorithms.gradient_based.scipy._common import (
    SciPyConfig,
    SciPyObjectiveAdapter,
)
from dfbench.core.objective import Objective
from tests.conftest import QuadraticProblem


class ConstrainedQuadraticProblem(QuadraticProblem):
    """Mock problem exposing unsupported constraint metadata."""

    def __init__(self) -> None:
        super().__init__()
        self.constraints = [{"type": "ineq", "fun": lambda x: 1.0}]


class TestSciPyObjectiveAdapter:
    def test_fun_jac_cache_deduplicates_logging(self, mock_problem):
        obj = Objective(
            mock_problem,
            unbounded=True,
            max_evals=10,
            save=["grad"],
        )
        adapter = SciPyObjectiveAdapter(
            obj,
            SciPyConfig(method="BFGS", unbounded=True, use_bounds=False),
        )

        adapter.warmup()
        obj.start_logging()

        x = np.array([0.1, -0.2])
        _ = adapter.fun(x)
        grad = adapter.jac(x)

        assert obj.eval_count == 1
        assert len(obj.loss_history) == 1
        assert len(obj.grad_history) == 1
        np.testing.assert_allclose(grad, np.array(obj.grad_history[0]), atol=1e-6)

    def test_hessp_logging_uses_explicit_strategy(self, mock_problem):
        obj = Objective(
            mock_problem,
            unbounded=True,
            max_evals=10,
            save=["grad"],
        )
        adapter = SciPyObjectiveAdapter(
            obj,
            SciPyConfig(
                method="trust-ncg",
                unbounded=True,
                use_bounds=False,
                use_hessp=True,
                cache_hessp=True,
            ),
        )

        adapter.warmup()
        obj.start_logging()

        x = np.array([0.25, -0.5])
        vector = np.array([1.0, 1.0])
        hessp = adapter.hessp(x, vector)
        hessp_cached = adapter.hessp(x, vector)

        assert obj.eval_count == 1
        assert len(obj.loss_history) == 1
        assert len(obj.grad_history) == 1
        np.testing.assert_allclose(hessp, hessp_cached, atol=1e-6)

    def test_constraints_fail_loudly(self):
        obj = Objective(ConstrainedQuadraticProblem(), max_evals=10)
        algo = TrustConstr()

        with pytest.raises(NotImplementedError, match="box constraints"):
            algo.optimize(obj, random_seed=0, maxiter=1)


class TestSciPyAlgorithmsBehavior:
    def test_budget_stop_is_clean(self, mock_problem):
        obj = Objective(mock_problem, max_evals=1, max_time=60.0)
        algo = BFGS()

        algo.optimize(obj, random_seed=0, maxiter=50)

        assert obj.eval_count >= 1
        assert obj.budget_exceeded is True
        assert obj.best_loss is not None

    def test_unsuccessful_status_warns(self, mock_problem):
        obj = Objective(mock_problem, max_evals=50, max_time=60.0)
        algo = BFGS()

        with pytest.warns(RuntimeWarning, match="exited with SciPy status"):
            algo.optimize(obj, random_seed=0, maxiter=0)

    def test_trust_ncg_budget_stop_is_clean(self, mock_problem):
        obj = Objective(mock_problem, max_evals=1, max_time=60.0)
        algo = TrustNCG()

        algo.optimize(obj, random_seed=0, maxiter=50)

        assert obj.eval_count >= 1
        assert obj.budget_exceeded is True
