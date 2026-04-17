"""SciPy optimizer smoke tests."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import SR1 as ScipySR1

from dfbench.algorithms import (
    BFGS,
    COBYLA,
    COBYQA,
    Dogleg,
    LBFGSB,
    NewtonCG,
    NonlinearCG,
    SLSQP,
    SR1,
    TNC,
    TrustConstr,
    TrustKrylov,
    TrustNCG,
)
from dfbench.core.objective import Objective


SCIPY_UNBOUNDED_ALGORITHMS = [
    BFGS,
    NonlinearCG,
    NewtonCG,
    TrustNCG,
    TrustKrylov,
    Dogleg,
]

SCIPY_BOUNDED_ALGORITHMS = [
    LBFGSB,
    TrustConstr,
    TNC,
    SLSQP,
    COBYQA,
    COBYLA,
    SR1,
]

ALL_SCIPY_ALGORITHMS = SCIPY_UNBOUNDED_ALGORITHMS + SCIPY_BOUNDED_ALGORITHMS


@pytest.mark.parametrize("cls", ALL_SCIPY_ALGORITHMS, ids=lambda c: c.__name__)
def test_scipy_algorithm_smoke(cls, mock_problem):
    algo = cls()
    obj = Objective(mock_problem, max_evals=30, max_time=60.0)

    algo.optimize(obj, random_seed=42)

    assert obj.eval_count > 0
    assert obj.best_loss is not None
    assert np.isfinite(float(obj.best_loss))


@pytest.mark.parametrize("cls", SCIPY_UNBOUNDED_ALGORITHMS, ids=lambda c: c.__name__)
def test_unbounded_scipy_best_params_bounded_are_in_bounds(cls, mock_problem):
    algo = cls()
    obj = Objective(mock_problem, max_evals=30, max_time=60.0)

    algo.optimize(obj, random_seed=42)

    bounds = np.array(mock_problem.bounds)
    best = np.array(obj.best_params_bounded)
    assert np.all(best >= bounds[0] - 1e-6)
    assert np.all(best <= bounds[1] + 1e-6)


@pytest.mark.parametrize("cls", SCIPY_BOUNDED_ALGORITHMS, ids=lambda c: c.__name__)
def test_bounded_scipy_raw_best_params_stay_in_bounds(cls, mock_problem):
    algo = cls()
    obj = Objective(mock_problem, max_evals=30, max_time=60.0)

    algo.optimize(obj, random_seed=42)

    bounds = np.array(mock_problem.bounds)
    best = np.array(obj.best_params)
    assert np.all(best >= bounds[0] - 1e-6)
    assert np.all(best <= bounds[1] + 1e-6)


def test_dogleg_logs_dense_hessian(mock_problem):
    obj = Objective(
        mock_problem,
        max_evals=30,
        max_time=60.0,
        save_hessian_history=True,
    )
    algo = Dogleg()

    algo.optimize(obj, random_seed=42)

    assert len(obj.hessian_history) > 0
    assert any(entry is not None for entry in obj.hessian_history)


def test_sr1_uses_sr1_hessian_update_strategy(mock_problem):
    obj = Objective(mock_problem, max_evals=30, max_time=60.0)
    algo = SR1()

    algo.optimize(obj, random_seed=42)

    assert isinstance(algo._last_hessian_update_strategy, ScipySR1)
