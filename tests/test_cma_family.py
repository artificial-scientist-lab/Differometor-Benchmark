"""Algorithm-specific unit tests for the CMA-family algorithm batch.

These tests cover algorithm-specific knobs that the shared parametrised suite
in ``test_algorithms_uniform.py`` does not exercise (init params, sigma
plumbing, restart strategies, pop-size handling, mu/lambda validation).
Common protocol checks (algorithm_str, eval_count > 0, etc.) are covered by
the REGISTRY in ``test_algorithms_uniform.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.core.objective import Objective

# JAX algorithms have no optional deps
from dfbench.algorithms.evolutionary.jax_es import JAXOnePlusOneES, JAXMuLambdaES

# ---------------------------------------------------------------------------
# Optional imports: skip tests if backend unavailable
# ---------------------------------------------------------------------------

pycma_available = True
try:
    from dfbench.algorithms.evolutionary.pycma_cmaes import (
        PyCMACMAES,
        PyCMAActiveCMAES,
        PyCMAIPOP,
        PyCMABIPOP,
    )
except ImportError:
    pycma_available = False  # pragma: no cover

cmaes_available = True
try:
    from dfbench.algorithms.evolutionary.cmaes_sep_cma import CMAESSepCMA
except ImportError:
    cmaes_available = False  # pragma: no cover

evosax_available = True
try:
    from dfbench.algorithms.evolutionary.evosax_es import EvosaxMAES, EvosaxLMMAES
except ImportError:
    evosax_available = False  # pragma: no cover


skip_pycma = pytest.mark.skipif(not pycma_available, reason="pycma not installed")
skip_cmaes = pytest.mark.skipif(not cmaes_available, reason="cmaes not installed")
skip_evosax = pytest.mark.skipif(not evosax_available, reason="evosax not installed")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MAX_EVALS = 60  # enough for a few generations of most algorithms


def _make_obj(problem, max_evals: int = _MAX_EVALS):
    return Objective(problem, max_evals=max_evals, max_time=60)


# ---------------------------------------------------------------------------
# pycma-specific knobs
# ---------------------------------------------------------------------------


@skip_pycma
class TestPyCMAAlgorithms:
    def test_cmaes_smoke(self, mock_problem):
        obj = _make_obj(mock_problem)
        PyCMACMAES().optimize(obj, random_seed=1, sigma0=0.5, pop_size=5)
        assert obj.eval_count > 0

    def test_acmaes_smoke(self, mock_problem):
        obj = _make_obj(mock_problem)
        PyCMAActiveCMAES().optimize(obj, random_seed=1, pop_size=5)
        assert obj.best_loss is not None

    def test_ipop_smoke(self, mock_problem):
        obj = _make_obj(mock_problem, max_evals=80)
        PyCMAIPOP().optimize(obj, random_seed=2, pop_size=4, max_restarts=2)
        assert obj.eval_count > 0

    def test_bipop_smoke(self, mock_problem):
        obj = _make_obj(mock_problem, max_evals=80)
        PyCMABIPOP().optimize(obj, random_seed=3, pop_size=4, max_restarts=4)
        assert obj.eval_count > 0

    def test_init_params_accepted(self, mock_problem):
        """init_params are used when provided."""
        x0 = np.zeros(mock_problem._n_params)
        obj = _make_obj(mock_problem)
        PyCMACMAES().optimize(obj, init_params=x0, random_seed=4, pop_size=5)
        assert obj.eval_count > 0

    def test_bounded_mode(self, mock_problem):
        """best_params_bounded stays within problem bounds."""
        obj = _make_obj(mock_problem)
        PyCMACMAES().optimize(obj, random_seed=5, pop_size=5)
        bp = np.asarray(obj.best_params_bounded)
        lb = np.asarray(mock_problem.bounds[0])
        ub = np.asarray(mock_problem.bounds[1])
        assert np.all(bp >= lb - 1e-6)
        assert np.all(bp <= ub + 1e-6)

    def test_max_iterations_respected(self, mock_problem):
        """max_iterations caps the number of CMA generations."""
        obj = Objective(mock_problem, max_evals=10_000, max_time=60)
        PyCMACMAES().optimize(obj, random_seed=6, max_iterations=2, pop_size=5)
        # 2 generations × pop_size=5 = 10 evals (+ optional warmup)
        assert obj.eval_count <= 15  # generous upper bound

    def test_cmaes_vs_acmaes_str_differ(self):
        """Vanilla CMA and active CMA have different algorithm_str values."""
        assert PyCMACMAES().algorithm_str != PyCMAActiveCMAES().algorithm_str

    def test_ipop_vs_bipop_str_differ(self):
        assert PyCMAIPOP().algorithm_str != PyCMABIPOP().algorithm_str


# ---------------------------------------------------------------------------
# Section D: cmaes-specific tests
# ---------------------------------------------------------------------------


@skip_cmaes
class TestCMAESSepCMA:
    def test_smoke(self, mock_problem):
        obj = _make_obj(mock_problem)
        CMAESSepCMA().optimize(obj, random_seed=10, pop_size=6)
        assert obj.eval_count > 0

    def test_bounded(self, mock_problem):
        obj = _make_obj(mock_problem)
        CMAESSepCMA().optimize(obj, random_seed=11, pop_size=6)
        bp = np.asarray(obj.best_params_bounded)
        lb = np.asarray(mock_problem.bounds[0])
        ub = np.asarray(mock_problem.bounds[1])
        assert np.all(bp >= lb - 1e-6)
        assert np.all(bp <= ub + 1e-6)

    def test_max_no_improvement_stops(self, mock_problem):
        """max_no_improvement terminates the run early on flat objectives."""
        obj = Objective(mock_problem, max_evals=10_000, max_time=60)
        CMAESSepCMA().optimize(obj, random_seed=12, max_no_improvement=2, pop_size=6)
        # Should not use all 10k evals (stop on stagnation)
        assert obj.eval_count < 500


# ---------------------------------------------------------------------------
# Section E: evosax-specific tests
# ---------------------------------------------------------------------------


@skip_evosax
class TestEvosaxAlgorithms:
    def test_maes_smoke(self, mock_problem):
        obj = _make_obj(mock_problem)
        EvosaxMAES().optimize(obj, random_seed=20, pop_size=10)
        assert obj.eval_count > 0

    def test_lm_maes_smoke(self, mock_problem):
        obj = _make_obj(mock_problem)
        EvosaxLMMAES().optimize(obj, random_seed=21, pop_size=10)
        assert obj.eval_count > 0

    def test_maes_bounded(self, mock_problem):
        obj = _make_obj(mock_problem)
        EvosaxMAES().optimize(obj, random_seed=22, pop_size=10)
        bp = np.asarray(obj.best_params_bounded)
        lb = np.asarray(mock_problem.bounds[0])
        ub = np.asarray(mock_problem.bounds[1])
        assert np.all(bp >= lb - 1e-6)
        assert np.all(bp <= ub + 1e-6)

    def test_maes_vs_lm_maes_str_differ(self):
        assert EvosaxMAES().algorithm_str != EvosaxLMMAES().algorithm_str


# ---------------------------------------------------------------------------
# Section F: native JAX ES tests (no optional deps, always run)
# ---------------------------------------------------------------------------


class TestJAXOnePlusOneES:
    def test_smoke(self, mock_problem):
        obj = _make_obj(mock_problem)
        JAXOnePlusOneES().optimize(obj, random_seed=30)
        assert obj.eval_count > 0

    def test_best_loss_set(self, mock_problem):
        obj = _make_obj(mock_problem)
        JAXOnePlusOneES().optimize(obj, random_seed=31)
        assert obj.best_loss is not None

    def test_bounded(self, mock_problem):
        obj = _make_obj(mock_problem)
        JAXOnePlusOneES().optimize(obj, random_seed=32)
        bp = np.asarray(obj.best_params_bounded)
        lb = np.asarray(mock_problem.bounds[0])
        ub = np.asarray(mock_problem.bounds[1])
        assert np.all(bp >= lb - 1e-6)
        assert np.all(bp <= ub + 1e-6)

    def test_sigma_min_stops(self, mock_problem):
        """sigma_min=1e3 (very large) should stop immediately after warmup."""
        obj = Objective(mock_problem, max_evals=10_000, max_time=60)
        JAXOnePlusOneES().optimize(obj, random_seed=33, sigma0=1e-11, sigma_min=1e-10)
        # sigma drops below sigma_min immediately -> very few evals
        assert obj.eval_count < 20

    def test_max_iterations_cap(self, mock_problem):
        """max_iterations limits evaluation count."""
        obj = Objective(mock_problem, max_evals=10_000, max_time=60)
        JAXOnePlusOneES().optimize(obj, random_seed=34, max_iterations=5)
        # 5 offspring + 1 initial evaluation = 6 total
        assert obj.eval_count <= 10

    def test_init_params(self, mock_problem):
        x0 = np.array([1.0, -1.0])
        obj = _make_obj(mock_problem)
        JAXOnePlusOneES().optimize(obj, init_params=x0, random_seed=35)
        assert obj.eval_count > 0

    def test_reproducibility(self, mock_problem):
        obj1 = _make_obj(mock_problem)
        JAXOnePlusOneES().optimize(obj1, random_seed=36)

        obj2 = _make_obj(mock_problem)
        JAXOnePlusOneES().optimize(obj2, random_seed=36)

        np.testing.assert_allclose(
            [float(l) for l in obj1.loss_history],
            [float(l) for l in obj2.loss_history],
            atol=1e-5,
        )


class TestJAXMuLambdaES:
    def test_smoke(self, mock_problem):
        obj = _make_obj(mock_problem, max_evals=100)
        JAXMuLambdaES().optimize(obj, random_seed=40, mu=3, lam=10)
        assert obj.eval_count > 0

    def test_best_loss_set(self, mock_problem):
        obj = _make_obj(mock_problem, max_evals=100)
        JAXMuLambdaES().optimize(obj, random_seed=41, mu=3, lam=10)
        assert obj.best_loss is not None

    def test_bounded(self, mock_problem):
        obj = _make_obj(mock_problem, max_evals=100)
        JAXMuLambdaES().optimize(obj, random_seed=42, mu=3, lam=10)
        bp = np.asarray(obj.best_params_bounded)
        lb = np.asarray(mock_problem.bounds[0])
        ub = np.asarray(mock_problem.bounds[1])
        assert np.all(bp >= lb - 1e-6)
        assert np.all(bp <= ub + 1e-6)

    def test_invalid_mu_raises(self):
        """mu >= lam must raise ValueError at optimize() time."""
        with pytest.raises(ValueError):
            JAXMuLambdaES().optimize(None, random_seed=0, mu=10, lam=5)

    def test_equal_mu_lam_raises(self):
        with pytest.raises(ValueError):
            JAXMuLambdaES().optimize(None, random_seed=0, mu=5, lam=5)

    def test_max_iterations(self, mock_problem):
        obj = Objective(mock_problem, max_evals=10_000, max_time=60)
        JAXMuLambdaES().optimize(obj, random_seed=43, max_iterations=3, mu=2, lam=6)
        # 3 generations × 6 offspring = 18 evals
        assert obj.eval_count <= 25

    def test_reproducibility(self, mock_problem):
        obj1 = Objective(mock_problem, max_evals=100, max_time=60)
        JAXMuLambdaES().optimize(obj1, random_seed=44, mu=3, lam=10)

        obj2 = Objective(mock_problem, max_evals=100, max_time=60)
        JAXMuLambdaES().optimize(obj2, random_seed=44, mu=3, lam=10)

        np.testing.assert_allclose(
            [float(l) for l in obj1.loss_history],
            [float(l) for l in obj2.loss_history],
            atol=1e-5,
        )

    def test_5d_problem(self, mock_problem_5d):
        """Runs without error on a 5-dimensional problem."""
        obj = Objective(mock_problem_5d, max_evals=100, max_time=60)
        JAXMuLambdaES().optimize(obj, random_seed=45, mu=5, lam=20)
        assert obj.eval_count > 0
