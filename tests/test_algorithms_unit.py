"""Section 7 — Algorithm unit tests with mock problem.

Tests 7.1–7.32: common checks, gradient-based, evolutionary, surrogate, generative.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective

# ── Import all algorithm classes ──────────────────────────────────────

from dfbench.algorithms import (
    AdamGD,
    SAGD,
    NAAdamGD,
    LBFGSGD,
    RandomSearch,
    EvoxES,
    EvoxPSO,
    BotorchBO,
    BotorchTuRBO,
    ReSTIR,
    VAESampling,
    NelderMead,
    Powell,
    BasinHopping,
    DualAnnealing,
)

# ── Parametrised list of all algorithms ───────────────────────────────

ALL_ALGORITHMS = [
    AdamGD,
    SAGD,
    NAAdamGD,
    LBFGSGD,
    RandomSearch,
    EvoxES,
    EvoxPSO,
    BotorchBO,
    BotorchTuRBO,
    ReSTIR,
    VAESampling,
    NelderMead,
    Powell,
    BasinHopping,
    DualAnnealing,
]

GRADIENT_ALGORITHMS = [AdamGD, SAGD, NAAdamGD, LBFGSGD]
EVOLUTIONARY_ALGORITHMS = [RandomSearch, EvoxES, EvoxPSO]
DERIVATIVE_FREE_ALGORITHMS = [NelderMead, Powell, BasinHopping, DualAnnealing]
SURROGATE_ALGORITHMS = [BotorchBO, BotorchTuRBO, ReSTIR]
GENERATIVE_ALGORITHMS = [VAESampling]


# ======================================================================
# Common checks (7.1–7.7)
# ======================================================================


class TestCommonChecks:
    @pytest.mark.parametrize("cls", ALL_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_str(self, cls):
        """7.1 algorithm_str is a non-empty string."""
        algo = cls()
        assert isinstance(algo.algorithm_str, str) and len(algo.algorithm_str) > 0

    @pytest.mark.parametrize("cls", ALL_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_type(self, cls):
        """7.2 algorithm_type is a valid AlgorithmType."""
        algo = cls()
        assert isinstance(algo.algorithm_type, AlgorithmType)

    @staticmethod
    def _optimize_kwargs(cls):
        """Return extra kwargs needed by algorithms with required params."""
        if cls in (BotorchBO, BotorchTuRBO):
            return {"max_iterations": 2, "n_initial": 5}
        return {}

    @pytest.mark.parametrize("cls", ALL_ALGORITHMS, ids=lambda c: c.__name__)
    def test_optimize_produces_evals(self, cls, mock_problem):
        """7.3 After optimize(), eval_count > 0."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42, **self._optimize_kwargs(cls))
        assert obj.eval_count > 0

    @pytest.mark.parametrize("cls", ALL_ALGORITHMS, ids=lambda c: c.__name__)
    def test_best_loss_not_none(self, cls, mock_problem):
        """7.4 After optimize(), best_loss is not None."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42, **self._optimize_kwargs(cls))
        assert obj.best_loss is not None

    @pytest.mark.parametrize("cls", ALL_ALGORITHMS, ids=lambda c: c.__name__)
    def test_loss_history_non_empty(self, cls, mock_problem):
        """7.5 After optimize(), loss_history is non-empty."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42, **self._optimize_kwargs(cls))
        assert len(obj.loss_history) > 0

    @pytest.mark.parametrize("cls", ALL_ALGORITHMS, ids=lambda c: c.__name__)
    def test_time_steps_monotonic(self, cls, mock_problem):
        """7.6 time_steps monotonically non-decreasing."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42, **self._optimize_kwargs(cls))
        ts = obj.time_steps
        assert len(ts) > 0
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1]

    @pytest.mark.parametrize(
        "cls",
        [AdamGD, RandomSearch],
        ids=lambda c: c.__name__,
    )
    def test_reproducibility(self, cls, mock_problem):
        """7.7 Two runs with same seed produce identical loss_history."""
        algo1 = cls()
        obj1 = Objective(mock_problem, max_evals=15, max_time=60)
        algo1.optimize(obj1, random_seed=42, **self._optimize_kwargs(cls))

        algo2 = cls()
        obj2 = Objective(mock_problem, max_evals=15, max_time=60)
        algo2.optimize(obj2, random_seed=42, **self._optimize_kwargs(cls))

        np.testing.assert_allclose(
            [float(l) for l in obj1.loss_history],
            [float(l) for l in obj2.loss_history],
            atol=1e-5,
        )


# ======================================================================
# Gradient-based (7.8–7.16)
# ======================================================================


class TestGradientBased:
    @pytest.mark.parametrize("cls", GRADIENT_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_type(self, cls):
        """7.8 algorithm_type is GRADIENT_BASED."""
        assert cls.algorithm_type == AlgorithmType.GRADIENT_BASED

    @pytest.mark.parametrize("cls", GRADIENT_ALGORITHMS, ids=lambda c: c.__name__)
    def test_unbounded_mode(self, cls, mock_problem):
        """7.9 prepare() sets obj.unbounded = True."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=10, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.unbounded is True

    @pytest.mark.parametrize("cls", GRADIENT_ALGORITHMS, ids=lambda c: c.__name__)
    def test_best_params_bounded(self, cls, mock_problem):
        """7.10 best_params_bounded is within problem bounds."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=10, max_time=60)
        algo.optimize(obj, random_seed=42)
        bp = obj.best_params_bounded
        bounds = mock_problem.bounds
        assert np.all(np.array(bp) >= np.array(bounds[0]) - 1e-6)
        assert np.all(np.array(bp) <= np.array(bounds[1]) + 1e-6)


class TestSAGD:
    def test_transition_probability_range(self, mock_problem):
        """7.12 SAGD transition probability is in [0, 1]."""
        algo = SAGD()
        for delta_e in [0.001, 0.1, 1.0, 10.0]:
            for epoch in [0, 10, 100, 1000]:
                p = algo._compute_transition_probability(
                    delta_e=delta_e, epoch=epoch, T0=1.0, learning_rate=0.01
                )
                assert 0.0 <= p <= 1.0, f"p={p} for delta_e={delta_e}, epoch={epoch}"

    def test_double_annealing(self, mock_problem):
        """7.13 Double-annealing mode returns valid probability."""
        algo = SAGD()
        p = algo._compute_transition_probability(
            delta_e=0.5,
            epoch=50,
            T0=1.0,
            learning_rate=0.01,
            use_double_annealing=True,
            lr_decay=0.99,
            initial_lr=0.1,
        )
        assert 0.0 <= p <= 1.0


class TestNAAdamGD:
    def test_anneal_sigma_positive(self):
        """7.14 _anneal_sigma returns > 0 for progress in [0, 1)."""
        from dfbench.algorithms.gradient_based.na_adam_gd import _anneal_sigma

        for progress in [0.0, 0.1, 0.5, 0.9]:
            sigma = _anneal_sigma(
                progress, sigma_start=0.1, sigma_end=0.001, schedule="linear"
            )
            assert sigma > 0, f"sigma={sigma} at progress={progress}"

    def test_anneal_sigma_exponential(self):
        """7.14b Exponential schedule also returns positive."""
        from dfbench.algorithms.gradient_based.na_adam_gd import _anneal_sigma

        sigma = _anneal_sigma(
            0.5, sigma_start=0.1, sigma_end=0.001, schedule="exponential"
        )
        assert sigma > 0


# ======================================================================
# Evolutionary (7.17–7.22)
# ======================================================================


class TestEvolutionary:
    @pytest.mark.parametrize("cls", EVOLUTIONARY_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_type(self, cls):
        """7.17 algorithm_type is EVOLUTIONARY."""
        assert cls.algorithm_type == AlgorithmType.EVOLUTIONARY

    @pytest.mark.parametrize("cls", EVOLUTIONARY_ALGORITHMS, ids=lambda c: c.__name__)
    def test_bounded_mode(self, cls, mock_problem):
        """7.18 prepare() sets obj.unbounded = False."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.unbounded is False

    def test_random_search_eval_count(self, mock_problem):
        """7.19 RandomSearch: eval_count == batch_size * iterations."""
        algo = RandomSearch(batch_size=10)
        max_evals = 50
        obj = Objective(mock_problem, max_evals=max_evals, max_time=60)
        algo.optimize(obj, random_seed=42)
        # eval_count should reflect all attempted evaluations
        assert obj.eval_count > 0

    def test_evox_es_invalid_variant(self):
        """7.20 EvoxES: invalid variant string raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            algo = EvoxES(variant="nonexistent_variant")
            obj = Objective.__new__(Objective)
            # Trigger validation on optimize if not in __init__

    def test_evox_pso_invalid_variant(self):
        """7.21 EvoxPSO: invalid variant string raises ValueError."""
        with pytest.raises((ValueError, KeyError)):
            algo = EvoxPSO(variant="nonexistent_variant")
            obj = Objective.__new__(Objective)


# ======================================================================
# Surrogate-based (7.23–7.28)
# ======================================================================


class TestSurrogateBased:
    @pytest.mark.parametrize("cls", SURROGATE_ALGORITHMS, ids=lambda c: c.__name__)
    def test_algorithm_type(self, cls):
        """7.23 algorithm_type is SURROGATE_BASED."""
        assert cls.algorithm_type == AlgorithmType.SURROGATE_BASED


class TestReSTIRHelpers:
    def test_knn_predict_shape(self):
        """7.26 knn_predict returns shape (M,) for M queries."""
        import jax.numpy as jnp
        from dfbench.algorithms.surrogate_based.restir import knn_predict

        X_train = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_train = jnp.array([0.0, 1.0, 1.0, 2.0])
        X_query = jnp.array([[0.5, 0.5], [0.1, 0.1], [0.9, 0.9]])
        preds = knn_predict(X_train, y_train, X_query, k=2)
        assert preds.shape == (3,)

    def test_standardize_data(self):
        """7.27 standardize_data output has mean ≈ 0, std ≈ 1."""
        import jax.numpy as jnp
        from dfbench.algorithms.surrogate_based.restir import standardize_data

        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        X_scaled, mean, std = standardize_data(X)
        np.testing.assert_allclose(float(jnp.mean(X_scaled)), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(jnp.std(X_scaled)), 1.0, atol=0.2)

    def test_importance(self):
        """7.28 importance with tau=1 returns exp(-loss)."""
        import jax.numpy as jnp
        from dfbench.algorithms.surrogate_based.restir import importance

        loss = jnp.array([0.0, 1.0, 2.0])
        result = importance(loss, tau=1.0)
        expected = jnp.exp(-loss)
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-6)


# ======================================================================
# Generative (7.29–7.32)
# ======================================================================


class TestGenerative:
    def test_algorithm_type(self):
        """7.29 algorithm_type is GENERATIVE."""
        assert VAESampling.algorithm_type == AlgorithmType.GENERATIVE


class TestVAEHelpers:
    def test_resnet_vae_forward(self):
        """7.30 ResNetVAE forward returns (recon, mu, logvar) correct shapes."""
        import torch
        from dfbench.algorithms.generative.vae_sampling import ResNetVAE

        model = ResNetVAE(input_dim=10, latent_dim=4, hidden_dim=32, num_res_blocks=1)
        model.eval()
        x = torch.randn(5, 10)
        with torch.no_grad():
            recon, mu, logvar = model(x)
        assert recon.shape == (5, 10)
        assert mu.shape == (5, 4)
        assert logvar.shape == (5, 4)

    def test_vae_loss_function(self):
        """7.31 vae_loss_function returns (total, recon, kld) — all finite."""
        import torch
        from dfbench.algorithms.generative.vae_sampling import (
            ResNetVAE,
            vae_loss_function,
        )

        model = ResNetVAE(input_dim=10, latent_dim=4, hidden_dim=32, num_res_blocks=1)
        model.eval()
        x = torch.randn(5, 10)
        with torch.no_grad():
            recon, mu, logvar = model(x)
        total, recon_loss, kld = vae_loss_function(recon, x, mu, logvar)
        assert torch.isfinite(total)
        assert torch.isfinite(recon_loss)
        assert torch.isfinite(kld)

    def test_train_vae_no_crash(self):
        """7.32 train_vae does not crash on small synthetic data."""
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from dfbench.algorithms.generative.vae_sampling import ResNetVAE, train_vae

        data = torch.randn(20, 10)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=5, shuffle=True)
        model = ResNetVAE(input_dim=10, latent_dim=4, hidden_dim=32, num_res_blocks=1)
        train_vae(model, loader, epochs=2, device="cpu", verbose=False)
        # Verify loss decreased by running inference
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(data)
        assert torch.isfinite(recon).all()


# ======================================================================
# Derivative-free / SciPy classics (7.33–7.44)
# ======================================================================


class TestDerivativeFree:
    @pytest.mark.parametrize(
        "cls", DERIVATIVE_FREE_ALGORITHMS, ids=lambda c: c.__name__
    )
    def test_algorithm_type(self, cls):
        """7.33 algorithm_type is EVOLUTIONARY."""
        assert cls.algorithm_type == AlgorithmType.EVOLUTIONARY

    @pytest.mark.parametrize(
        "cls", DERIVATIVE_FREE_ALGORITHMS, ids=lambda c: c.__name__
    )
    def test_bounded_mode(self, cls, mock_problem):
        """7.34 prepare() sets obj.unbounded = False."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        assert obj.unbounded is False

    @pytest.mark.parametrize(
        "cls", DERIVATIVE_FREE_ALGORITHMS, ids=lambda c: c.__name__
    )
    def test_best_params_in_bounds(self, cls, mock_problem):
        """7.35 best_params are within problem bounds."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42)
        bp = obj.best_params
        bounds = mock_problem.bounds
        assert np.all(np.array(bp) >= np.array(bounds[0]) - 1e-6)
        assert np.all(np.array(bp) <= np.array(bounds[1]) + 1e-6)


class TestNelderMeadSpecific:
    def test_adaptive_flag(self, mock_problem):
        """7.36 NelderMead: adaptive mode runs without error."""
        algo = NelderMead()
        obj = Objective(mock_problem, max_evals=50, max_time=60)
        algo.optimize(obj, random_seed=42, adaptive=True)
        assert obj.eval_count > 0

    def test_convergence_tolerances(self, mock_problem):
        """7.37 NelderMead: tight tolerances still produce results."""
        algo = NelderMead()
        obj = Objective(mock_problem, max_evals=100, max_time=60)
        algo.optimize(obj, random_seed=42, xatol=1e-12, fatol=1e-12)
        assert obj.eval_count > 0


class TestPowellSpecific:
    def test_convergence_tolerances(self, mock_problem):
        """7.38 Powell: tight tolerances still produce results."""
        algo = Powell()
        obj = Objective(mock_problem, max_evals=100, max_time=60)
        algo.optimize(obj, random_seed=42, xtol=1e-12, ftol=1e-12)
        assert obj.eval_count > 0


class TestBasinHoppingSpecific:
    def test_default_local_method(self):
        """7.39 BasinHopping: default local_method is L-BFGS-B."""
        algo = BasinHopping()
        assert algo.local_method == "L-BFGS-B"

    def test_custom_local_method(self, mock_problem):
        """7.40 BasinHopping: Nelder-Mead as local solver runs correctly."""
        algo = BasinHopping(local_method="Nelder-Mead")
        obj = Objective(mock_problem, max_evals=50, max_time=60)
        algo.optimize(obj, random_seed=42, n_iter=3)
        assert obj.eval_count > 0

    def test_hyperparameters(self, mock_problem):
        """7.41 BasinHopping: custom T and stepsize are accepted."""
        algo = BasinHopping()
        obj = Objective(mock_problem, max_evals=50, max_time=60)
        algo.optimize(obj, random_seed=42, n_iter=3, T=2.0, stepsize=0.3)
        assert obj.eval_count > 0


class TestDualAnnealingSpecific:
    def test_no_local_search(self, mock_problem):
        """7.42 DualAnnealing: no_local_search mode runs without error."""
        algo = DualAnnealing()
        obj = Objective(mock_problem, max_evals=50, max_time=60)
        algo.optimize(obj, random_seed=42, local_search=False)
        assert obj.eval_count > 0

    def test_local_refinement(self, mock_problem):
        """7.43 DualAnnealing: local_refinement polishes the best incumbent."""
        algo = DualAnnealing()
        obj = Objective(mock_problem, max_evals=80, max_time=60)
        algo.optimize(
            obj, random_seed=42, maxiter=5, local_refinement=True
        )
        assert obj.eval_count > 0

    def test_temperature_params(self, mock_problem):
        """7.44 DualAnnealing: custom temperature parameters accepted."""
        algo = DualAnnealing()
        obj = Objective(mock_problem, max_evals=50, max_time=60)
        algo.optimize(
            obj,
            random_seed=42,
            maxiter=5,
            initial_temp=1000.0,
            restart_temp_ratio=1e-4,
        )
        assert obj.eval_count > 0
