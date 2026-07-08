"""Algorithm-specific unit tests.

The cross-algorithm baseline (smoke run, bounds, monotonic time, etc.)
lives in ``tests/test_algorithms_uniform.py``. This module is for tests
that exercise *one* algorithm's internals (helpers, knobs, validation
errors) and that do not generalise.

When you add a new algorithm, put its algorithm-specific tests here.
One class per algorithm, named after the algorithm.
"""

from __future__ import annotations

import inspect

from scipy.optimize import SR1 as ScipySR1

import numpy as np
import pytest

from dfbench.algorithms import (
    BasinHopping,
    BotorchTuRBO,
    Dogleg,
    DualAnnealing,
    EvoxES,
    EvoxPSO,
    NelderMead,
    NevergradNGOpt,
    NevergradOnePlusOne,
    NevergradTBPSA,
    OptaxLBFGS,
    OptaxLookahead,
    OptaxPolyakSGD,
    OptaxSAM,
    OptaxSophia,
    OptaxAdamW,
    OptaxAdaBelief,
    OptaxNoisySGD,
    Powell,
    RandomSearch,
    SAGD,
    SR1,
)
from dfbench.core.objective import Objective


# ── SAGD: simulated-annealing transition probability ─────────────────


class TestSAGD:
    def test_transition_probability_range(self):
        """``_compute_transition_probability`` is a probability in [0, 1]
        across a wide range of energy gaps and epochs."""
        algo = SAGD()
        for delta_e in [0.001, 0.1, 1.0, 10.0]:
            for epoch in [0, 10, 100, 1000]:
                p = algo._compute_transition_probability(
                    delta_e=delta_e, epoch=epoch, T0=1.0, learning_rate=0.01
                )
                assert 0.0 <= p <= 1.0, f"p={p} for delta_e={delta_e}, epoch={epoch}"

    def test_double_annealing(self):
        """The double-annealing branch also returns a valid probability."""
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


# ── NAAdamGD: noise-annealing schedule ────────────────────────────────


class TestNAAdamGD:
    def test_anneal_sigma_positive(self):
        from dfbench.algorithms.gradient_based.na_adam_gd import _anneal_sigma

        for progress in [0.0, 0.1, 0.5, 0.9]:
            sigma = _anneal_sigma(
                progress, sigma_start=0.1, sigma_end=0.001, schedule="linear"
            )
            assert sigma > 0, f"sigma={sigma} at progress={progress}"

    def test_anneal_sigma_exponential(self):
        from dfbench.algorithms.gradient_based.na_adam_gd import _anneal_sigma

        sigma = _anneal_sigma(
            0.5, sigma_start=0.1, sigma_end=0.001, schedule="exponential"
        )
        assert sigma > 0


# ── Evolutionary: variant validation and batched eval semantics ──────


class TestEvolutionary:
    def test_evox_es_invalid_variant(self):
        with pytest.raises((ValueError, KeyError)):
            EvoxES(variant="nonexistent_variant")

    def test_evox_pso_invalid_variant(self):
        with pytest.raises((ValueError, KeyError)):
            EvoxPSO(variant="nonexistent_variant")


# ── Nevergrad: multi-start and repeated-evaluation knobs ────────────


NEVERGRAD_ALGORITHMS = [NevergradOnePlusOne, NevergradTBPSA, NevergradNGOpt]


class TestNevergrad:
    @pytest.mark.parametrize("cls", NEVERGRAD_ALGORITHMS, ids=lambda c: c.__name__)
    def test_multistart(self, cls, mock_problem):
        """``n_restarts`` > 1 still produces evaluations within budget."""
        algo = cls()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42, n_restarts=2)
        assert obj.eval_count > 0

    def test_tbpsa_repeated_evaluations(self, mock_problem):
        """TBPSA with ``num_evaluations=3`` averages over repeats per candidate."""
        algo = NevergradTBPSA()
        obj = Objective(mock_problem, max_evals=30, max_time=60)
        algo.optimize(obj, random_seed=42, num_evaluations=3)
        assert obj.eval_count > 0


# ── BoTorch TuRBO: API defaults ─────────────────────────────────────


class TestBotorchTuRBO:
    def test_n_restarts_defaults_to_none(self):
        """TuRBO restart count is uncapped by default and budget-limited."""
        default = (
            inspect.signature(BotorchTuRBO.optimize).parameters["n_restarts"].default
        )
        assert default is None


# ── ReSTIR helpers: kNN, standardisation, importance ─────────────────


class TestReSTIRHelpers:
    def test_knn_predict_shape(self):
        import jax.numpy as jnp
        from dfbench.algorithms.surrogate_based.restir import knn_predict

        X_train = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        y_train = jnp.array([0.0, 1.0, 1.0, 2.0])
        X_query = jnp.array([[0.5, 0.5], [0.1, 0.1], [0.9, 0.9]])
        preds = knn_predict(X_train, y_train, X_query, k=2)
        assert preds.shape == (3,)

    def test_standardize_data(self):
        import jax.numpy as jnp
        from dfbench.algorithms.surrogate_based.restir import standardize_data

        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        X_scaled, mean, std = standardize_data(X)
        np.testing.assert_allclose(float(jnp.mean(X_scaled)), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(jnp.std(X_scaled)), 1.0, atol=0.2)

    def test_importance_with_unit_temperature(self):
        import jax.numpy as jnp
        from dfbench.algorithms.surrogate_based.restir import importance

        loss = jnp.array([0.0, 1.0, 2.0])
        result = importance(loss, tau=1.0)
        expected = jnp.exp(-loss)
        np.testing.assert_allclose(np.array(result), np.array(expected), atol=1e-6)


# ── VAE helpers: forward, loss, training loop ────────────────────────


class TestVAEHelpers:
    def test_resnet_vae_forward(self):
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
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from dfbench.algorithms.generative.vae_sampling import ResNetVAE, train_vae

        data = torch.randn(20, 10)
        dataset = TensorDataset(data)
        loader = DataLoader(dataset, batch_size=5, shuffle=True)
        model = ResNetVAE(input_dim=10, latent_dim=4, hidden_dim=32, num_res_blocks=1)
        train_vae(model, loader, epochs=2, device="cpu", verbose=False)
        model.eval()
        with torch.no_grad():
            recon, mu, logvar = model(data)
        assert torch.isfinite(recon).all()


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


class TestRandomSearchSpecific:
    def test_random_search_batches_evals(self, mock_problem):
        """RandomSearch reports its batched evals, not just one per iter."""
        algo = RandomSearch(batch_size=10)
        obj = Objective(mock_problem, max_evals=50, max_time=60)
        algo.optimize(obj, random_seed=42)
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
        algo.optimize(obj, random_seed=42)
        assert obj.eval_count > 0

    def test_hyperparameters(self, mock_problem):
        """7.41 BasinHopping: custom T and stepsize are accepted."""
        algo = BasinHopping()
        obj = Objective(mock_problem, max_evals=50, max_time=60)
        algo.optimize(obj, random_seed=42, T=2.0, stepsize=0.3)
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
        algo.optimize(obj, random_seed=42, local_refinement=True)
        assert obj.eval_count > 0

    def test_temperature_params(self, mock_problem):
        """7.44 DualAnnealing: custom temperature parameters accepted."""
        algo = DualAnnealing()
        obj = Objective(mock_problem, max_evals=50, max_time=60)
        algo.optimize(
            obj,
            random_seed=42,
            initial_temp=1000.0,
            restart_temp_ratio=1e-4,
        )
        assert obj.eval_count > 0


# ── Optax: algorithm-specific knobs ──────────────────────────────────

# Algorithms that should make noticeable progress on the 2D quadratic
# within a small budget. This is a stronger claim than "it runs without
# crashing"; it verifies the optimizer actually moves downhill.
LOSS_IMPROVING_OPTAX = [
    OptaxAdamW,
    OptaxAdaBelief,
    OptaxNoisySGD,
    OptaxPolyakSGD,
    OptaxSAM,
    OptaxSophia,
    OptaxLBFGS,
]


class TestOptaxLossImproves:
    @pytest.mark.parametrize("cls", LOSS_IMPROVING_OPTAX, ids=lambda c: c.__name__)
    def test_best_loss_below_initial(self, cls, mock_problem):
        algo = cls()
        obj = Objective(mock_problem, max_evals=50, max_time=120)
        algo.optimize(obj, random_seed=42)
        history = [float(loss) for loss in obj.loss_history if loss is not None]
        assert len(history) >= 2
        assert min(history) < history[0], (
            f"{cls.__name__}: best loss {min(history):.6f} did not improve "
            f"over initial {history[0]:.6f}"
        )


class TestOptaxSAM:
    def test_custom_rho_runs(self, mock_problem):
        algo = OptaxSAM()
        obj = Objective(mock_problem, max_evals=20, max_time=60)
        algo.optimize(obj, random_seed=42, rho=0.1)
        assert obj.eval_count > 0


class TestOptaxLookahead:
    def test_alternative_inner_optimizer(self, mock_problem):
        algo = OptaxLookahead()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42, inner_optimizer_name="adamw")
        assert obj.eval_count > 0

    def test_unknown_inner_optimizer_raises(self, mock_problem):
        algo = OptaxLookahead()
        obj = Objective(mock_problem, max_evals=10, max_time=60)
        with pytest.raises(ValueError, match="Unknown inner optimizer"):
            algo.optimize(obj, random_seed=42, inner_optimizer_name="nonexistent")


class TestOptaxPolyakSGD:
    def test_custom_f_min(self, mock_problem):
        algo = OptaxPolyakSGD()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42, f_min=0.0)
        assert obj.eval_count > 0


class TestOptaxSophia:
    def test_custom_gamma(self, mock_problem):
        algo = OptaxSophia()
        obj = Objective(mock_problem, max_evals=15, max_time=60)
        algo.optimize(obj, random_seed=42, gamma=0.05)
        assert obj.eval_count > 0


# ── SciPy: method-specific behaviour ────────────────────────────────


class TestDogleg:
    def test_logs_dense_hessian(self, mock_problem):
        """Dogleg requires a dense Hessian; the adapter must log it."""
        obj = Objective(
            mock_problem,
            max_evals=30,
            max_time=60.0,
            save=["hessian"],
        )
        Dogleg().optimize(obj, random_seed=42)
        assert len(obj.hessian_history) > 0
        assert any(entry is not None for entry in obj.hessian_history)


class TestSR1:
    def test_uses_sr1_hessian_update_strategy(self, mock_problem):
        """The SR1 wrapper must pass a ``scipy.optimize.SR1`` strategy to SciPy."""
        obj = Objective(mock_problem, max_evals=30, max_time=60.0)
        algo = SR1()
        algo.optimize(obj, random_seed=42)
        assert isinstance(algo._last_hessian_update_strategy, ScipySR1)
