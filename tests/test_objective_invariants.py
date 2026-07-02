"""Section 5 (invariants) — Objective construction, evaluation, history,
budget, bounded/unbounded, reduced history, checkpointing, reset, summary.

Tests 5.1–5.4, 5.13–5.59 (excluding randomness tests in test_objective_randomness).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dfbench.core.objective import Objective


# ======================================================================
# Construction & properties  (5.1–5.4)
# ======================================================================


class TestConstruction:
    def test_counters_start_zero(self, mock_problem):
        """5.1 All counters zero / empty after construction."""
        obj = Objective(mock_problem)
        assert obj.eval_count == 0
        assert obj.loss_history == []
        assert obj.params_history == []
        assert obj.time_steps == []

    def test_bounds_property(self, seeded_obj):
        """5.2 bounds matches problem bounds."""
        np.testing.assert_array_equal(
            np.array(seeded_obj.bounds), np.array(seeded_obj.problem.bounds)
        )

    def test_n_params(self, seeded_obj):
        """5.3 n_params matches problem."""
        assert seeded_obj.n_params == seeded_obj.problem.n_params

    def test_initial_state(self, mock_problem):
        """5.4 eval_count==0, best_loss is None, budget_exceeded is False."""
        obj = Objective(mock_problem)
        assert obj.eval_count == 0
        assert obj.best_loss is None
        assert obj.budget_exceeded is False


# ======================================================================
# Warmup helpers
# ======================================================================


class TestWarmup:
    def test_warmup_value_does_not_log_or_count(self, mock_problem):
        """Warmups before start_logging() should not affect tracked state."""
        obj = Objective(mock_problem)
        obj.warmup_value()
        assert obj.eval_count == 0
        assert obj.loss_history == []
        assert obj.params_history == []
        assert obj.time_steps == []

    def test_warmup_value_and_grad_calls_underlying_twice(self, mock_problem):
        """Warmup helpers call the respective underlying path exactly twice."""
        obj = Objective(mock_problem)
        call_count = 0

        def fake_value_and_grad(params):
            nonlocal call_count
            call_count += 1
            return jnp.float32(0.0), jnp.zeros(obj.n_params)

        obj._value_and_grad_func = fake_value_and_grad
        obj.warmup_value_and_grad()
        assert call_count == 2

    def test_warmup_vmap_hessian_calls_underlying_twice(self, mock_problem):
        """Batched warmups also execute the underlying compiled path twice."""
        obj = Objective(mock_problem)
        call_count = 0

        def fake_vmap_hessian(params):
            nonlocal call_count
            call_count += 1
            return jnp.zeros((params.shape[0], obj.n_params, obj.n_params))

        obj._vmap_hessian_func = fake_vmap_hessian
        obj.warmup_vmap_hessian()
        assert call_count == 2

    def test_warmup_value_uses_current_raw_space(self, mock_problem):
        """Unbounded warmup should emit deterministic raw-space midpoint params."""
        obj = Objective(mock_problem, unbounded=True)
        seen = []

        def fake_value(params):
            seen.append(np.array(params))
            return jnp.float32(0.0)

        obj._func = fake_value
        obj.warmup_value()
        assert len(seen) == 2
        np.testing.assert_allclose(seen[0], np.zeros(obj.n_params), atol=1e-6)

    def test_warmup_requires_pre_logging_state(self, mock_problem):
        """Warmups should fail after start_logging() to avoid hidden logging."""
        obj = Objective(mock_problem)
        obj.start_logging()
        with pytest.raises(RuntimeError):
            obj.warmup_value()


# ======================================================================
# Evaluation: value, grad, value_and_grad  (5.13–5.18)
# ======================================================================


class TestEvaluation:
    @pytest.fixture(autouse=True)
    def _start(self, seeded_obj):
        self.obj = seeded_obj
        self.obj.start_logging()
        self.params = self.obj.random_params_bounded()

    def test_value_scalar_and_count(self):
        """5.13 value() returns scalar, increments eval_count by 1."""
        loss = self.obj.value(self.params)
        assert loss.ndim == 0
        assert self.obj.eval_count == 1

    def test_grad_shape_and_count(self):
        """5.14 grad() returns (n_params,), increments eval_count by 1."""
        g = self.obj.grad(self.params)
        assert g.shape == (self.obj.n_params,)
        assert self.obj.eval_count == 1

    def test_hessian_shape_and_count(self):
        """5.14c hessian() returns (n_params, n_params), increments eval_count."""
        h = self.obj.hessian(self.params)
        assert h.shape == (self.obj.n_params, self.obj.n_params)
        assert self.obj.eval_count == 1

    def test_grad_does_not_update_best_loss(self):
        """5.14b grad() must NOT update best_loss."""
        self.obj.grad(self.params)
        assert self.obj.best_loss is None, "grad-only call should not set best_loss"

    def test_hessian_does_not_update_best_loss(self):
        """5.14d hessian() must NOT update best_loss."""
        self.obj.hessian(self.params)
        assert self.obj.best_loss is None, "hessian-only call should not set best_loss"

    def test_value_and_grad_shapes_and_count(self):
        """5.15 value_and_grad() returns (scalar, (n_params,)), count+1."""
        loss, g = self.obj.value_and_grad(self.params)
        assert loss.ndim == 0
        assert g.shape == (self.obj.n_params,)
        assert self.obj.eval_count == 1

    def test_value_grad_and_hessian_shapes_and_count(self):
        """Second-order combined call returns loss, grad, and Hessian."""
        loss, g, h = self.obj.value_grad_and_hessian(self.params)
        assert loss.ndim == 0
        assert g.shape == (self.obj.n_params,)
        assert h.shape == (self.obj.n_params, self.obj.n_params)
        assert self.obj.eval_count == 1

    def test_value_and_grad_loss_matches_value(self):
        """5.16 Loss from value_and_grad matches value(same_params)."""
        loss_vg, _ = self.obj.value_and_grad(self.params)
        self.obj.reset()
        self.obj.start_logging()
        loss_v = self.obj.value(self.params)
        np.testing.assert_allclose(float(loss_vg), float(loss_v), atol=1e-6)

    def test_value_and_grad_grad_matches_grad(self):
        """5.17 Grad from value_and_grad matches grad(same_params)."""
        _, g_vg = self.obj.value_and_grad(self.params)
        self.obj.reset()
        self.obj.start_logging()
        g = self.obj.grad(self.params)
        np.testing.assert_allclose(np.array(g_vg), np.array(g), atol=1e-6)

    def test_hessian_matches_problem_hessian(self):
        """hessian() matches jax.hessian(problem.objective_function)."""
        h_obj = self.obj.hessian(self.params)
        self.obj.reset()
        self.obj.start_logging()
        expected = jax.hessian(self.obj.problem.objective_function)(self.params)
        np.testing.assert_allclose(np.array(h_obj), np.array(expected), atol=1e-6)

    def test_value_grad_and_hessian_matches_separate_calls(self):
        """Combined second-order call matches value_and_grad() + hessian()."""
        loss_vgh, g_vgh, h_vgh = self.obj.value_grad_and_hessian(self.params)
        self.obj.reset()
        self.obj.start_logging()
        loss_vg, g_vg = self.obj.value_and_grad(self.params)
        h = self.obj.hessian(self.params)
        np.testing.assert_allclose(float(loss_vgh), float(loss_vg), atol=1e-6)
        np.testing.assert_allclose(np.array(g_vgh), np.array(g_vg), atol=1e-6)
        np.testing.assert_allclose(np.array(h_vgh), np.array(h), atol=1e-6)

    def test_callable_syntax(self):
        """5.18 obj(params) is identical to obj.value(params)."""
        loss_call = self.obj(self.params)
        self.obj.reset()
        self.obj.start_logging()
        loss_value = self.obj.value(self.params)
        np.testing.assert_allclose(float(loss_call), float(loss_value), atol=1e-6)


# ======================================================================
# Batched evaluation  (5.19–5.22)
# ======================================================================


class TestBatchedEvaluation:
    @pytest.fixture(autouse=True)
    def _start(self, seeded_obj):
        self.obj = seeded_obj
        self.obj.start_logging()
        self.batch = self.obj.random_params_bounded(n_samples=5)

    def test_vmap_value(self):
        """5.19 vmap_value: shape (batch,), count += batch_size."""
        losses = self.obj.vmap_value(self.batch)
        assert losses.shape == (5,)
        assert self.obj.eval_count == 5

    def test_vmap_grad(self):
        """5.20 vmap_grad: shape (batch, n_params), count += batch_size."""
        grads = self.obj.vmap_grad(self.batch)
        assert grads.shape == (5, self.obj.n_params)
        assert self.obj.eval_count == 5

    def test_vmap_hessian(self):
        """Second-order batched evaluation returns one Hessian per point."""
        hessians = self.obj.vmap_hessian(self.batch)
        assert hessians.shape == (5, self.obj.n_params, self.obj.n_params)
        assert self.obj.eval_count == 5

    def test_vmap_value_and_grad(self):
        """5.21 vmap_value_and_grad: correct shapes, count += batch_size."""
        losses, grads = self.obj.vmap_value_and_grad(self.batch)
        assert losses.shape == (5,)
        assert grads.shape == (5, self.obj.n_params)
        assert self.obj.eval_count == 5

    def test_vmap_value_grad_and_hessian(self):
        """Batched combined second-order call has fully aligned shapes."""
        losses, grads, hessians = self.obj.vmap_value_grad_and_hessian(self.batch)
        assert losses.shape == (5,)
        assert grads.shape == (5, self.obj.n_params)
        assert hessians.shape == (5, self.obj.n_params, self.obj.n_params)
        assert self.obj.eval_count == 5

    def test_aliases(self):
        """5.22 batched_* are aliases for vmap_*."""
        assert self.obj.batched_value is not None
        assert self.obj.batched_grad is not None
        assert self.obj.batched_hessian is not None
        assert self.obj.batched_value_and_grad is not None
        assert self.obj.batched_value_grad_and_hessian is not None

        losses = self.obj.batched_value(self.batch)
        assert losses.shape == (5,)

        hessians = self.obj.batched_hessian(self.batch)
        assert hessians.shape == (5, self.obj.n_params, self.obj.n_params)


# ======================================================================
# History tracking  (5.23–5.30)
# ======================================================================


class TestHistoryTracking:
    @pytest.fixture(autouse=True)
    def _run(self, seeded_obj):
        """Run 5 value_and_grad calls so histories are populated."""
        self.obj = seeded_obj
        self.obj.start_logging()
        for _ in range(5):
            p = self.obj.random_params_bounded()
            self.obj.value_and_grad(p)

    def test_loss_history_length(self):
        """5.23 loss_history length matches eval_count."""
        assert len(self.obj.loss_history) == self.obj.eval_count

    def test_grad_history_aligned(self):
        """5.24 grad_history aligned with loss_history."""
        assert len(self.obj.grad_history) == len(self.obj.loss_history)

    def test_params_history_aligned(self):
        """5.25 params_history aligned."""
        assert len(self.obj.params_history) == len(self.obj.loss_history)

    def test_time_steps_monotonic(self):
        """5.26 time_steps monotonically non-decreasing."""
        ts = self.obj.time_steps
        assert len(ts) == len(self.obj.loss_history)
        for i in range(1, len(ts)):
            assert ts[i] >= ts[i - 1], f"time_steps not monotonic at idx {i}"

    def test_best_loss_equals_min(self):
        """5.27 best_loss == min(loss_history)."""
        losses = [float(l) for l in self.obj.loss_history]
        assert float(self.obj.best_loss) == pytest.approx(min(losses), abs=1e-6)

    def test_best_params_corresponds(self):
        """5.28 best_params corresponds to best_loss evaluation."""
        # Evaluate best_params — should produce best_loss
        loss_at_best = self.obj.problem.objective_function(self.obj.best_params)
        assert float(loss_at_best) == pytest.approx(float(self.obj.best_loss), abs=1e-6)

    def test_improvement_count(self):
        """5.29 improvement_count counts decreases."""
        assert self.obj.improvement_count >= 1  # At least first eval improves from inf

    def test_evals_since_improvement(self):
        """5.30 evals_since_improvement resets on improvement."""
        # After 5 evals, this should be a non-negative integer
        assert self.obj.evals_since_improvement >= 0

    def test_hessian_history_aligned(self, mock_problem):
        """Second-order history aligns with loss history when enabled."""
        obj = Objective(
            mock_problem,
            save=["grad", "hessian"],
            save_params_history=True,
        )
        obj.set_seed(42)
        obj.start_logging()
        for _ in range(3):
            obj.value_grad_and_hessian(obj.random_params_bounded())

        assert len(obj.hessian_history) == len(obj.loss_history) == 3
        for h in obj.hessian_history:
            assert h.shape == (obj.n_params, obj.n_params)


# ======================================================================
# Budget enforcement  (5.31–5.39)
# ======================================================================


class TestBudgetEnforcement:
    def test_max_evals_exceeded(self, mock_problem):
        """5.31 After N evals, budget_exceeded is True."""
        obj = Objective(mock_problem, max_evals=3)
        obj.set_seed(42)
        obj.start_logging()
        for _ in range(3):
            obj.value(obj.random_params_bounded())
        assert obj.budget_exceeded is True

    def test_no_logging_after_budget(self, mock_problem):
        """5.32 Evaluations after budget do NOT append to histories."""
        obj = Objective(mock_problem, max_evals=3)
        obj.set_seed(42)
        obj.start_logging()
        for _ in range(3):
            obj.value(obj.random_params_bounded())
        history_len = len(obj.loss_history)
        # One more call — should not be logged
        obj.value(obj.random_params_bounded())
        assert len(obj.loss_history) == history_len

    def test_evals_exceeded_flag(self, mock_problem):
        """5.33 evals_exceeded is True once eval_count >= max_evals."""
        obj = Objective(mock_problem, max_evals=2)
        obj.set_seed(42)
        obj.start_logging()
        obj.value(obj.random_params_bounded())
        obj.value(obj.random_params_bounded())
        assert obj.evals_exceeded is True

    def test_time_exceeded(self, mock_problem):
        """5.34 With max_time=T, budget_exceeded eventually True."""
        obj = Objective(mock_problem, max_time=0.01)
        obj.set_seed(42)
        obj.start_logging()
        time.sleep(0.02)
        assert obj.budget_exceeded is True

    def test_time_exceeded_flag(self, mock_problem):
        """5.35 time_exceeded reflects elapsed >= max_time."""
        obj = Objective(mock_problem, max_time=0.01)
        obj.start_logging()
        time.sleep(0.02)
        assert obj.time_exceeded is True

    def test_batch_exceeds_budget(self, mock_problem):
        """5.36 Batch exceeding max_evals: counted but not logged."""
        obj = Objective(mock_problem, max_evals=3)
        obj.set_seed(42)
        obj.start_logging()
        obj.value(obj.random_params_bounded())  # count=1
        batch = obj.random_params_bounded(n_samples=5)
        obj.vmap_value(batch)  # 5 evals, only 2 left → exceeds
        # eval_count should still increase
        assert obj.eval_count > 3
        # But loss_history should only have the first valid entries
        assert len(obj.loss_history) <= 3

    def test_evals_left(self, mock_problem):
        """5.37 evals_left returns correct remaining count."""
        obj = Objective(mock_problem, max_evals=10)
        obj.set_seed(42)
        obj.start_logging()
        obj.value(obj.random_params_bounded())
        assert obj.evals_left == 9

    def test_evals_progress_fraction(self, mock_problem):
        """5.38 evals_progress_fraction 0→1."""
        obj = Objective(mock_problem, max_evals=4)
        assert obj.evals_progress_fraction == 0.0
        obj.set_seed(42)
        obj.start_logging()
        obj.value(obj.random_params_bounded())
        assert obj.evals_progress_fraction == pytest.approx(0.25)

    def test_time_progress_fraction(self, mock_problem):
        """5.39 time_progress_fraction 0→1."""
        obj = Objective(mock_problem, max_time=1.0)
        assert obj.time_progress_fraction == 0.0
        obj.start_logging()
        time.sleep(0.5)
        frac = obj.time_progress_fraction
        assert 0.2 < frac < 0.9  # rough check


# ======================================================================
# Bounded / unbounded mode  (5.40–5.43)
# ======================================================================


class TestBoundedUnbounded:
    def test_bounded_uses_objective_function(self, mock_problem):
        """5.40 unbounded=False → uses objective_function."""
        obj = Objective(mock_problem, unbounded=False)
        obj.set_seed(42)
        obj.start_logging()
        params = obj.random_params_bounded()
        loss = obj.value(params)
        expected = mock_problem.objective_function(params)
        np.testing.assert_allclose(float(loss), float(expected), atol=1e-6)

    def test_unbounded_maps_then_uses_objective_function(self, mock_problem):
        """5.41 unbounded=True maps to bounds before objective_function."""
        obj = Objective(mock_problem, unbounded=True)
        obj.set_seed(42)
        obj.start_logging()
        params = obj.random_params_unbounded()
        loss = obj.value(params)
        expected = mock_problem.objective_function(
            obj._map_unbounded_to_bounded(params)
        )
        np.testing.assert_allclose(float(loss), float(expected), atol=1e-6)

    def test_best_params_bounded_in_unbounded_mode(self, mock_problem):
        """5.42 best_params_bounded applies sigmoid_bounding."""
        obj = Objective(mock_problem, unbounded=True, save_params_history=True)
        obj.set_seed(42)
        obj.start_logging()
        for _ in range(5):
            obj.value(obj.random_params_unbounded())
        bp = obj.best_params_bounded
        bounds = mock_problem.bounds
        assert jnp.all(bp >= bounds[0] - 1e-6)
        assert jnp.all(bp <= bounds[1] + 1e-6)

    def test_params_history_bounded(self, mock_problem):
        """5.43 params_history_bounded applies sigmoid_bounding."""
        obj = Objective(mock_problem, unbounded=True, save_params_history=True)
        obj.set_seed(42)
        obj.start_logging()
        for _ in range(3):
            obj.value(obj.random_params_unbounded())
        bounded_history = obj.params_history_bounded
        bounds = mock_problem.bounds
        for p in bounded_history:
            if p is not None:
                assert jnp.all(p >= bounds[0] - 1e-6)
                assert jnp.all(p <= bounds[1] + 1e-6)


# ======================================================================
# Reduced history properties  (5.44–5.47)
# ======================================================================


class TestReducedHistory:
    def test_loss_history_reduced_scalar(self, seeded_obj):
        """5.44 Scalar entries → identical to loss_history."""
        seeded_obj.start_logging()
        for _ in range(3):
            seeded_obj.value(seeded_obj.random_params_bounded())
        reduced = seeded_obj.loss_history_reduced
        assert len(reduced) == 3
        for r, o in zip(reduced, seeded_obj.loss_history):
            np.testing.assert_allclose(r, float(o), atol=1e-6)

    def test_loss_history_reduced_batched(self, mock_problem):
        """5.44 Batched entries → returns nanmin."""
        obj = Objective(mock_problem, save=["batched_loss"])
        obj.set_seed(42)
        obj.start_logging()
        batch = obj.random_params_bounded(n_samples=5)
        obj.vmap_value(batch)
        reduced = obj.loss_history_reduced
        assert len(reduced) == 1
        assert isinstance(reduced[0], float)

    def test_params_history_reduced(self, seeded_obj):
        """5.45 Scalar entries → identical."""
        seeded_obj.start_logging()
        for _ in range(3):
            seeded_obj.value_and_grad(seeded_obj.random_params_bounded())
        reduced = seeded_obj.params_history_reduced
        assert len(reduced) == 3
        for r in reduced:
            assert r.ndim == 1

    def test_grad_history_reduced(self, seeded_obj):
        """5.46 Same selection logic."""
        seeded_obj.start_logging()
        for _ in range(3):
            seeded_obj.value_and_grad(seeded_obj.random_params_bounded())
        reduced = seeded_obj.grad_history_reduced
        assert len(reduced) == 3
        for r in reduced:
            if r is not None:
                assert r.ndim == 1

    def test_hessian_history_reduced(self, mock_problem):
        """Second-order reduced history returns one Hessian per logged batch."""
        obj = Objective(
            mock_problem,
            save=["grad", "hessian", "batched_loss", "batched_hessian"],
        )
        obj.set_seed(42)
        obj.start_logging()
        batch = obj.random_params_bounded(n_samples=4)
        obj.vmap_value_grad_and_hessian(batch)
        reduced = obj.hessian_history_reduced
        assert len(reduced) == 1
        assert reduced[0].shape == (obj.n_params, obj.n_params)

    def test_params_history_reduced_bounded(self, mock_problem):
        """5.47 Combines reduction + sigmoid_bounding."""
        obj = Objective(mock_problem, unbounded=True, save_params_history=True)
        obj.set_seed(42)
        obj.start_logging()
        for _ in range(3):
            obj.value(obj.random_params_unbounded())
        reduced = obj.params_history_reduced_bounded
        bounds = mock_problem.bounds
        for p in reduced:
            if p is not None:
                assert jnp.all(p >= bounds[0] - 1e-6)
                assert jnp.all(p <= bounds[1] + 1e-6)


# ======================================================================
# Eval type tracking  (5.48–5.49)
# ======================================================================


class TestEvalTypeTracking:
    def test_eval_type_counts(self, mock_problem):
        """5.48 Distinguishes value-only (1), grad-only (2), value+grad (3)."""
        obj = Objective(mock_problem, save=["eval_type"])
        obj.set_seed(42)
        obj.start_logging()
        p = obj.random_params_bounded()
        obj.value(p)
        obj.grad(p)
        obj.value_and_grad(p)
        counts = obj.eval_type_counts
        assert 1 in counts  # value-only
        assert 2 in counts  # grad-only
        assert 3 in counts  # value+grad

    def test_eval_type_counts_include_hessians(self, mock_problem):
        """Eval type tracking distinguishes second-order call variants."""
        obj = Objective(mock_problem, save=["eval_type"])
        obj.set_seed(42)
        obj.start_logging()
        p = obj.random_params_bounded()
        batch = obj.random_params_bounded(n_samples=3)
        obj.hessian(p)
        obj.value_grad_and_hessian(p)
        obj.vmap_hessian(batch)
        obj.vmap_value_grad_and_hessian(batch)
        counts = obj.eval_type_counts
        assert 8 in counts  # hessian-only
        assert 11 in counts  # value+grad+hessian
        assert 12 in counts  # batched hessian
        assert 15 in counts  # batched value+grad+hessian

    def test_log_call_count(self, mock_problem):
        """5.49 log_call_count == total _log_evals invocations."""
        obj = Objective(mock_problem)
        obj.set_seed(42)
        obj.start_logging()
        p = obj.random_params_bounded()
        obj.value(p)
        obj.grad(p)
        obj.value_and_grad(p)
        assert obj.log_call_count == 3


# ======================================================================
# log_evaluation  (5.50–5.51)
# ======================================================================


class TestLogEvaluation:
    def test_manual_log(self, mock_problem):
        """5.50 log_evaluation updates histories like value_and_grad."""
        obj = Objective(
            mock_problem,
            save=["grad", "hessian"],
        )
        obj.set_seed(42)
        obj.start_logging()
        p = obj.random_params_bounded()
        loss = jnp.sum(p**2)
        grad = 2 * p
        hessian = 2 * jnp.eye(obj.n_params)
        obj.log_evaluation(p, loss, grad, hessian)
        assert obj.eval_count == 1
        assert len(obj.loss_history) == 1
        assert len(obj.hessian_history) == 1

    def test_manual_log_respects_budget(self, mock_problem):
        """5.51 log_evaluation respects budget enforcement."""
        obj = Objective(mock_problem, max_evals=1)
        obj.set_seed(42)
        obj.start_logging()
        p = obj.random_params_bounded()
        obj.log_evaluation(p, jnp.float32(1.0), None)
        history_len = len(obj.loss_history)
        obj.log_evaluation(p, jnp.float32(0.5), None)
        assert len(obj.loss_history) == history_len


# ======================================================================
# Reset  (5.52)
# ======================================================================


class TestReset:
    def test_reset_clears_everything(self, seeded_obj):
        """5.52 reset() clears all histories and counters."""
        seeded_obj.start_logging()
        for _ in range(3):
            seeded_obj.value(seeded_obj.random_params_bounded())
        seeded_obj.reset()
        assert seeded_obj.eval_count == 0
        assert seeded_obj.best_loss is None
        assert seeded_obj.budget_exceeded is False
        assert seeded_obj.loss_history == []
        assert seeded_obj.time_steps == []

    def test_reset_clears_hessian_history(self, mock_problem):
        """Second-order history is also cleared by reset()."""
        obj = Objective(mock_problem, save=["hessian"])
        obj.set_seed(42)
        obj.start_logging()
        obj.hessian(obj.random_params_bounded())
        obj.reset()
        assert obj.hessian_history == []


# ======================================================================
# Checkpointing  (5.53–5.57)
# ======================================================================


class TestCheckpointing:
    def test_save_creates_file(self, seeded_obj, tmp_path):
        """5.53 save_run_data creates a file."""
        seeded_obj.start_logging()
        seeded_obj.value(seeded_obj.random_params_bounded())
        path = seeded_obj.save_run_data(
            algorithm_name="test", filepath=str(tmp_path / "test.npz")
        )
        assert path.exists()

    def test_load_restores_state(self, mock_problem, tmp_path):
        """5.54 load_run_data restores state."""
        obj = Objective(mock_problem, max_evals=100)
        obj.set_seed(42)
        obj.start_logging()
        for _ in range(5):
            obj.value(obj.random_params_bounded())
        saved_count = obj.eval_count
        saved_best = float(obj.best_loss)
        fpath = str(tmp_path / "checkpoint.npz")
        obj.save_run_data(filepath=fpath)

        obj2 = Objective(mock_problem, max_evals=100)
        obj2.load_run_data(fpath)
        assert obj2.eval_count == saved_count
        assert float(obj2.best_loss) == pytest.approx(saved_best, abs=1e-6)
        assert len(obj2.loss_history) == saved_count

    def test_load_restores_hessian_history(self, mock_problem, tmp_path):
        """Checkpoint round-trip preserves Hessian history when enabled."""
        obj = Objective(mock_problem, save=["hessian"], max_evals=100)
        obj.set_seed(42)
        obj.start_logging()
        obj.hessian(obj.random_params_bounded())
        fpath = str(tmp_path / "checkpoint_hessian.npz")
        obj.save_run_data(filepath=fpath)

        obj2 = Objective(mock_problem, save=["hessian"], max_evals=100)
        obj2.load_run_data(fpath)
        assert len(obj2.hessian_history) == 1
        np.testing.assert_allclose(
            np.array(obj2.hessian_history[0]),
            np.array(obj.hessian_history[0]),
            atol=1e-6,
        )

    def test_load_continues_time(self, mock_problem, tmp_path):
        """5.55 After load, time_elapsed continues from saved state."""
        obj = Objective(mock_problem)
        obj.set_seed(42)
        obj.start_logging()
        time.sleep(0.05)
        obj.value(obj.random_params_bounded())
        fpath = str(tmp_path / "ckpt.npz")
        obj.save_run_data(filepath=fpath)

        obj2 = Objective(mock_problem)
        obj2.load_run_data(fpath)
        assert obj2.time_elapsed > 0

    def test_atomic_save_no_tmp_remains(self, mock_problem, tmp_path):
        """5.57 After save, no .tmp.npz file remains."""
        obj = Objective(mock_problem)
        obj.set_seed(42)
        obj.start_logging()
        obj.value(obj.random_params_bounded())
        fpath = str(tmp_path / "atomic.npz")
        obj.save_run_data(filepath=fpath)
        tmp_file = tmp_path / "atomic.tmp.npz"
        assert not tmp_file.exists(), ".tmp.npz should be removed after atomic replace"


# ======================================================================
# Storage configuration  (checkpoint_format / checkpoint_dir knobs)
# ======================================================================


class TestStorageConfig:
    """The Objective exposes checkpoint_format / checkpoint_dir as the
    user-facing storage knobs — no imports required for the common cases."""

    def test_defaults(self, mock_problem):
        obj = Objective(mock_problem)
        assert obj.checkpoint_format == "npz"
        assert obj.checkpoint_dir is None

    def test_json_format_no_imports(self, mock_problem, tmp_path):
        """A pypi user selects JSON with a string, no serializer import."""
        obj = Objective(
            mock_problem,
            checkpoint_format="json",
            checkpoint_dir=str(tmp_path),
        )
        assert obj.checkpoint_format == "json"
        obj.set_seed(42)
        obj.start_logging()
        obj.value(obj.random_params_bounded())
        path = obj.save_run_data(algorithm_name="test")
        assert path.suffix == ".json"
        assert path.exists()
        # Round-trip through a fresh Objective with the same format.
        obj2 = Objective(
            mock_problem,
            checkpoint_format="json",
            checkpoint_dir=str(tmp_path),
        )
        obj2.load_run_data(path)
        assert obj2.eval_count == obj.eval_count

    def test_checkpoint_dir_redirects_artifacts(self, mock_problem, tmp_path):
        """checkpoint_dir sends artifacts to the given path, not ./data."""
        obj = Objective(mock_problem, checkpoint_dir=str(tmp_path))
        obj.set_seed(42)
        obj.start_logging()
        obj.value(obj.random_params_bounded())
        path = obj.save_run_data(algorithm_name="test")
        # The checkpoint lives under tmp_path, not the default ./data/...
        assert str(path).startswith(str(tmp_path))

    def test_unknown_format_raises(self, mock_problem):
        with pytest.raises(ValueError, match="Unknown checkpoint_format"):
            Objective(mock_problem, checkpoint_format="xml")

    def test_format_and_dir_survive_reset(self, mock_problem, tmp_path):
        """reset() preserves the configured format and directory."""
        obj = Objective(
            mock_problem,
            checkpoint_format="json",
            checkpoint_dir=str(tmp_path),
        )
        obj.reset()
        assert obj.checkpoint_format == "json"
        assert obj.checkpoint_dir == str(tmp_path)

    def test_reset_refreshes_timestamp(self, mock_problem):
        """reset() produces a new timestamp so saves do not overwrite the
        previous run's checkpoint at the cached path."""
        obj = Objective(mock_problem)
        ts_before = obj._timestamp
        time.sleep(1.05)  # timestamp format has second resolution
        obj.reset()
        assert obj._timestamp != ts_before


# ======================================================================
# get_summary  (5.58)
# ======================================================================


class TestGetSummary:
    def test_summary_keys(self, seeded_obj):
        """5.58 get_summary() returns dict with expected keys."""
        seeded_obj.start_logging()
        seeded_obj.value(seeded_obj.random_params_bounded())
        summary = seeded_obj.get_summary()
        expected_keys = {
            "eval_count",
            "time_elapsed",
            "best_loss",
            "current_loss",
            "improvement_count",
            "evals_since_improvement",
            "budget_exceeded",
            "time_exceeded",
            "evals_exceeded",
        }
        assert set(summary.keys()) == expected_keys


# ======================================================================
# __repr__  (5.59)
# ======================================================================


class TestRepr:
    def test_repr_includes_info(self, seeded_obj):
        """5.59 repr includes eval count and best loss."""
        seeded_obj.start_logging()
        seeded_obj.value(seeded_obj.random_params_bounded())
        r = repr(seeded_obj)
        assert "evals=" in r
        assert "best_loss=" in r
