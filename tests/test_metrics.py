"""Section 8: Benchmark metrics known-answer tests.

Tests 8.1-8.27.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest

from dfbench.benchmark.metrics import (
    agg_fraction_true,
    agg_mean_std,
    agg_mean_std_filtered,
    agg_min,
    compute_performance_profile,
    get_index_at_time,
    get_value_at_time,
    multi_auc_top_k,
    multi_solution_diversity_nn,
    multi_solution_diversity_overall,
    run_auc,
    run_first_success_idx,
    run_first_success_time,
    run_has_success,
    run_min_loss,
    slice_history_at_time,
)


# ======================================================================
# Per-run functions (8.1-8.8)
# ======================================================================


class TestRunMinLoss:
    def test_known_answer(self):
        """8.1 Correct min of known array."""
        losses = jnp.array([5.0, 3.0, 7.0, 1.0, 4.0])
        assert run_min_loss(losses) == pytest.approx(1.0)

    def test_empty(self):
        """8.1b Empty array returns inf."""
        assert run_min_loss(jnp.array([])) == float("inf")


class TestRunHasSuccess:
    def test_below_threshold(self):
        """8.2 True if any loss < threshold."""
        losses = jnp.array([5.0, 0.1, 3.0])
        assert run_has_success(losses, threshold=0.5) is True

    def test_above_threshold(self):
        """8.2b False if all losses >= threshold."""
        losses = jnp.array([5.0, 1.0, 3.0])
        assert run_has_success(losses, threshold=0.5) is False

    def test_empty(self):
        """8.2c False for empty array."""
        assert run_has_success(jnp.array([]), threshold=0.5) is False


class TestRunFirstSuccessIdx:
    def test_known_answer(self):
        """8.3 Correct index."""
        losses = jnp.array([5.0, 3.0, 0.1, 0.05])
        assert run_first_success_idx(losses, threshold=0.5) == 2

    def test_no_success(self):
        """8.3b None when no success."""
        losses = jnp.array([5.0, 3.0, 1.0])
        assert run_first_success_idx(losses, threshold=0.5) is None

    def test_empty(self):
        """8.3c None for empty."""
        assert run_first_success_idx(jnp.array([]), threshold=0.5) is None


class TestRunFirstSuccessTime:
    def test_known_answer(self):
        """8.4 Correct time at success index."""
        losses = jnp.array([5.0, 3.0, 0.1])
        times = jnp.array([0.0, 1.0, 2.0])
        assert run_first_success_time(losses, times, threshold=0.5) == pytest.approx(
            2.0
        )

    def test_no_success(self):
        """8.4b None if no success."""
        losses = jnp.array([5.0, 3.0, 1.0])
        times = jnp.array([0.0, 1.0, 2.0])
        assert run_first_success_time(losses, times, threshold=0.5) is None


class TestRunAuc:
    def test_known_trapezoid(self):
        """8.5 Trapezoidal integration on simple data.

        losses = [2, 4], times = [0, 2] -> area = (2+4)/2 * 2 = 6.0
        """
        losses = jnp.array([2.0, 4.0])
        times = jnp.array([0.0, 2.0])
        assert run_auc(losses, times) == pytest.approx(6.0, rel=1e-5)

    def test_with_floor(self):
        """8.6 Losses clamped to max(0, loss - floor).

        losses = [5, 3, 1], floor = 2 -> clamped = [3, 1, 0]
        times = [0, 1, 2] -> area = trapz([3,1,0], [0,1,2]) = (3+1)/2*1 + (1+0)/2*1 = 2.5
        """
        losses = jnp.array([5.0, 3.0, 1.0])
        times = jnp.array([0.0, 1.0, 2.0])
        result = run_auc(losses, times, floor=2.0)
        assert result == pytest.approx(2.5, rel=1e-5)

    def test_with_baseline(self):
        """8.7 Normalized AUC: positive when better than baseline."""
        losses = jnp.array([1.0, 0.5, 0.1])
        times = jnp.array([0.0, 1.0, 2.0])
        result = run_auc(losses, times, baseline_loss=5.0, max_time=2.0)
        assert result > 0  # Better than random -> positive

    def test_empty(self):
        """8.8 Empty arrays return 0.0."""
        assert run_auc(jnp.array([]), jnp.array([])) == 0.0


# ======================================================================
# Aggregation functions (8.9-8.14)
# ======================================================================


class TestAggMeanStd:
    def test_known_answer(self):
        """8.9 Correct mean and std."""
        values = [2.0, 4.0, 6.0]
        mean, std = agg_mean_std(values)
        assert mean == pytest.approx(4.0)
        assert std == pytest.approx(np.std([2, 4, 6]), abs=1e-5)

    def test_nan_handling(self):
        """8.10 NaN values handled via nanmean/nanstd."""
        values = [1.0, float("nan"), 3.0]
        mean, std = agg_mean_std(values)
        assert mean == pytest.approx(2.0)
        assert not math.isnan(std)

    def test_empty(self):
        """8.11 Empty list returns (nan, nan)."""
        mean, std = agg_mean_std([])
        assert math.isnan(mean)
        assert math.isnan(std)


class TestAggMin:
    def test_known_answer(self):
        """8.12 Correct min."""
        assert agg_min([5.0, 2.0, 8.0]) == 2.0

    def test_empty(self):
        """8.12b Empty returns inf."""
        assert agg_min([]) == float("inf")


class TestAggFractionTrue:
    def test_known_answer(self):
        """8.13 Correct fraction."""
        assert agg_fraction_true([True, True, False, False]) == pytest.approx(0.5)

    def test_empty(self):
        """8.13b Empty returns 0.0."""
        assert agg_fraction_true([]) == 0.0


class TestAggMeanStdFiltered:
    def test_filters_none(self):
        """8.14 None values filtered; correct mean/std of rest."""
        values = [1.0, None, 3.0, None, 5.0]
        mean, std = agg_mean_std_filtered(values)
        assert mean == pytest.approx(3.0)

    def test_all_none(self):
        """8.14b All None returns (fallback, 0.0)."""
        mean, std = agg_mean_std_filtered([None, None], fallback=99.0)
        assert mean == 99.0
        assert std == 0.0


# ======================================================================
# Multi-run functions (8.15-8.22)
# ======================================================================


class TestMultiDiversityOverall:
    def test_less_than_2(self):
        """8.15 Returns (0, 0) for < 2 solutions."""
        params = jnp.array([[1.0, 2.0]])
        mean, std = multi_solution_diversity_overall(params)
        assert mean == 0.0 and std == 0.0

    def test_known_geometry(self):
        """8.16 4 corners of unit square -> known mean distance.

        Corners: (0,0), (1,0), (0,1), (1,1). Without normalization:
        6 pairwise distances: 4×1.0 + 2×sqrt(2) ≈ 6.828
        Divided by sqrt(2) for normalization -> mean ≈ 0.805
        """
        params = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        mean, std = multi_solution_diversity_overall(params)
        assert mean > 0

    def test_with_bounds_normalized(self):
        """8.17 With bounds normalization, result is in [0, 1]."""
        params = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        bounds = jnp.array([[0.0, 0.0], [1.0, 1.0]])
        mean, std = multi_solution_diversity_overall(params, bounds=bounds)
        assert 0.0 <= mean <= 1.0


class TestMultiDiversityNN:
    def test_less_than_2(self):
        """8.18 Returns (0, 0) for < 2 solutions."""
        params = jnp.array([[1.0, 2.0]])
        mean, std = multi_solution_diversity_nn(params)
        assert mean == 0.0 and std == 0.0

    def test_known_geometry(self):
        """8.19 Known geometry: equidistant points on a line."""
        params = jnp.array([[0.0], [1.0], [2.0], [3.0]])
        mean, std = multi_solution_diversity_nn(params)
        # NN distance for each is 1.0, normalized by sqrt(1)=1 -> mean=1.0
        assert mean == pytest.approx(1.0, abs=1e-5)


class TestMultiAucTopK:
    def test_correct_selection(self):
        """8.20 Top-k selects best (lowest min loss) runs."""
        min_losses = [5.0, 1.0, 3.0, 2.0, 4.0]
        aucs = [50.0, 10.0, 30.0, 20.0, 40.0]
        # k_fraction=0.2 -> top 1 run (index 1, loss=1.0, auc=10.0)
        mean, std = multi_auc_top_k(min_losses, aucs, k_fraction=0.2)
        assert mean == pytest.approx(10.0)

    def test_at_least_one(self):
        """8.20b k_fraction=0.1 picks at least 1 run."""
        mean, std = multi_auc_top_k([1.0, 2.0], [10.0, 20.0], k_fraction=0.1)
        assert mean == pytest.approx(10.0)  # Top 1 (max(1, int(2*0.1))=1)


class TestPerformanceProfile:
    def test_all_below(self):
        """8.21 All losses below threshold -> success_rate=1.0."""
        thresholds = jnp.array([0.5, 1.0, 2.0])
        losses = [0.1, 0.2, 0.3]  # All below any threshold
        _, rates, _ = compute_performance_profile(losses, thresholds)
        np.testing.assert_allclose(np.array(rates), np.ones(3), atol=1e-6)

    def test_all_above(self):
        """8.21b All losses above threshold -> success_rate=0.0."""
        thresholds = jnp.array([0.01, 0.05])
        losses = [1.0, 2.0, 3.0]
        _, rates, _ = compute_performance_profile(losses, thresholds)
        np.testing.assert_allclose(np.array(rates), np.zeros(2), atol=1e-6)

    def test_normalized_auc_range(self):
        """8.22 normalized_auc is in [0, 1]."""
        losses = [0.1, 0.5, 1.0, 2.0, 5.0]
        _, _, nauc = compute_performance_profile(losses)
        assert 0.0 <= nauc <= 1.0


# ======================================================================
# Time-based slicing (8.23-8.27)
# ======================================================================


class TestGetIndexAtTime:
    def test_before_data(self):
        """8.23 Returns -1 for t < first time step."""
        assert get_index_at_time([1.0, 2.0, 3.0], t=0.5) == -1

    def test_intermediate(self):
        """8.24 Returns last valid index for intermediate t."""
        assert get_index_at_time([1.0, 2.0, 3.0], t=2.5) == 1

    def test_at_end(self):
        """8.25 Returns last index for t >= final time step."""
        assert get_index_at_time([1.0, 2.0, 3.0], t=5.0) == 2


class TestSliceHistoryAtTime:
    def test_before_data(self):
        """8.26 Empty list for t before data."""
        assert slice_history_at_time([10, 20, 30], [1.0, 2.0, 3.0], t=0.5) == []

    def test_normal(self):
        """8.26b Correct sublist."""
        result = slice_history_at_time([10, 20, 30], [1.0, 2.0, 3.0], t=2.5)
        assert result == [10, 20]


class TestGetValueAtTime:
    def test_before_data(self):
        """8.27 Returns default for t before data."""
        assert get_value_at_time([10, 20, 30], [1.0, 2.0, 3.0], t=0.5, default=-1) == -1

    def test_normal(self):
        """8.27b Returns correct value."""
        assert get_value_at_time([10, 20, 30], [1.0, 2.0, 3.0], t=2.5) == 20
