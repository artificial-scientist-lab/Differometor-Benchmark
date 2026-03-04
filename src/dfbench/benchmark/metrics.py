"""Metric computation functions for benchmarking.

This module contains all metric computation functions, separated from the main
Benchmark class for clarity. Functions are organized into three categories:

1. Per-run functions (_run_*): Compute values for a single run's data
2. Aggregation functions (_agg_*): Combine per-run results into statistics
3. Multi-run functions (_multi_*): Require data from all runs (e.g., diversity)
"""

import jax.numpy as jnp
from jaxtyping import Array, Float


# --------- Per-run functions ---------
# These compute values for a single run's data (1D array of losses/params)
# Used in list comprehensions to process each run independently


def run_min_loss(losses: Float[Array, "iterations"]) -> float:
    """Minimum loss achieved in a single run."""
    if len(losses) == 0:
        return float("inf")
    return float(jnp.nanmin(losses))


def run_has_success(losses: Float[Array, "iterations"], threshold: float) -> bool:
    """Whether a single run achieved success (any loss below threshold)."""
    if len(losses) == 0:
        return False
    return bool(jnp.any(losses < threshold))


def run_first_success_idx(
    losses: Float[Array, "iterations"], threshold: float
) -> int | None:
    """Iteration index of first success, or None if no success."""
    if len(losses) == 0:
        return None
    mask = losses < threshold
    if jnp.any(mask):
        return int(jnp.argmax(mask))
    return None


def run_first_success_time(
    losses: Float[Array, "iterations"],
    time_steps: Float[Array, "iterations"],
    threshold: float,
) -> float | None:
    """Time of first success, or None if no success."""
    idx = run_first_success_idx(losses, threshold)
    if idx is None:
        return None
    return float(time_steps[idx])


def run_auc(
    losses: Float[Array, "iterations"],
    time_steps: Float[Array, "iterations"],
    floor: float | None = None,
    baseline_loss: float | None = None,
    max_time: float | None = None,
) -> float:
    """Area under the loss curve for a single run, optionally normalized.

    Args:
        losses: Loss values per iteration
        time_steps: Time at each iteration
        floor: Optional floor value (e.g., success_loss). Losses clamped to max(0, loss - floor)
        baseline_loss: Expected loss of random guess. If provided with max_time,
            returns normalized AUC: -log2(algorithm_auc / baseline_auc)
        max_time: Total time budget. Required if baseline_loss is provided.

    Returns:
        Area under the (possibly clamped) loss curve, optionally normalized.
        If normalized: positive = better than random, negative = worse, zero = equal.
    """
    if len(losses) == 0 or len(time_steps) == 0:
        return 0.0

    losses = jnp.asarray(losses)
    time_steps = jnp.asarray(time_steps)

    if floor is not None:
        losses = jnp.maximum(0, losses - floor)

    trapezoid_result = jnp.trapezoid(losses, time_steps)
    # Handle case where trapezoid returns an array instead of scalar
    if trapezoid_result.ndim > 0:
        if trapezoid_result.size == 1:
            trapezoid_result = trapezoid_result.item()
        else:
            # If multiple values, take the sum (integrate over all dimensions)
            trapezoid_result = float(jnp.sum(trapezoid_result))
    algorithm_auc = float(trapezoid_result)

    # Normalize by baseline if provided
    if baseline_loss is not None and max_time is not None:
        if floor is not None:
            baseline_loss = max(0, baseline_loss - floor)
        baseline_auc = baseline_loss * max_time
        if algorithm_auc <= 0:
            return float("inf")  # Perfect performance
        ratio = algorithm_auc / baseline_auc
        return -float(jnp.log2(ratio))

    return algorithm_auc


# --------- Aggregation functions ---------
# These combine per-run results into benchmark metrics


def agg_mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean and std from a list of per-run values.

    Uses nanmean/nanstd to handle NaN values gracefully.
    """
    if len(values) == 0:
        return float("nan"), float("nan")
    arr = jnp.array(values)
    return float(jnp.nanmean(arr)), float(jnp.nanstd(arr))


def agg_min(values: list[float]) -> float:
    """Global minimum across all runs."""
    if len(values) == 0:
        return float("inf")
    return float(min(values))


def agg_fraction_true(values: list[bool]) -> float:
    """Fraction of True values (e.g., fraction of successful runs)."""
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


def agg_mean_std_filtered(
    values: list[float | None], fallback: float = float("nan")
) -> tuple[float, float]:
    """Mean and std of non-None values. Returns (fallback, 0.0) if all None."""
    filtered = [v for v in values if v is not None]
    if filtered:
        arr = jnp.array(filtered)
        return float(jnp.mean(arr)), float(jnp.std(arr))
    return fallback, 0.0


# --------- Multi-run functions ---------
# These inherently need data from all runs (e.g., diversity, top-k selection)


def multi_solution_diversity_overall(
    params: Float[Array, "n_solutions n_params"],
    bounds: Float[Array, "2 n_params"] | None = None,
) -> tuple[float, float]:
    """Overall diversity as mean pairwise distance, normalized to [0, 1].

    Each dimension is normalized to [0, 1] by bounds, then the Euclidean
    distance is divided by sqrt(n_params) so max distance = 1.

    Args:
        params: Solution parameters, shape (n_solutions, n_params)
        bounds: Parameter bounds [lower, upper], shape (2, n_params).
            If None, no normalization is applied.

    Returns:
        (mean, std) of normalized pairwise distances in [0, 1], or (0.0, 0.0) if < 2 solutions.
    """
    if params.shape[0] < 2:
        return 0.0, 0.0

    n_solutions = params.shape[0]
    n_params = params.shape[1]

    # Normalize parameters by bounds if provided
    if bounds is not None:
        bounds = jnp.asarray(bounds)
        param_ranges = bounds[1] - bounds[0]
        param_ranges = jnp.where(param_ranges > 0, param_ranges, 1.0)
        params_normalized = (params - bounds[0]) / param_ranges
    else:
        params_normalized = params

    diff = params_normalized[:, None, :] - params_normalized[None, :, :]
    distances = jnp.linalg.norm(diff, axis=2) / jnp.sqrt(n_params)

    mask = ~jnp.eye(n_solutions, dtype=bool)
    pairwise_distances = distances[mask]

    return float(jnp.mean(pairwise_distances)), float(jnp.std(pairwise_distances))


def multi_solution_diversity_nn(
    params: Float[Array, "n_solutions n_params"],
    bounds: Float[Array, "2 n_params"] | None = None,
) -> tuple[float, float]:
    """Nearest-neighbor diversity, normalized to [0, 1].

    Args:
        params: Solution parameters, shape (n_solutions, n_params)
        bounds: Parameter bounds [lower, upper], shape (2, n_params).

    Returns:
        (mean, std) of normalized NN distances in [0, 1], or (0.0, 0.0) if < 2 solutions.
    """
    if params.shape[0] < 2:
        return 0.0, 0.0

    n_solutions = params.shape[0]
    n_params = params.shape[1]

    if bounds is not None:
        bounds = jnp.asarray(bounds)
        param_ranges = bounds[1] - bounds[0]
        param_ranges = jnp.where(param_ranges > 0, param_ranges, 1.0)
        params_normalized = (params - bounds[0]) / param_ranges
    else:
        params_normalized = params

    diff = params_normalized[:, None, :] - params_normalized[None, :, :]
    distances = jnp.linalg.norm(diff, axis=2) / jnp.sqrt(n_params)

    distances_no_diag = jnp.where(jnp.eye(n_solutions, dtype=bool), jnp.inf, distances)
    nearest_neighbor_distances = jnp.min(distances_no_diag, axis=1)

    return float(jnp.mean(nearest_neighbor_distances)), float(
        jnp.std(nearest_neighbor_distances)
    )


def multi_auc_top_k(
    run_min_losses: list[float],
    run_aucs: list[float],
    k_fraction: float = 0.1,
) -> tuple[float, float]:
    """AUC statistics for top k% of runs (by final min loss).

    Args:
        run_min_losses: Min loss achieved by each run
        run_aucs: AUC for each run
        k_fraction: Fraction of runs to consider as "top" (default 10%)

    Returns:
        (mean, std) of AUC for top k% runs
    """
    n_runs = len(run_min_losses)
    if n_runs == 0:
        return float("nan"), float("nan")

    n_top = max(1, int(n_runs * k_fraction))

    sorted_indices = sorted(range(n_runs), key=lambda i: run_min_losses[i])
    top_indices = sorted_indices[:n_top]

    top_aucs = [run_aucs[i] for i in top_indices]
    return agg_mean_std(top_aucs)


def compute_performance_profile(
    run_min_losses: list[float],
    loss_thresholds: Float[Array, "n_thresholds"] | None = None,
) -> tuple[Float[Array, "n_thresholds"], Float[Array, "n_thresholds"], float]:
    """Compute performance profile (empirical CDF of final losses).

    This is similar to an ROC curve: for each loss threshold, compute the
    fraction of runs that achieved a loss below that threshold.

    Args:
        run_min_losses: Minimum loss achieved by each run
        loss_thresholds: Array of loss thresholds to evaluate at.
            If None, defaults to linspace from -1 to 5 with 601 points.

    Returns:
        tuple of (thresholds, success_rates, normalized_auc):
            - thresholds: Loss threshold values
            - success_rates: Fraction of runs achieving loss < threshold (0 to 1)
            - normalized_auc: Area under the curve, normalized by threshold range
    """
    if loss_thresholds is None:
        loss_thresholds = jnp.linspace(-1.0, 5.0, 601)

    losses_array = jnp.array(run_min_losses)
    n_runs = len(run_min_losses)

    if n_runs == 0:
        return loss_thresholds, jnp.zeros_like(loss_thresholds), 0.0

    below_threshold = losses_array[:, None] < loss_thresholds[None, :]
    success_rates = jnp.mean(below_threshold, axis=0)

    threshold_range = float(loss_thresholds[-1] - loss_thresholds[0])
    if threshold_range > 0:
        raw_auc = float(jnp.trapezoid(success_rates, loss_thresholds))
        normalized_auc = raw_auc / threshold_range
    else:
        normalized_auc = 0.0

    return loss_thresholds, success_rates, normalized_auc


# --------- Time-based slicing utilities ---------


def get_index_at_time(time_steps: list[float] | Float[Array, "n"], t: float) -> int:
    """Get the last index where time_steps[i] <= t.

    Args:
        time_steps: Sorted array of timestamps
        t: Target time

    Returns:
        Index of last evaluation at or before time t. Returns -1 if t < time_steps[0].
    """
    time_steps = jnp.asarray(time_steps)
    if len(time_steps) == 0 or t < time_steps[0]:
        return -1

    # Find last index where time_steps <= t
    mask = time_steps <= t
    if not jnp.any(mask):
        return -1
    return int(jnp.sum(mask) - 1)


def slice_history_at_time(
    history: list,
    time_steps: list[float],
    t: float,
) -> list:
    """Get history entries up to and including time t.

    Args:
        history: List of values (losses, params, etc.)
        time_steps: Corresponding timestamps
        t: Target time

    Returns:
        Sublist of history up to time t (inclusive).
    """
    idx = get_index_at_time(time_steps, t)
    if idx < 0:
        return []
    return history[: idx + 1]


def get_value_at_time(
    history: list,
    time_steps: list[float],
    t: float,
    default=None,
):
    """Get history value at time t (last value where time_steps <= t).

    Args:
        history: List of values
        time_steps: Corresponding timestamps
        t: Target time
        default: Value to return if no data at or before time t

    Returns:
        Value at time t, or default if not available.
    """
    idx = get_index_at_time(time_steps, t)
    if idx < 0:
        return default
    return history[idx]
