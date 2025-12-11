"""Benchmarking module for optimization algorithms.

This module provides tools to benchmark multiple optimization algorithms on a problem,
running each algorithm multiple times and computing various performance metrics.

Key classes:
    - Benchmark: Main benchmarking class that runs algorithms and evaluates performance
    - BenchmarkResult: Dataclass containing performance metrics across time steps
    - AlgorithmConfig: Configuration wrapper for algorithm + hyperparameters

Compatibility:
    - Works with any algorithm implementing OptimizationAlgorithm protocol
    - Requires algorithms to return: (best_params, best_params_history, losses, ...)
    - Requires return_best_params_history=True and wall_time parameters
"""

import numpy as np
import jax.numpy as jnp
import csv
import json
from pathlib import Path
from datetime import datetime
from jaxtyping import Array, Float
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, fields

from dfbench import OptimizationAlgorithm, ContinuousProblem, AlgorithmType


@dataclass
class SingleMetric:
    """Metric with a single value per time step (not aggregated across runs)."""

    value: Float[Array, "n_time_steps"]


@dataclass
class AggregateMetric:
    """Metric aggregated across multiple runs with statistics."""

    mean: Float[Array, "n_time_steps"]
    std: Float[Array, "n_time_steps"]


# =============================================================================
#                             Helper functions
# =============================================================================
# Architecture:
#   _run_*   : Per-run computations (operate on single 1D loss/param array)
#   _agg_*   : Aggregation across runs (combine per-run values into metrics)
#   _multi_* : Multi-run computations (inherently need all runs, e.g., diversity)
#
# The evaluation loop:
#   1. For each time step, get per-run indices from wall_time_indices
#   2. Compute per-run values using _run_* functions in list comprehensions
#   3. Aggregate into benchmark metrics using _agg_* functions
#   4. For special metrics (diversity, top-k), use _multi_* functions
#
# DESIGN NOTE: Alternative Vectorized Masking Approach
# ----------------------------------------------------
# We considered padding all runs to max_iterations and using boolean masks:
#   mask = iter_range[None, :] < wall_time_indices[:, t, None]
#   masked_losses = jnp.where(mask, all_losses, jnp.inf)
#
# Performance Analysis:
# - Both approaches are essentially equivalent for our use case
# - The Python loop is over runs (~100), not iterations (~100k)
# - Heavy compute (reductions) is already vectorized JAX in both approaches
#
# We chose list iteration because:
# - Simpler to reason about, fewer edge cases with neutral values
# - No memory overhead from padding (important if iteration counts vary widely)
# - Code clarity: explicit per-run logic is easier to debug
#
# The matrix approach would be preferred if:
# - Evaluation needs to be JIT-compiled (e.g., inside an optimization loop)
# - Running on GPU where kernel launch overhead dominates
# =============================================================================


# =============================================================================
# PER-RUN FUNCTIONS (_run_*)
# =============================================================================
# These compute values for a single run's data (1D array of losses/params).
# Used in list comprehensions to process each run independently.
# =============================================================================


def _run_min_loss(losses: Float[Array, "iterations"]) -> float:
    """Minimum loss achieved in a single run."""
    return float(jnp.min(losses))


def _run_has_success(losses: Float[Array, "iterations"], threshold: float) -> bool:
    """Whether a single run achieved success (any loss below threshold)."""
    return bool(jnp.any(losses < threshold))


def _run_first_success_iter(
    losses: Float[Array, "iterations"], threshold: float
) -> int | None:
    """Iteration index of first success, or None if no success."""
    mask = losses < threshold
    if jnp.any(mask):
        return int(jnp.argmax(mask))
    return None


def _run_auc(
    losses: Float[Array, "iterations"],
    dx: float,
    floor: float | None = None,
    baseline_loss: float | None = None,
    wall_time: float | None = None,
) -> float:
    """Area under the loss curve for a single run, normalized by the random-search baseline.

    Args:
        losses: Loss values per iteration
        dx: Time step between iterations
        floor: Optional floor value (e.g., success_loss). Losses are calculated as max(0, loss - floor)
        baseline_loss: Expected loss of random guess. If provided with wall_time,
            returns normalized AUC: -log(algorithm_auc / baseline_auc)
        wall_time: Total wall time budget. Required if baseline_loss is provided.

    Returns:
        Area under the (possibly clamped) loss curve, optionally normalized by baseline.
        If normalized: positive values mean better than random (lower AUC ratio),
        negative means worse (higher AUC ratio). Zero means equal to random.
        Formula: -log(algorithm_auc / baseline_auc) = log(baseline_auc / algorithm_auc)
    """
    if floor is not None:
        losses = jnp.maximum(0, losses - floor)

    algorithm_auc = float(jnp.trapezoid(losses, dx=dx))

    # Normalize by baseline if provided
    if baseline_loss is not None and wall_time is not None:
        baseline_auc = baseline_loss * wall_time
        # -log(ratio): positive when algorithm_auc < baseline_auc (better than random)
        # Avoid division by zero or log of zero
        if algorithm_auc <= 0:
            return float("inf")  # Perfect performance
        ratio = algorithm_auc / baseline_auc
        return -float(jnp.log2(ratio))

    return algorithm_auc


# =============================================================================
# AGGREGATION FUNCTIONS (_agg_*)
# =============================================================================
# These combine per-run results into benchmark metrics (the actual numbers
# that characterize algorithm performance across many runs).
# =============================================================================


def _agg_mean_std(values: list[float]) -> tuple[float, float]:
    """Compute mean and std from a list of per-run values.

    Uses nanmean/nanstd to handle NaN values gracefully.
    """
    arr = jnp.array(values)
    return float(jnp.nanmean(arr)), float(jnp.nanstd(arr))


def _agg_min(values: list[float]) -> float:
    """Global minimum across all runs."""
    return float(min(values))


def _agg_fraction_true(values: list[bool]) -> float:
    """Fraction of True values (e.g., fraction of successful runs)."""
    return sum(values) / len(values) if values else 0.0


def _agg_mean_std_filtered(
    values: list[float | None], fallback: float
) -> tuple[float, float]:
    """Mean and std of non-None values. Returns (fallback, 0.0) if all None."""
    filtered = [v for v in values if v is not None]
    if filtered:
        arr = jnp.array(filtered)
        return float(jnp.mean(arr)), float(jnp.std(arr))
    return fallback, 0.0


# =============================================================================
# MULTI-RUN FUNCTIONS
# =============================================================================
# These inherently need data from all runs (e.g., diversity, top-k selection).
# They operate on collections of solutions, not individual runs.
# =============================================================================


def _multi_solution_diversity_overall(
    params: Float[Array, "n_solutions n_params"],
) -> tuple[float, float]:
    """Overall diversity as mean pairwise Euclidean distance (mean ± std).

    Returns (0.0, 0.0) if fewer than 2 solutions.
    """
    n_solutions = params.shape[0]
    if n_solutions < 2:
        return 0.0, 0.0

    diff = params[:, None, :] - params[None, :, :]
    distances = jnp.linalg.norm(diff, axis=2)

    mask = ~jnp.eye(n_solutions, dtype=bool)
    pairwise_distances = distances[mask]

    return float(jnp.mean(pairwise_distances)), float(jnp.std(pairwise_distances))


def _multi_solution_diversity_nn(
    params: Float[Array, "n_solutions n_params"],
) -> tuple[float, float]:
    """Nearest-neighbor diversity (mean ± std of distances to nearest neighbor).

    Returns (0.0, 0.0) if fewer than 2 solutions.
    """
    n_solutions = params.shape[0]
    if n_solutions < 2:
        return 0.0, 0.0

    diff = params[:, None, :] - params[None, :, :]
    distances = jnp.linalg.norm(diff, axis=2)

    distances_no_diag = jnp.where(jnp.eye(n_solutions, dtype=bool), jnp.inf, distances)
    nearest_neighbor_distances = jnp.min(distances_no_diag, axis=1)

    return float(jnp.mean(nearest_neighbor_distances)), float(
        jnp.std(nearest_neighbor_distances)
    )


def _multi_auc_top_k(
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
    n_top = max(1, int(n_runs * k_fraction))

    # Sort by min loss, get top indices
    sorted_indices = sorted(range(n_runs), key=lambda i: run_min_losses[i])
    top_indices = sorted_indices[:n_top]

    top_aucs = [run_aucs[i] for i in top_indices]
    return _agg_mean_std(top_aucs)


@dataclass
class BenchmarkResult:
    """Results from a benchmark run using one algorithm across multiple time steps.

    All metric arrays have shape (n_time_steps,) corresponding to self.wall_time_steps.

    Metrics are categorized by type:
    - SingleMetric: Properties of the distribution (single value, not aggregated)
    - AggregateMetric: Statistics computed across runs (mean, std, optionally median/min/max)
    """

    algorithm_name: str

    # Single-value metrics (distribution properties)
    fraction_of_success: SingleMetric
    min_loss: SingleMetric

    # Aggregate metrics (statistics across runs)
    avg_loss: AggregateMetric
    time_to_success: AggregateMetric
    calls_to_success: AggregateMetric
    solution_diversity_overall: AggregateMetric
    solution_diversity_nn: AggregateMetric

    time_per_call: AggregateMetric

    # Top-k metrics (normalized AUC using -log ratio)
    # Formula: -log(algorithm_auc / baseline_auc)
    # Positive = better than random, Negative = worse, Zero = equal
    auc_top_1: SingleMetric
    auc_top_10: AggregateMetric


@dataclass
class AlgorithmRunData:
    """Raw run data for a single algorithm configuration.

    This dataclass stores all the raw data from running an algorithm multiple times,
    allowing results to be saved and re-evaluated later with different parameters
    (e.g., different success_loss thresholds).

    Attributes:
        algorithm_name: Name of the algorithm configuration
        calls_per_iteration: Number of objective function calls per iteration
        all_best_params: Best parameters found by each run, shape (n_runs, n_params)
        all_best_params_history: List of arrays, each (n_iterations, n_params)
        all_losses: List of arrays, each (n_iterations,)
        all_wall_time_indices: List of lists, each containing indices for wall_time checkpoints
    """

    algorithm_name: str
    calls_per_iteration: int
    all_best_params: Float[Array, "runs params"]
    all_best_params_history: List[Float[Array, "iterations params"]]
    all_losses: List[Float[Array, "iterations"]]
    all_wall_time_indices: List[List[int]]


class AlgorithmConfig:
    """Configuration for an algorithm to benchmark."""

    # Common hyperparameter names that indicate calls per iteration
    _CALLS_PER_ITER_KEYS = ["pop_size", "population_size", "n_particles"]

    def __init__(
        self,
        algorithm: OptimizationAlgorithm,
        hyperparameters: Dict[str, Any],
        name: Optional[str] = None,
        calls_per_iteration: Optional[int] = None,
    ):
        """
        Args:
            algorithm: The algorithm instance to benchmark
            hyperparameters: Dictionary of hyperparameters to pass to optimize()
            name: Optional custom name for this configuration
            calls_per_iteration: Number of objective function calls per iteration.
                If None, auto-detected from hyperparameters (e.g., pop_size) or
                inferred from algorithm type. Explicit value takes precedence.
        """
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters
        self.name = name or algorithm.algorithm_str
        self._calls_per_iteration = (
            calls_per_iteration or self._infer_calls_per_iteration()
        )

    def _infer_calls_per_iteration(self) -> int:
        """Infer calls per iteration from hyperparameters or algorithm type.

        Priority:
        1. Check common hyperparameter names (pop_size, population_size, etc.)
        2. Fall back to algorithm type (evolutionary -> 100, gradient -> 1)
        3. Default to 1 if unknown

        Returns:
            Inferred number of objective function calls per iteration
        """
        # Check common hyperparameter names
        for key in self._CALLS_PER_ITER_KEYS:
            if key in self.hyperparameters:
                return self.hyperparameters[key]

        # Fall back to algorithm type
        try:
            if self.algorithm.algorithm_type == AlgorithmType.EVOLUTIONARY:
                return 100  # Common default population size
            elif self.algorithm.algorithm_type == AlgorithmType.GRADIENT_BASED:
                return 1
        except AttributeError:
            pass  # Algorithm doesn't have algorithm_type

        return 1  # Default: 1 call per iteration

    @property
    def calls_per_iteration(self) -> int:
        """Number of objective function calls per iteration."""
        return self._calls_per_iteration


class Benchmark:
    """Benchmark multiple optimization algorithms on a problem.

    Runs multiple algorithms with different hyperparameter configurations,
    evaluates their performance across multiple runs, and computes various
    metrics including success rate, convergence time, and solution diversity.
    """

    def __init__(
        self,
        problem: ContinuousProblem,
        success_loss: float,
        configs: list[AlgorithmConfig],
        n_runs: int = 100,
        wall_time_steps: list[float] = [300],  # 5 minutes in seconds
        random_baseline_loss: float = 463.5,
    ):
        """Initialize the benchmark suite.

        Args:
            problem: The optimization problem to benchmark algorithms on
            success_loss: Loss threshold below which a run is considered successful
            configs: List of algorithm configurations to benchmark
            n_runs: Number of independent runs per algorithm configuration
            wall_time_steps: Time checkpoints (in seconds) at which to evaluate metrics
            random_baseline_loss: Expected loss of a random guess (default: 463.7 for Voyager).
        """
        self._problem = problem
        self._success_loss = success_loss
        self._configs = configs
        self._n_runs = n_runs
        self._wall_time_steps = sorted(wall_time_steps)
        self._random_baseline_loss = random_baseline_loss
        # Warmup jit compilation
        _ = problem.objective_function(jnp.zeros(problem.n_params))

    def _collect_run_data(self, config: AlgorithmConfig) -> AlgorithmRunData:
        """Run algorithm multiple times and collect raw data.

        Args:
            config: Algorithm configuration

        Returns:
            AlgorithmRunData containing all raw run results
        """
        print(f"\n{'=' * 70}")
        print(f"Running: {config.name}")
        print(f"Hyperparameters: {config.hyperparameters}")
        print(f"Calls per iteration: {config.calls_per_iteration}")
        print(f"Number of runs: {self._n_runs}")
        print(f"Wall time limit: {self._wall_time_steps[-1]:.1f}s")
        print(f"{'=' * 70}\n")

        all_best_params = []
        all_best_params_history = []
        all_losses = []
        all_wall_time_indices = []

        for i_run in range(self._n_runs):
            print(f"Run {i_run + 1}/{self._n_runs}...", end=" ", flush=True)

            # Run algorithm with wall_times
            best_params, best_params_history, losses, wall_time_indices, *_ = (
                config.algorithm.optimize(
                    save_to_file=False,
                    return_best_params_history=True,
                    wall_times=self._wall_time_steps,
                    **config.hyperparameters,
                )
            )

            # Store results (keep as lists, no padding)
            all_best_params.append(best_params)
            all_best_params_history.append(best_params_history)
            all_losses.append(losses)
            all_wall_time_indices.append(wall_time_indices)

            # Print progress
            final_loss = losses[-1] if len(losses) > 0 else float("inf")
            print(f"Final loss: {final_loss:.6f}")

        print(f"\nCompleted {self._n_runs} runs for {config.name}")

        return AlgorithmRunData(
            algorithm_name=config.name,
            calls_per_iteration=config.calls_per_iteration,
            all_best_params=jnp.array(all_best_params),
            all_best_params_history=all_best_params_history,
            all_losses=all_losses,
            all_wall_time_indices=all_wall_time_indices,
        )

    def _evaluate_run_data(self, run_data: AlgorithmRunData) -> BenchmarkResult:
        """Evaluate an AlgorithmRunData and return BenchmarkResult.

        Args:
            run_data: Raw run data for one algorithm

        Returns:
            BenchmarkResult with computed metrics
        """
        return self._evaluate_runs(
            run_data.algorithm_name,
            run_data.calls_per_iteration,
            run_data.all_best_params,
            run_data.all_best_params_history,
            run_data.all_losses,
            run_data.all_wall_time_indices,
        )

    def _evaluate_runs(
        self,
        algorithm_name: str,
        calls_per_iteration: int,
        all_best_params: Float[Array, "runs params"],
        all_best_params_history: list[Float[Array, "iterations params"]],
        all_losses: list[Float[Array, "iterations"]],
        all_wall_time_indices: list[list[int]],
    ) -> BenchmarkResult:
        """Evaluate an algorithm's runs using per-run functions and aggregation.

        Args:
            algorithm_name: Name of the algorithm
            calls_per_iteration: Number of objective function calls per iteration
            all_best_params: Best parameters found during each run
            all_best_params_history: History of best params per iteration (list of arrays)
            all_losses: Losses per iteration for each run (list of arrays)
            all_wall_time_indices: Per-run indices for each wall_time checkpoint

        Returns:
            BenchmarkResult containing all metrics as arrays
        """
        print("Evaluating runs...")

        # Result containers (lists, converted to arrays at the end)
        fraction_of_success_list = []
        min_loss_list = []
        avg_loss_list = []  # list of (mean, std) tuples
        time_to_success_list = []
        calls_to_success_list = []
        time_per_call_list = []
        auc_top_1_list = []
        auc_top_10_list = []
        diversity_overall_list = []
        diversity_nn_list = []

        # Get calls_per_iteration
        calls_per_iter = calls_per_iteration

        for i_time, wall_time in enumerate(self._wall_time_steps):
            # Get per-run iteration indices for this time step
            indices = [wti[i_time] for wti in all_wall_time_indices]

            # Per-run time_per_iter (each run may have different iteration counts)
            run_time_per_iter = [wall_time / idx if idx > 0 else 0.0 for idx in indices]

            # Average time_per_iter for metrics that need a single value (e.g., time_per_call)
            avg_iters = sum(indices) / len(indices) if indices else 1
            avg_time_per_iter = wall_time / avg_iters if avg_iters > 0 else 0.0

            # Time per objective function call (accounting for population/batch size)
            time_per_call = (
                avg_time_per_iter / calls_per_iter if calls_per_iter > 0 else 0.0
            )

            # === Per-run computations via list comprehensions ===
            run_min_losses = [
                _run_min_loss(losses[:idx]) for losses, idx in zip(all_losses, indices)
            ]
            run_has_success = [
                _run_has_success(losses[:idx], self._success_loss)
                for losses, idx in zip(all_losses, indices)
            ]
            run_first_success_iters = [
                _run_first_success_iter(losses[:idx], self._success_loss)
                for losses, idx in zip(all_losses, indices)
            ]
            # AUC normalized by random baseline (positive = better than random)
            # Floor at success_loss prevents negative contribution from "over-success"
            run_aucs = [
                _run_auc(
                    losses[:idx],
                    dx=tpi,
                    floor=self._success_loss,
                    baseline_loss=self._random_baseline_loss,
                    wall_time=wall_time,
                )
                for losses, idx, tpi in zip(all_losses, indices, run_time_per_iter)
            ]

            # === Aggregate into benchmark metrics ===

            # Fraction of success
            fraction_of_success_list.append(_agg_fraction_true(run_has_success))

            # Min loss (global minimum across all runs)
            min_loss_list.append(_agg_min(run_min_losses))

            # Average loss (mean ± std of per-run minimums)
            avg_loss_list.append(_agg_mean_std(run_min_losses))

            # Time to success using per-run time scale (fixes drift issue)
            success_times = [
                iter_idx * run_time_per_iter[i]
                for i, iter_idx in enumerate(run_first_success_iters)
                if iter_idx is not None
            ]
            if success_times:
                time_to_success_list.append(_agg_mean_std(success_times))
            else:
                # No successes: use NaN to indicate "not applicable"
                time_to_success_list.append((float("nan"), float("nan")))

            # Calls to success (only successful runs) - multiply iterations by calls_per_iter
            success_calls = [
                i * calls_per_iter for i in run_first_success_iters if i is not None
            ]
            if success_calls:
                calls_to_success_list.append(
                    _agg_mean_std([float(c) for c in success_calls])
                )
            else:
                # No successes: use NaN to indicate "not applicable"
                calls_to_success_list.append((float("nan"), float("nan")))

            # Time per call (in milliseconds)
            time_per_call_ms = time_per_call * 1000
            time_per_call_list.append((time_per_call_ms, 0.0))

            # AUC top 1 (best run by min loss)
            best_run_idx = run_min_losses.index(min(run_min_losses))
            auc_top_1_list.append(run_aucs[best_run_idx])

            # AUC top 10%
            auc_top_10_list.append(
                _multi_auc_top_k(run_min_losses, run_aucs, k_fraction=0.1)
            )

            # === Diversity (needs successful params) ===
            successful_params = jnp.array(
                [
                    all_best_params_history[i][idx - 1]
                    for i, (idx, success) in enumerate(zip(indices, run_has_success))
                    if success and idx > 0
                ]
            )

            if successful_params.shape[0] >= 2:
                diversity_overall_list.append(
                    _multi_solution_diversity_overall(successful_params)
                )
                diversity_nn_list.append(
                    _multi_solution_diversity_nn(successful_params)
                )
            else:
                diversity_overall_list.append((0.0, 0.0))
                diversity_nn_list.append((0.0, 0.0))

        # Convert lists to arrays for BenchmarkResult
        return BenchmarkResult(
            algorithm_name=algorithm_name,
            fraction_of_success=SingleMetric(value=jnp.array(fraction_of_success_list)),
            min_loss=SingleMetric(value=jnp.array(min_loss_list)),
            avg_loss=AggregateMetric(
                mean=jnp.array([m for m, _ in avg_loss_list]),
                std=jnp.array([s for _, s in avg_loss_list]),
            ),
            time_to_success=AggregateMetric(
                mean=jnp.array([m for m, _ in time_to_success_list]),
                std=jnp.array([s for _, s in time_to_success_list]),
            ),
            calls_to_success=AggregateMetric(
                mean=jnp.array([m for m, _ in calls_to_success_list]),
                std=jnp.array([s for _, s in calls_to_success_list]),
            ),
            solution_diversity_overall=AggregateMetric(
                mean=jnp.array([m for m, _ in diversity_overall_list]),
                std=jnp.array([s for _, s in diversity_overall_list]),
            ),
            solution_diversity_nn=AggregateMetric(
                mean=jnp.array([m for m, _ in diversity_nn_list]),
                std=jnp.array([s for _, s in diversity_nn_list]),
            ),
            time_per_call=AggregateMetric(
                mean=jnp.array([m for m, _ in time_per_call_list]),
                std=jnp.array([s for _, s in time_per_call_list]),
            ),
            auc_top_1=SingleMetric(value=jnp.array(auc_top_1_list)),
            auc_top_10=AggregateMetric(
                mean=jnp.array([m for m, _ in auc_top_10_list]),
                std=jnp.array([s for _, s in auc_top_10_list]),
            ),
        )

    def run_benchmark(
        self,
        save_csv: bool = True,
        save_run_data: bool = False,
        load_from: Optional[str | Path] = None,
        run_data_dir: str | Path = "./benchmark_run_data",
    ) -> List[BenchmarkResult]:
        """Run benchmark for all algorithm configurations and return results.

        This is the single entry point for benchmarking. It can either:
        - Run algorithms and collect fresh data (default)
        - Load previously saved run data and re-evaluate

        Args:
            save_csv: Whether to save metrics to CSV file (default: True)
            save_run_data: Whether to save raw run data for later re-evaluation (default: False)
            load_from: Path to directory with saved run data. If provided, skips running
                algorithms and loads data instead. Supports re-evaluation with different
                success_loss (set via __init__ or by modifying self._success_loss before call).
            run_data_dir: Directory for saving run data when save_run_data=True (default: ./benchmark_run_data)

        Returns:
            List of BenchmarkResult, one for each algorithm configuration
        """
        # Determine data source
        if load_from is not None:
            run_data_list = self._load_run_data(Path(load_from))
            mode_str = f"LOADING from {load_from}"
        else:
            mode_str = "RUNNING algorithms"

        # Print header
        print("\n" + "=" * 70)
        print(f"BENCHMARK SUITE ({mode_str})")
        print("=" * 70)
        print(
            f"Problem: {self._problem._name if hasattr(self._problem, '_name') else 'Unknown'}"
        )
        print(f"Success threshold: {self._success_loss}")
        if load_from is None:
            print(f"Number of algorithms: {len(self._configs)}")
            print(f"Runs per algorithm: {self._n_runs}")
        print(f"Wall time checkpoints: {self._wall_time_steps}")
        print("=" * 70)

        # Collect or load data, then evaluate
        results = []
        run_data_list_for_saving = []

        if load_from is not None:
            # Data already loaded above
            for i, run_data in enumerate(run_data_list, 1):
                print(
                    f"\n[{i}/{len(run_data_list)}] Evaluating: {run_data.algorithm_name}"
                )
                result = self._evaluate_run_data(run_data)
                results.append(result)
                self._print_result_summary(result)
        else:
            # Run algorithms and collect data
            for i, config in enumerate(self._configs, 1):
                print(f"\n[{i}/{len(self._configs)}] Benchmarking: {config.name}")

                # Collect raw data
                run_data = self._collect_run_data(config)
                run_data_list_for_saving.append(run_data)

                # Evaluate
                result = self._evaluate_run_data(run_data)
                results.append(result)
                self._print_result_summary(result)

        # Print footer
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)

        # Save outputs
        if save_run_data and load_from is None:
            self._save_run_data(run_data_list_for_saving, run_data_dir)

        if save_csv:
            self._save_results_to_csv(results)

        return results

    def _print_result_summary(self, result: BenchmarkResult) -> None:
        """Print summary for a single algorithm result."""
        print(f"\n--- Summary for {result.algorithm_name} ---")
        print(
            f"  Success rate (final): {float(result.fraction_of_success.value[-1]):.1%}"
        )
        print(f"  Min loss (final): {float(result.min_loss.value[-1]):.6f}")
        print(
            f"  Avg loss (final): {float(result.avg_loss.mean[-1]):.6f} ± {float(result.avg_loss.std[-1]):.6f}"
        )
        print(
            f"  Time to success: {float(result.time_to_success.mean[-1]):.2f} ± {float(result.time_to_success.std[-1]):.2f}s"
        )
        print(
            f"  Diversity (overall): {float(result.solution_diversity_overall.mean[-1]):.4f} ± {float(result.solution_diversity_overall.std[-1]):.4f}"
        )

    def _load_run_data(self, data_dir: Path) -> List[AlgorithmRunData]:
        """Load saved run data from disk.

        Args:
            data_dir: Directory containing metadata.json and NPZ files

        Returns:
            List of AlgorithmRunData, one per algorithm
        """
        # Load metadata
        with open(data_dir / "metadata.json", "r") as f:
            metadata = json.load(f)

        saved_wall_time_steps = metadata["wall_time_steps"]

        # Validate wall_time_steps compatibility
        for wt in self._wall_time_steps:
            if wt not in saved_wall_time_steps:
                raise ValueError(
                    f"Requested wall_time {wt} not in saved wall_time_steps: {saved_wall_time_steps}. "
                    f"Current wall_time_steps: {self._wall_time_steps}"
                )

        print(f"\nLoading run data from: {data_dir}")
        print(f"  Saved wall_time_steps: {saved_wall_time_steps}")
        print(f"  Using wall_time_steps: {self._wall_time_steps}")

        run_data_list = []

        for algo_info in metadata["algorithms"]:
            npz_path = data_dir / algo_info["file"]
            print(f"  Loading {algo_info['name']}...")

            with np.load(npz_path, allow_pickle=True) as data:
                # Load raw arrays
                all_wall_time_indices_raw = [
                    list(arr) for arr in data["all_wall_time_indices"]
                ]

                # Filter wall_time_indices to match current wall_time_steps
                if self._wall_time_steps != saved_wall_time_steps:
                    idx_map = [
                        saved_wall_time_steps.index(wt) for wt in self._wall_time_steps
                    ]
                    all_wall_time_indices = [
                        [wti[i] for i in idx_map] for wti in all_wall_time_indices_raw
                    ]
                else:
                    all_wall_time_indices = all_wall_time_indices_raw

                run_data = AlgorithmRunData(
                    algorithm_name=str(data["algorithm_name"]),
                    calls_per_iteration=int(data["calls_per_iteration"]),
                    all_best_params=jnp.array(data["all_best_params"]),
                    all_best_params_history=[
                        jnp.array(arr) for arr in data["all_best_params_history"]
                    ],
                    all_losses=[jnp.array(arr) for arr in data["all_losses"]],
                    all_wall_time_indices=all_wall_time_indices,
                )
                run_data_list.append(run_data)

        return run_data_list

    def _save_run_data(
        self,
        run_data_list: List[AlgorithmRunData],
        output_dir: str | Path,
    ) -> Path:
        """Save raw run data to NPZ files for later re-evaluation.

        Args:
            run_data_list: List of AlgorithmRunData to save
            output_dir: Base directory for saving (will create timestamped subdirectory)

        Returns:
            Path to the created directory
        """
        output_dir = Path(output_dir)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        problem_name = (
            self._problem._name if hasattr(self._problem, "_name") else "problem"
        )
        save_dir = output_dir / f"{problem_name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata
        metadata = {
            "problem_name": problem_name,
            "success_loss": self._success_loss,
            "n_runs": self._n_runs,
            "wall_time_steps": self._wall_time_steps,
            "timestamp": timestamp,
            "algorithms": [],
        }

        for run_data in run_data_list:
            safe_name = run_data.algorithm_name.replace("/", "_").replace(" ", "_")
            npz_path = save_dir / f"{safe_name}.npz"

            np.savez(
                npz_path,
                algorithm_name=run_data.algorithm_name,
                calls_per_iteration=run_data.calls_per_iteration,
                all_best_params=np.array(run_data.all_best_params),
                all_best_params_history=np.array(
                    run_data.all_best_params_history, dtype=object
                ),
                all_losses=np.array(run_data.all_losses, dtype=object),
                all_wall_time_indices=np.array(
                    run_data.all_wall_time_indices, dtype=object
                ),
            )

            metadata["algorithms"].append(
                {
                    "name": run_data.algorithm_name,
                    "calls_per_iteration": run_data.calls_per_iteration,
                    "file": f"{safe_name}.npz",
                }
            )

        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nRun data saved to: {save_dir}")
        return save_dir

    def print_summary(self, results: list[BenchmarkResult]):
        """Print a summary comparison of all algorithm results.

        Displays a formatted table comparing algorithms on key metrics
        including success rate, min/avg loss, time to success, and diversity.

        Args:
            results: List of BenchmarkResult from run_benchmark()
        """
        print("\n" + "=" * 90)
        print("BENCHMARK SUMMARY (at final time step)")
        print("=" * 90)

        # Header
        print(
            f"{'Algorithm':<20} {'Success%':>10} {'Min Loss':>12} {'Avg Loss':>15} {'Time(s)':>15} {'Diversity':>15}"
        )
        print("-" * 90)

        # Results for each algorithm
        for result in results:
            name = result.algorithm_name[:19]  # Truncate if too long
            success = float(result.fraction_of_success.value[-1]) * 100
            min_loss = float(result.min_loss.value[-1])
            avg_loss_str = f"{float(result.avg_loss.mean[-1]):.4f}±{float(result.avg_loss.std[-1]):.4f}"
            time_str = f"{float(result.time_to_success.mean[-1]):.2f}±{float(result.time_to_success.std[-1]):.2f}"
            diversity_str = f"{float(result.solution_diversity_overall.mean[-1]):.3f}±{float(result.solution_diversity_overall.std[-1]):.3f}"

            print(
                f"{name:<20} {success:>9.1f}% {min_loss:>12.6f} {avg_loss_str:>15} {time_str:>15} {diversity_str:>15}"
            )

        print("=" * 90)

    def _save_results_to_csv(self, results: list[BenchmarkResult]):
        """Save benchmark results to a CSV file with timestamp.

        Creates a ./benchmark_results/ directory and saves results with filename
        format: benchmark_{problem_name}_{timestamp}.csv

        Args:
            results: List of BenchmarkResult from run_benchmark()
        """
        # Create benchmark_results directory
        output_dir = Path("./benchmark_results")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        problem_name = (
            self._problem._name if hasattr(self._problem, "_name") else "problem"
        )
        output_path = output_dir / f"benchmark_{problem_name}_{timestamp}.csv"

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # 1. UNIFORM HEADER GENERATION
            # Every metric gets a _mean and _std suffix, even single ones,
            # making the CSV easier to parse
            header = ["algorithm_name", "wall_time_step"]
            for field in fields(BenchmarkResult):
                if field.name == "algorithm_name":
                    continue
                # Regardless of type, we reserve two columns
                header.append(f"{field.name}_mean")
                header.append(f"{field.name}_std")

            writer.writerow(header)

            # 2. UNIFORM DATA WRITING
            for result in results:
                n_time_steps = len(self._wall_time_steps)

                for i in range(n_time_steps):
                    row = [result.algorithm_name, float(self._wall_time_steps[i])]

                    for field in fields(BenchmarkResult):
                        if field.name == "algorithm_name":
                            continue
                        metric = getattr(result, field.name)

                        if isinstance(metric, SingleMetric):
                            # For SingleMetric, the "Std" is always 0.0
                            row.append(float(metric.value[i]))
                            row.append(0.0)
                        elif isinstance(metric, AggregateMetric):
                            row.append(float(metric.mean[i]))
                            row.append(float(metric.std[i]))

                    writer.writerow(row)

        print(f"\nBenchmark results saved to: {output_path}")
