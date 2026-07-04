"""Benchmarking module for optimization algorithms.

This module provides tools to benchmark multiple optimization algorithms on a problem,
running each algorithm multiple times and computing various performance metrics.

Key classes:
    - Benchmark: Main benchmarking class that runs algorithms and evaluates performance
    - BenchmarkResult: Dataclass containing performance metrics across time samples
    - AlgorithmConfig: Configuration wrapper for algorithm + hyperparameters

Usage:
    >>> from dfbench.benchmark import Benchmark, AlgorithmConfig
    >>> from dfbench.algorithms import AdamGD, EvoxPSO
    >>>
    >>> problem = VoyagerProblem()
    >>> configs = [
    ...     AlgorithmConfig(AdamGD(), {"learning_rate": 0.1}, name="Adam_lr0.1"),
    ...     AlgorithmConfig(EvoxPSO(variant="PSO"), {"pop_size": 100}, name="PSO_100"),
    ... ]
    >>> benchmark = Benchmark(problem, success_loss=0.1, configs=configs, n_runs=100, max_time=300)
    >>> results = benchmark.run()
"""

from __future__ import annotations

import csv
import io
import json
import numpy as np
import jax.numpy as jnp
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, fields, field
from typing import Any
from jaxtyping import Array, Float

from dfbench import Objective
from dfbench.core.problem import ContinuousProblem, build_problem_from_spec
from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.storage import (
    LocalFilesystemBackend,
    NpzRunCollectionSerializer,
    RunCollectionSerializer,
    RunMetadata,
    RunState,
    StorageBackend,
)
from dfbench.benchmark.metrics import (
    run_min_loss,
    run_has_success,
    run_first_success_time,
    run_auc,
    agg_mean_std,
    agg_min,
    agg_fraction_true,
    agg_mean_std_filtered,
    multi_solution_diversity_overall,
    multi_solution_diversity_nn,
    multi_auc_top_k,
    compute_performance_profile,
    slice_history_at_time,
    get_value_at_time,
)


# --------- Data classes ---------


@dataclass
class SingleMetric:
    """Metric with a single value per time sample (not aggregated across runs)."""

    value: Float[Array, "n_time_samples"]


@dataclass
class AggregateMetric:
    """Metric aggregated across multiple runs with statistics."""

    mean: Float[Array, "n_time_samples"]
    std: Float[Array, "n_time_samples"]


@dataclass
class RunData:
    """Serializable data extracted from a single optimization run (Objective instance).

    This is what gets saved to disk for later re-evaluation. All arrays are numpy
    for efficient serialization.

    Attributes:
        loss_history: Loss at each evaluation
        time_steps: Elapsed time at each evaluation
        params_history: Parameters at each evaluation (bounded)
        best_loss: Best loss achieved
        best_params: Best parameters found (bounded)
        eval_count: Total number of evaluations
    """

    loss_history: np.ndarray  # shape: (n_evals,)
    time_steps: np.ndarray  # shape: (n_evals,)
    params_history: np.ndarray  # shape: (n_evals, n_params) or ragged
    best_loss: float
    best_params: np.ndarray  # shape: (n_params,)
    eval_count: int

    @classmethod
    def from_objective(cls, obj) -> "RunData":
        """Extract RunData from an Objective instance.

        Args:
            obj: Objective instance after optimization

        Returns:
            RunData with extracted arrays
        """
        # Use reduced (non-batched) properties for consistent benchmark data
        best_params = obj.best_params_bounded

        return cls(
            loss_history=np.array(obj.loss_history_reduced, dtype=np.float32),
            time_steps=np.array(obj.time_steps, dtype=np.float32),
            params_history=np.array(obj.params_history_reduced_bounded, dtype=object),
            best_loss=float(obj.best_loss)
            if obj.best_loss is not None
            else float("inf"),
            best_params=np.array(best_params)
            if best_params is not None
            else np.array([]),
            eval_count=obj.eval_count,
        )

    @classmethod
    def from_run_state(cls, state: RunState) -> "RunData":
        """Build a :class:`RunData` from a loaded :class:`RunState`.

        The benchmark metrics only need the reduced (non-batched) loss /
        params histories plus timing. This conversion bridges the
        canonical storage contract to the metric-evaluation layer.
        """
        loss = np.asarray(state.loss_history, dtype=object)
        # Reduce batched losses to per-step minima (matches loss_history_reduced)
        loss_reduced = np.array(
            [
                float(np.nanmin(np.asarray(step)))
                if np.asarray(step).ndim > 0
                else float(step)
                for step in loss.tolist()
            ],
            dtype=np.float32,
        )
        # Best params come from the state directly (already bounded at save time)
        best_params = (
            np.asarray(state.best_params, dtype=np.float64)
            if state.best_params.size > 0
            else np.array([])
        )
        # Params history: reduce batched entries to the argmin-of-loss representative
        params = np.asarray(state.params_history, dtype=object)
        params_reduced = []
        for i, step_params in enumerate(params.tolist()):
            arr = np.asarray(step_params)
            if arr.ndim > 1:  # batched
                step_loss = np.asarray(loss[i]) if i < len(loss) else None
                if step_loss is not None and step_loss.ndim > 0:
                    idx = int(np.nanargmin(step_loss))
                else:
                    idx = 0
                params_reduced.append(arr[idx])
            elif arr.ndim == 1:
                params_reduced.append(arr)
            else:
                params_reduced.append(np.array([]))
        return cls(
            loss_history=loss_reduced,
            time_steps=np.asarray(state.time_steps, dtype=np.float32),
            params_history=np.array(params_reduced, dtype=object)
            if params_reduced
            else np.array([], dtype=object),
            best_loss=float(state.best_loss),
            best_params=best_params,
            eval_count=int(state.eval_count),
        )

    def to_run_state(self, metadata=None) -> RunState:
        """Build a :class:`RunState` from this :class:`RunData`.

        Used when saving benchmark data: the benchmark extracts
        :class:`RunData` (the reduced view used by metrics) and this
        wraps it in the canonical storage contract for serialization.
        Histories not tracked by :class:`RunData` (gradients, Hessians,
        eval types) are stored as empty arrays.
        """
        from dfbench.core.storage.state import RunMetadata

        return RunState(
            loss_history=np.asarray(self.loss_history, dtype=object),
            grad_history=np.array([], dtype=object),
            hessian_history=np.array([], dtype=object),
            params_history=np.asarray(self.params_history, dtype=object),
            eval_type_history=np.array([], dtype=object),
            time_steps=np.asarray(self.time_steps, dtype=object),
            eval_count=int(self.eval_count),
            best_loss=float(self.best_loss),
            best_params=(
                np.asarray(self.best_params, dtype=np.float64)
                if self.best_params.size > 0
                else np.array([], dtype=np.float64)
            ),
            improvement_count=0,
            evals_since_improvement=0,
            log_call_count=0,
            eval_type_counts={},
            metadata=metadata or RunMetadata(),
        )


@dataclass
class AlgorithmRunData:
    """Collection of run data for one algorithm configuration.

    Attributes:
        algorithm_name: Name of the algorithm configuration
        runs: List of RunData, one per optimization run
        hyperparameters: Dictionary of hyperparameters used
    """

    algorithm_name: str
    runs: list[RunData]
    hyperparameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Results from benchmarking one algorithm across multiple time samples.

    All metric arrays have shape (n_time_samples,) corresponding to the
    sampled time points.

    Attributes:
        algorithm_name: Name of the algorithm configuration
        time_samples: Time points at which metrics were computed
        n_runs: Number of independent runs

        Metrics:
        - fraction_of_success: Fraction of runs achieving success at each time
        - min_loss: Global minimum loss across all runs at each time
        - avg_loss: Mean and std of per-run minimum losses at each time
        - time_to_success: Mean and std of first success time (successful runs only)
        - evals_to_success: Mean and std of evaluations to first success
        - solution_diversity_overall: Mean pairwise distance of successful solutions
        - solution_diversity_nn: Mean nearest-neighbor distance
        - auc_top_1: AUC of the best run (by min loss)
        - auc_top_10: Mean and std AUC of top 10% runs
        - performance_profile_auc: Normalized AUC of performance profile
    """

    algorithm_name: str
    time_samples: Float[Array, "n_time_samples"]
    n_runs: int

    # Single-value metrics
    fraction_of_success: SingleMetric
    min_loss: SingleMetric
    performance_profile_auc: SingleMetric
    auc_top_1: SingleMetric

    # Aggregate metrics
    avg_loss: AggregateMetric
    time_to_success: AggregateMetric
    evals_to_success: AggregateMetric
    solution_diversity_overall: AggregateMetric
    solution_diversity_nn: AggregateMetric
    auc_top_10: AggregateMetric


# --------- Algorithm Configuration ---------


class AlgorithmConfig:
    """Configuration for an algorithm to benchmark.

    Wraps an algorithm instance with its hyperparameters and optional name.
    """

    def __init__(
        self,
        algorithm: OptimizationAlgorithm,
        hyperparameters: dict[str, Any] | None = None,
        name: str | None = None,
    ):
        """Initialize algorithm configuration.

        Args:
            algorithm: The algorithm instance to benchmark
            hyperparameters: Dictionary of hyperparameters to pass to optimize()
            name: Optional custom name for this configuration (defaults to algorithm_str)
        """
        self.algorithm = algorithm
        self.hyperparameters = hyperparameters or {}
        self.name = name or algorithm.algorithm_str

    def __repr__(self) -> str:
        return f"AlgorithmConfig({self.name}, {self.hyperparameters})"


# --------- Main Benchmark Class ---------


class Benchmark:
    """Benchmark multiple optimization algorithms on a problem.

    Runs algorithms multiple times, collects data via Objective instances,
    and computes performance metrics at sampled time points.

    Example:
        >>> benchmark = Benchmark(
        ...     problem=problem,
        ...     success_loss=0.1,
        ...     configs=[AlgorithmConfig(AdamGD(), {"learning_rate": 0.1})],
        ...     n_runs=100,
        ...     max_time=300,
        ... )
        >>> results = benchmark.run()
        >>> benchmark.print_summary(results)
    """

    def __init__(
        self,
        problem: ContinuousProblem,
        success_loss: float,
        configs: list[AlgorithmConfig],
        random_seed: int | None = None,
        n_runs: int = 100,
        max_time: float = 300.0,
        n_time_samples: int = 100,
        random_baseline_loss: float | None = None,
        storage_backend: StorageBackend | None = None,
        run_collection_serializer: RunCollectionSerializer | None = None,
    ):
        """Initialize the benchmark suite.

        Args:
            problem: The optimization problem to benchmark algorithms on
            success_loss: Loss threshold below which a run is considered successful
            configs: List of algorithm configurations to benchmark
            n_runs: Number of independent runs per algorithm configuration
            max_time: Maximum wall-clock time per run in seconds
            n_time_samples: Number of time points to sample for metrics (default: 100)
            random_baseline_loss: Expected loss of a random guess for normalized AUC.
                If None, AUC normalization is disabled.
            random_seed: Master random seed for reproducibility. If provided, generates
                deterministic per-run seeds.
            storage_backend: Where benchmark run data (NPZ/JSON) is
                physically stored. Defaults to a local filesystem backend
                (current working directory). Swapping this redirects all
                benchmark artifacts without code changes here.
            run_collection_serializer: How the multi-run collection for
                each algorithm is encoded on disk. Defaults to
                :class:`NpzRunCollectionSerializer`, which stores each
                run as a full self-describing :class:`RunState` (with
                embedded ``problem_spec``). Swapping this changes the
                on-disk format for benchmark data.
        """
        self._problem = problem
        self._success_loss = success_loss
        self._configs = configs
        self._n_runs = n_runs
        self._max_time = max_time
        self._n_time_samples = n_time_samples
        self._random_baseline_loss = random_baseline_loss
        self._random_seed = random_seed
        self._storage_backend: StorageBackend = (
            storage_backend or LocalFilesystemBackend()
        )
        self._collection_serializer: RunCollectionSerializer = (
            run_collection_serializer or NpzRunCollectionSerializer()
        )

        # Generate time sample points (excluding 0, evenly spaced up to max_time)
        # E.g., n_time_samples=10, max_time=30 -> [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        self._time_samples = np.linspace(
            max_time / n_time_samples, max_time, n_time_samples
        )

    @property
    def time_samples(self) -> np.ndarray:
        """Time points at which metrics are computed."""
        return self._time_samples

    # --------- Main entry point ---------

    def run(
        self,
        verbose: int = 1,
        save_csv: bool = True,
        save_run_data: bool = False,
        load_from: str | Path | None = None,
        output_dir: str | Path = "./data/benchmark_run_data",
    ) -> list[BenchmarkResult]:
        """Run the benchmark and return results.

        Args:
            save_csv: Whether to save metrics to CSV file (default: True)
            save_run_data: Whether to save raw run data for later re-evaluation
            load_from: Path to directory with saved run data. If provided, loads
                data instead of running algorithms.
            output_dir: Base directory for saving run data
            verbose: Verbosity level (default: 1)

        Returns:
            List of BenchmarkResult, one per algorithm configuration
        """
        self._verbose = verbose  # TODO implement verbosity levels
        self._print_header(load_from)

        if load_from is not None:
            all_run_data = self._load_run_data(Path(load_from))
        else:
            all_run_data = self._collect_all_run_data(save_run_data, output_dir)

        # Evaluate metrics for each algorithm
        results = []
        for i, algo_data in enumerate(all_run_data, 1):
            print(f"\n[{i}/{len(all_run_data)}] Evaluating: {algo_data.algorithm_name}")
            result = self._evaluate_algorithm(algo_data)
            results.append(result)
            self._print_result_summary(result)

        self._print_footer()

        if save_csv:
            self._save_results_to_csv(results)

        return results

    # --------- Data collection ---------

    def _collect_all_run_data(
        self,
        save_incrementally: bool,
        output_dir: str | Path,
    ) -> list[AlgorithmRunData]:
        """Run all algorithms and collect data.

        Args:
            save_incrementally: Whether to save data after each algorithm
            output_dir: Directory for saving

        Returns:
            List of AlgorithmRunData, one per configuration
        """
        # Prepare output directory if saving
        save_dir = None
        if save_incrementally:
            save_dir = self._prepare_save_dir(output_dir)
            print(f"\nRun data will be saved to: {save_dir}")

        run_seeds = self._generate_run_seeds()

        all_run_data = []

        for i, config in enumerate(self._configs, 1):
            print(f"\n{'=' * 70}")
            print(f"[{i}/{len(self._configs)}] Running: {config.name}")
            print(f"Hyperparameters: {config.hyperparameters}")
            print(f"Runs: {self._n_runs}, Max time: {self._max_time}s")
            print(f"{'=' * 70}")

            algo_data = self._collect_algorithm_runs(config, run_seeds)
            all_run_data.append(algo_data)

            if save_dir is not None:
                self._save_algorithm_run_data(algo_data, save_dir)

        # Save metadata
        if save_dir is not None:
            self._save_metadata(all_run_data, save_dir)

        return all_run_data

    def _generate_run_seeds(self) -> list[int] | None:
        """Derive deterministic per-run seeds from the benchmark master seed."""
        if self._random_seed is None:
            return None

        rng = np.random.RandomState(self._random_seed)
        return [int(rng.randint(0, 2**31)) for _ in range(self._n_runs)]

    def _collect_algorithm_runs(
        self,
        config: AlgorithmConfig,
        run_seeds: list[int] | None,
    ) -> AlgorithmRunData:
        """Run a single algorithm multiple times and collect data.

        Args:
            config: Algorithm configuration
            run_seeds: Per-run random seeds (or None for non-deterministic)

        Returns:
            AlgorithmRunData with all runs
        """
        runs = []

        for i_run in range(self._n_runs):
            print(f"  Run {i_run + 1}/{self._n_runs}...", end=" ", flush=True)

            # Prepare hyperparameters
            kwargs = config.hyperparameters.copy()

            # Determine unbounded based on algorithm type
            # Gradient-based algorithms use unbounded space, others use bounded
            unbounded = config.algorithm.algorithm_type == AlgorithmType.GRADIENT_BASED

            obj = Objective(
                problem=self._problem,
                unbounded=unbounded,
                max_time=self._max_time,
                save_time_steps=True,
                save_params_history=True,
                verbose=self._verbose - 1,
                print_every=100,  # Print every 100 evals if verbose >= 1
            )

            if run_seeds is not None:
                kwargs["random_seed"] = run_seeds[i_run]

            # Run optimization acts on Objective
            config.algorithm.optimize(objective=obj, **kwargs)

            # Extract data
            run_data = RunData.from_objective(obj)
            runs.append(run_data)

            print(f"loss={run_data.best_loss:.6f}, evals={run_data.eval_count}")

        return AlgorithmRunData(
            algorithm_name=config.name,
            runs=runs,
            hyperparameters=config.hyperparameters,
        )

    # --------- Metric evaluation ---------

    def _evaluate_algorithm(self, algo_data: AlgorithmRunData) -> BenchmarkResult:
        """Evaluate metrics for one algorithm across all time samples.

        Args:
            algo_data: Collected run data for the algorithm

        Returns:
            BenchmarkResult with all metrics
        """
        runs = algo_data.runs
        n_runs = len(runs)

        # Result containers
        fraction_of_success_list = []
        min_loss_list = []
        avg_loss_list = []
        time_to_success_list = []
        evals_to_success_list = []
        auc_top_1_list = []
        auc_top_10_list = []
        performance_profile_auc_list = []
        diversity_overall_list = []
        diversity_nn_list = []

        for t in self._time_samples:
            # Slice each run's data at time t
            run_losses_at_t = []
            run_params_at_t = []
            run_min_losses = []
            run_has_successes = []
            run_first_success_times = []
            run_first_success_evals = []
            run_aucs = []

            for run in runs:
                # Get data up to time t
                losses_slice = slice_history_at_time(
                    run.loss_history.tolist(), run.time_steps.tolist(), t
                )
                losses_arr = jnp.array(losses_slice) if losses_slice else jnp.array([])
                time_slice = slice_history_at_time(
                    run.time_steps.tolist(), run.time_steps.tolist(), t
                )
                time_arr = jnp.array(time_slice) if time_slice else jnp.array([])

                # Per-run metrics
                min_loss = run_min_loss(losses_arr)
                has_success = run_has_success(losses_arr, self._success_loss)
                first_success_t = run_first_success_time(
                    losses_arr, time_arr, self._success_loss
                )

                # First success eval count
                from dfbench.benchmark.metrics import run_first_success_idx

                first_success_idx = run_first_success_idx(
                    losses_arr, self._success_loss
                )
                first_success_evals = (
                    first_success_idx + 1 if first_success_idx is not None else None
                )

                # AUC
                auc = run_auc(
                    losses_arr,
                    time_arr,
                    floor=self._success_loss,
                    baseline_loss=self._random_baseline_loss,
                    max_time=t if t > 0 else 1.0,
                )

                run_losses_at_t.append(losses_arr)
                run_min_losses.append(min_loss)
                run_has_successes.append(has_success)
                run_first_success_times.append(first_success_t)
                run_first_success_evals.append(first_success_evals)
                run_aucs.append(auc)

                # Get params at t for diversity (only if successful)
                if has_success:
                    params_at_t = get_value_at_time(
                        run.params_history.tolist(), run.time_steps.tolist(), t
                    )
                    if params_at_t is not None:
                        run_params_at_t.append(params_at_t)

            # Aggregate metrics
            fraction_of_success_list.append(agg_fraction_true(run_has_successes))
            min_loss_list.append(agg_min(run_min_losses))
            avg_loss_list.append(agg_mean_std(run_min_losses))
            time_to_success_list.append(agg_mean_std_filtered(run_first_success_times))
            evals_to_success_list.append(agg_mean_std_filtered(run_first_success_evals))

            # AUC top 1 and top 10%
            if run_min_losses:
                best_run_idx = run_min_losses.index(min(run_min_losses))
                auc_top_1_list.append(run_aucs[best_run_idx])
            else:
                auc_top_1_list.append(float("nan"))
            auc_top_10_list.append(
                multi_auc_top_k(run_min_losses, run_aucs, k_fraction=0.1)
            )

            # Performance profile
            _, _, perf_auc = compute_performance_profile(run_min_losses)
            performance_profile_auc_list.append(perf_auc)

            # Diversity
            if len(run_params_at_t) >= 2:
                params_array = jnp.array(run_params_at_t)
                bounds = (
                    self._problem.bounds if hasattr(self._problem, "bounds") else None
                )
                diversity_overall_list.append(
                    multi_solution_diversity_overall(params_array, bounds)
                )
                diversity_nn_list.append(
                    multi_solution_diversity_nn(params_array, bounds)
                )
            else:
                diversity_overall_list.append((0.0, 0.0))
                diversity_nn_list.append((0.0, 0.0))

        # Build result
        return BenchmarkResult(
            algorithm_name=algo_data.algorithm_name,
            time_samples=jnp.array(self._time_samples),
            n_runs=n_runs,
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
            evals_to_success=AggregateMetric(
                mean=jnp.array([m for m, _ in evals_to_success_list]),
                std=jnp.array([s for _, s in evals_to_success_list]),
            ),
            solution_diversity_overall=AggregateMetric(
                mean=jnp.array([m for m, _ in diversity_overall_list]),
                std=jnp.array([s for _, s in diversity_overall_list]),
            ),
            solution_diversity_nn=AggregateMetric(
                mean=jnp.array([m for m, _ in diversity_nn_list]),
                std=jnp.array([s for _, s in diversity_nn_list]),
            ),
            auc_top_1=SingleMetric(value=jnp.array(auc_top_1_list)),
            auc_top_10=AggregateMetric(
                mean=jnp.array([m for m, _ in auc_top_10_list]),
                std=jnp.array([s for _, s in auc_top_10_list]),
            ),
            performance_profile_auc=SingleMetric(
                value=jnp.array(performance_profile_auc_list)
            ),
        )

    # --------- Save/Load ---------

    def _prepare_save_dir(self, base_dir: str | Path) -> Path:
        """Create timestamped save directory."""
        base_dir = Path(base_dir)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        problem_name = getattr(self._problem, "name", "problem")
        save_dir = base_dir / f"{problem_name}_{timestamp}"
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def _problem_spec(self) -> dict | None:
        """Return the problem's reconstructive spec as a typed-container dict.

        Produces a :class:`~dfbench.core.problem.ProblemSpec`-shaped dict
        (``{"type", "version", "params"}``) when the problem implements
        ``to_problem_spec``; falls back to the legacy ``to_spec()`` dict
        for older problem classes. Returns ``None`` if neither is
        available or the problem fails to describe itself.
        """
        spec_fn = getattr(self._problem, "to_problem_spec", None)
        if callable(spec_fn):
            try:
                return spec_fn().to_dict()
            except Exception:
                pass
        legacy = getattr(self._problem, "to_spec", None)
        if callable(legacy):
            try:
                return legacy()
            except Exception:
                return None
        return None

    def _save_algorithm_run_data(
        self,
        algo_data: AlgorithmRunData,
        save_dir: Path,
    ) -> Path:
        """Save one algorithm's runs as a self-describing collection.

        Each run is wrapped in a :class:`RunState` (with embedded
        :class:`RunMetadata` including the ``problem_spec``) and serialized
        via :attr:`_collection_serializer`. The backend writes the bytes
        atomically.
        """
        safe_name = algo_data.algorithm_name.replace("/", "_").replace(" ", "_")
        ext = getattr(self._collection_serializer, "extension", "npz")
        file_path = save_dir / f"{safe_name}.{ext}"

        # Wrap each RunData in a RunState with full metadata
        problem_name = getattr(self._problem, "name", "problem")
        spec = self._problem_spec()
        run_states = []
        for run in algo_data.runs:
            meta = RunMetadata(
                problem_name=problem_name,
                algorithm_name=algo_data.algorithm_name,
                timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                max_time=self._max_time,
                unbounded=False,
            )
            if spec is not None:
                meta.extra["problem_spec"] = spec
            run_states.append(run.to_run_state(metadata=meta))

        data = self._collection_serializer.serialize_collection(
            algorithm_name=algo_data.algorithm_name,
            hyperparameters=algo_data.hyperparameters,
            runs=run_states,
        )
        self._storage_backend.save_bytes(file_path, data)

        print(f"  Saved {len(algo_data.runs)} runs to {file_path.name}")
        return file_path

    def _save_metadata(
        self,
        all_run_data: list[AlgorithmRunData],
        save_dir: Path,
    ) -> None:
        """Save metadata.json file via the storage backend.

        Records the benchmark configuration plus the problem's
        reconstructive ``problem_spec`` so the benchmark directory is
        fully self-describing.
        """
        ext = getattr(self._collection_serializer, "extension", "npz")
        metadata = {
            "problem_name": getattr(self._problem, "name", "problem"),
            "success_loss": self._success_loss,
            "n_runs": self._n_runs,
            "max_time": self._max_time,
            "n_time_samples": self._n_time_samples,
            "random_seed": self._random_seed,
            "random_baseline_loss": self._random_baseline_loss,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "problem_spec": self._problem_spec(),
            "algorithms": [
                {
                    "name": ad.algorithm_name,
                    "hyperparameters": ad.hyperparameters,
                    "file": f"{ad.algorithm_name.replace('/', '_').replace(' ', '_')}.{ext}",
                }
                for ad in all_run_data
            ],
        }

        metadata_path = save_dir / "metadata.json"
        self._storage_backend.save_bytes(
            metadata_path, json.dumps(metadata, indent=2).encode("utf-8")
        )

        print(f"\nMetadata saved to: {metadata_path}")

    def _load_run_data(self, data_dir: Path) -> list[AlgorithmRunData]:
        """Load saved run data from disk via the storage backend.

        Supports the current collection format (self-describing
        :class:`RunState` per run) and the legacy per-algorithm NPZ
        format (with ``time_steps_list`` or ``all_wall_time_indices``).
        """
        metadata_bytes = self._storage_backend.load_bytes(data_dir / "metadata.json")
        metadata = json.loads(metadata_bytes.decode("utf-8"))

        print(f"\nLoading run data from: {data_dir}")
        print(f"  Saved max_time: {metadata.get('max_time', 'N/A')}")

        all_run_data = []

        for algo_info in metadata["algorithms"]:
            file_path = data_dir / algo_info["file"]
            print(f"  Loading {algo_info['name']}...")

            data_bytes = self._storage_backend.load_bytes(file_path)

            # Try the collection format first
            try:
                algo_name, hyperparams, run_states = (
                    self._collection_serializer.deserialize_collection(data_bytes)
                )
                runs = [RunData.from_run_state(s) for s in run_states]
            except Exception:
                # Fall back to legacy per-algorithm NPZ format
                runs = self._load_legacy_npz(data_bytes, metadata)

            all_run_data.append(
                AlgorithmRunData(
                    algorithm_name=algo_info["name"],
                    runs=runs,
                    hyperparameters=algo_info.get("hyperparameters", {}),
                )
            )

        return all_run_data

    def _load_legacy_npz(self, data_bytes: bytes, metadata: dict) -> list[RunData]:
        """Load runs from a legacy per-algorithm NPZ file.

        Handles both the ``time_steps_list`` format and the older
        ``all_wall_time_indices`` format, converting the latter via
        linear interpolation.
        """
        with np.load(io.BytesIO(data_bytes), allow_pickle=True) as data:
            if "time_steps_list" in data:
                runs = []
                n_runs = int(data["n_runs"])
                for i in range(n_runs):
                    runs.append(
                        RunData(
                            loss_history=data["loss_histories"][i],
                            time_steps=data["time_steps_list"][i],
                            params_history=data["params_histories"][i],
                            best_loss=float(data["best_losses"][i]),
                            best_params=data["best_params_list"][i],
                            eval_count=int(data["eval_counts"][i]),
                        )
                    )
                return runs

            if "all_wall_time_indices" in data:
                return self._load_legacy_format(data, metadata)

            raise ValueError(
                "Unknown legacy data format: missing time_steps_list and "
                "all_wall_time_indices."
            )

    def _load_legacy_format(self, data, metadata) -> list[RunData]:
        """Load runs from legacy format and convert wall_time_indices to time_steps.

        Legacy format has:
        - all_losses: list of loss arrays per run
        - all_wall_time_indices: list of indices at wall time checkpoints
        - all_best_params_history: list of params history per run

        We reconstruct time_steps by interpolating based on wall_time_indices.
        """
        wall_time_steps = metadata.get("wall_time_steps", [self._max_time])
        runs = []

        all_losses = data["all_losses"]
        all_wall_time_indices = data["all_wall_time_indices"]
        all_best_params = data["all_best_params"]
        all_best_params_history = data.get(
            "all_best_params_history", [None] * len(all_losses)
        )

        for i in range(len(all_losses)):
            losses = np.array(all_losses[i])
            wall_indices = list(all_wall_time_indices[i])
            best_params = np.array(all_best_params[i])
            params_history = all_best_params_history[i]

            # Convert wall_time_indices to time_steps
            time_steps = self._wall_time_indices_to_time_steps(
                n_iterations=len(losses),
                wall_time_indices=wall_indices,
                wall_time_steps=wall_time_steps,
            )

            runs.append(
                RunData(
                    loss_history=losses.astype(np.float32),
                    time_steps=time_steps.astype(np.float32),
                    params_history=np.array(params_history, dtype=object)
                    if params_history is not None
                    else np.array([]),
                    best_loss=float(np.min(losses))
                    if len(losses) > 0
                    else float("inf"),
                    best_params=best_params,
                    eval_count=len(losses),
                )
            )

        return runs

    def _wall_time_indices_to_time_steps(
        self,
        n_iterations: int,
        wall_time_indices: list[int],
        wall_time_steps: list[float],
    ) -> np.ndarray:
        """Convert legacy wall_time_indices to per-iteration time_steps.

        Uses linear interpolation between checkpoints.

        Args:
            n_iterations: Total number of iterations
            wall_time_indices: Iteration index at each wall time checkpoint
            wall_time_steps: Wall time values at each checkpoint

        Returns:
            Array of estimated time at each iteration
        """
        if n_iterations == 0:
            return np.array([])

        # Build interpolation points: (iteration, time)
        # Start with (0, 0)
        interp_iters = [0]
        interp_times = [0.0]

        for idx, t in zip(wall_time_indices, wall_time_steps):
            if idx > 0:  # Skip if index is 0 (already have it)
                interp_iters.append(idx)
                interp_times.append(t)

        # Interpolate
        iterations = np.arange(n_iterations)
        time_steps = np.interp(iterations, interp_iters, interp_times)

        return time_steps

    @staticmethod
    def reconstruct_problem(data_dir: str | Path) -> ContinuousProblem | None:
        """Rebuild the problem that produced a benchmark directory.

        Reads the ``problem_spec`` recorded in ``metadata.json`` and
        reconstructs the problem via :func:`build_problem_from_spec`.
        Accepts both the typed container form
        (``{"type", "version", "params"}``) and the legacy flat form
        (``{"type", <kwargs>}``); both are normalized via
        :meth:`ProblemSpec.from_dict`.

        Returns ``None`` if no spec was recorded (e.g. the problem did
        not implement ``to_spec`` or the data was saved by an older
        version).
        """
        data_dir = Path(data_dir)
        metadata_path = data_dir / "metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path) as f:
            metadata = json.load(f)
        spec = metadata.get("problem_spec")
        if spec is None:
            return None
        return build_problem_from_spec(spec)

    # --------- Output ---------

    def _print_header(self, load_from: str | Path | None) -> None:
        """Print benchmark header."""
        mode = "LOADING" if load_from else "RUNNING"
        print("\n" + "=" * 70)
        print(f"BENCHMARK ({mode})")
        print("=" * 70)
        print(f"Problem: {getattr(self._problem, 'name', 'Unknown')}")
        print(f"Success threshold: {self._success_loss}")
        print(f"Algorithms: {len(self._configs)}")
        print(f"Runs per algorithm: {self._n_runs}")
        print(f"Max time: {self._max_time}s")
        print(f"Time samples: {self._n_time_samples}")
        print("=" * 70)

    def _print_footer(self) -> None:
        """Print benchmark footer."""
        print("\n" + "=" * 70)
        print("BENCHMARK COMPLETE")
        print("=" * 70)

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
        tts_mean = float(result.time_to_success.mean[-1])
        tts_std = float(result.time_to_success.std[-1])
        if not np.isnan(tts_mean):
            print(f"  Time to success: {tts_mean:.2f} ± {tts_std:.2f}s")
        else:
            print("  Time to success: N/A (no successful runs)")

    def print_summary(self, results: list[BenchmarkResult]) -> None:
        """Print a summary comparison table of all algorithms."""
        print("\n" + "=" * 90)
        print("BENCHMARK SUMMARY (at final time)")
        print("=" * 90)

        header = f"{'Algorithm':<25} {'Success%':>10} {'Min Loss':>12} {'Avg Loss':>18} {'Time(s)':>15}"
        print(header)
        print("-" * 90)

        for result in results:
            name = result.algorithm_name[:24]
            success = float(result.fraction_of_success.value[-1]) * 100
            min_loss = float(result.min_loss.value[-1])
            avg_mean = float(result.avg_loss.mean[-1])
            avg_std = float(result.avg_loss.std[-1])
            tts_mean = float(result.time_to_success.mean[-1])
            tts_std = float(result.time_to_success.std[-1])

            avg_str = f"{avg_mean:.4f}±{avg_std:.4f}"
            tts_str = (
                f"{tts_mean:.1f}±{tts_std:.1f}" if not np.isnan(tts_mean) else "N/A"
            )

            print(
                f"{name:<25} {success:>9.1f}% {min_loss:>12.6f} {avg_str:>18} {tts_str:>15}"
            )

        print("=" * 90)

    def _save_results_to_csv(self, results: list[BenchmarkResult]) -> None:
        """Save benchmark results to a CSV file via the storage backend."""
        output_dir = Path("./data/benchmark_results")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        problem_name = getattr(self._problem, "name", "problem")
        output_path = output_dir / f"benchmark_{problem_name}_{timestamp}.csv"

        buffer = io.StringIO()
        writer = csv.writer(buffer)

        # Header
        header = ["algorithm_name", "time_sample"]
        for fld in fields(BenchmarkResult):
            if fld.name in ("algorithm_name", "time_samples", "n_runs"):
                continue
            header.extend([f"{fld.name}_mean", f"{fld.name}_std"])
        writer.writerow(header)

        # Data rows
        for result in results:
            for i, t in enumerate(self._time_samples):
                row = [result.algorithm_name, float(t)]

                for fld in fields(BenchmarkResult):
                    if fld.name in ("algorithm_name", "time_samples", "n_runs"):
                        continue
                    metric = getattr(result, fld.name)

                    if isinstance(metric, SingleMetric):
                        row.extend([float(metric.value[i]), 0.0])
                    elif isinstance(metric, AggregateMetric):
                        row.extend([float(metric.mean[i]), float(metric.std[i])])

                writer.writerow(row)

        self._storage_backend.save_bytes(output_path, buffer.getvalue().encode("utf-8"))

        print(f"\nResults saved to: {output_path}")
