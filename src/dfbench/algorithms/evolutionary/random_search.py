import jax
import jax.numpy as jnp
import numpy as np
import time
from collections import deque
from jax import random
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)


class RandomSearch(OptimizationAlgorithm):
    """Random Search optimization algorithm.

    Samples random parameters uniformly within the problem's bounds and evaluates them.
    Useful as a baseline for comparing more sophisticated optimization algorithms.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("random_search").
        algorithm_type (AlgorithmType): Type classification (EVOLUTIONARY).
        _problem (ContinuousProblem): The optimization problem instance.
        _batch_size (int): Number of samples to evaluate in parallel per batch.
        _vectorized_objective (Callable): JIT-compiled vectorized objective function.

    Note:
        This algorithm requires the problem to have a `bounds` attribute and uses
        `problem.objective_function` which expects bounded parameters.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = RandomSearch(problem, batch_size=100)
        >>> best_params, history, losses, wall_indices = optimizer.optimize(
        ...     n_samples=10000,
        ...     wall_times=[30, 60, 120],
        ... )
    """

    algorithm_str: str = "random_search"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self,
        problem: ContinuousProblem,
        batch_size: int = 100,
    ) -> None:
        """Initialize Random Search optimizer.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
                Must have a `bounds` attribute (shape [2, n_params]).
            batch_size (int): Number of samples to evaluate in parallel per batch.
                Defaults to 100.
        """
        self._problem = problem
        self._batch_size = batch_size

        # Validate that the problem has bounds
        if not hasattr(problem, "bounds"):
            raise ValueError(
                "RandomSearch requires the problem to have a 'bounds' attribute. "
                "The bounds should be a numpy array of shape [2, n_params] with "
                "[lower_bounds, upper_bounds]."
            )

        # Create vectorized objective function
        self._vectorized_objective = jax.jit(
            jax.vmap(self._problem.objective_function, in_axes=0)
        )

        # Warmup the function to compile it
        _ = self._vectorized_objective(
            jnp.zeros((self._batch_size, self._problem.n_params))
        )

    @jaxtyped(typechecker=typechecker)
    def optimize(
        self,
        save_to_file: bool = True,
        return_best_params_history: bool = False,
        random_seed: int | None = None,
        wall_times: list[int | float] | None = None,
        n_samples: int = 1000,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_batches {self._problem.n_params}"] | None,
        Float[Array, "n_samples"],
    ]:
        """Run Random Search optimization.

        Args:
            save_to_file (bool): Whether to save optimization results to file. Defaults to True.
            return_best_params_history (bool): Whether to track best parameters at each
                batch. Defaults to False.
            random_seed (int | None): Random seed for reproducibility. Defaults to None.
            wall_times (list[int | float] | None): List of wall-time checkpoints (in seconds).
                The algorithm runs until the maximum checkpoint. At each checkpoint,
                the current sample index is recorded. If None, runs for n_samples.
                Defaults to None.
            n_samples (int): Total number of random samples to evaluate. Defaults to 1000.

        Returns:
            tuple: A 4-tuple containing:
                - best_params (Float[Array, "n_params"]): Best parameters found.
                - best_params_history (Float[Array, "n_batches n_params"] | None): History of
                  best parameters per batch. None if return_best_params_history=False.
                - losses (Float[Array, "n_samples"]): Loss for each evaluated sample.
                - wall_time_indices (list[int] | None): Sample indices corresponding to
                  each wall_times checkpoint. None if wall_times is None.
        """
        # Set random seed
        seed = random_seed if random_seed is not None else 0
        key = random.PRNGKey(seed)

        # Get bounds
        lower, upper = self._problem.bounds[0], self._problem.bounds[1]

        # Initialize tracking variables
        all_losses = []
        all_params = []
        best_params_history = []
        best_loss = float("inf")
        best_params = None

        # Initialize wall_time_indices tracking
        wall_time_indices: list[int] | None = None
        wall_times_remaining: deque[int | float] | None = None
        max_wall_time: float | None = None
        if wall_times is not None:
            wall_time_indices = []
            wall_times_remaining = deque(sorted(wall_times))
            max_wall_time = wall_times_remaining[-1]

        n_batches = (n_samples + self._batch_size - 1) // self._batch_size
        sample_count = 0

        if wall_times is not None:
            # Wall-time constrained: run until max_wall_time
            start_time = time.time()
            batch_idx = 0

            while (time.time() - start_time) < max_wall_time:
                elapsed = time.time() - start_time

                # Record sample index at wall_times checkpoints
                while wall_times_remaining and elapsed >= wall_times_remaining[0]:
                    wall_time_indices.append(sample_count)
                    wall_times_remaining.popleft()

                # Generate random samples
                key, subkey = random.split(key)
                random_params = random.uniform(
                    subkey,
                    shape=(self._batch_size, self._problem.n_params),
                    minval=lower,
                    maxval=upper,
                )

                # Evaluate batch
                losses = self._vectorized_objective(random_params)
                all_losses.append(losses)
                all_params.append(random_params)

                # Update best
                batch_best_idx = jnp.argmin(losses)
                batch_best_loss = losses[batch_best_idx]
                if batch_best_loss < best_loss:
                    best_loss = float(batch_best_loss)
                    best_params = random_params[batch_best_idx]
                    print(f"Batch {batch_idx}: New best loss = {best_loss:.6f}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                sample_count += self._batch_size
                batch_idx += 1

                if batch_idx % 10 == 0:
                    print(f"Processed {sample_count} samples, elapsed: {elapsed:.1f}s")

            # Fill remaining wall_times that weren't reached
            while wall_times_remaining:
                wall_time_indices.append(sample_count)
                wall_times_remaining.popleft()
        else:
            # Sample-count constrained
            for batch_idx in range(n_batches):
                current_batch_size = min(
                    self._batch_size, n_samples - batch_idx * self._batch_size
                )
                key, subkey = random.split(key)

                # Sample uniformly within bounds
                random_params = random.uniform(
                    subkey,
                    shape=(current_batch_size, self._problem.n_params),
                    minval=lower,
                    maxval=upper,
                )

                # Evaluate batch
                losses = self._vectorized_objective(random_params)
                all_losses.append(losses)
                all_params.append(random_params)

                # Update best
                batch_best_idx = jnp.argmin(losses)
                batch_best_loss = losses[batch_best_idx]
                if batch_best_loss < best_loss:
                    best_loss = float(batch_best_loss)
                    best_params = random_params[batch_best_idx]
                    print(f"Batch {batch_idx}: New best loss = {best_loss:.6f}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                sample_count += current_batch_size

                if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                    print(f"Processed {sample_count}/{n_samples} samples")

        # Combine all batches
        all_losses = jnp.concatenate(all_losses)
        best_params_history = (
            jnp.array(best_params_history) if return_best_params_history else None
        )

        # Print final statistics
        print(f"\nRandom Search statistics ({sample_count} samples):")
        print(f"  Best:   {best_loss:.6f}")
        print(f"  Mean:   {float(jnp.mean(all_losses)):.6f}")
        print(f"  Std:    {float(jnp.std(all_losses)):.6f}")
        print(f"  Median: {float(jnp.median(all_losses)):.6f}")

        if save_to_file:
            self._problem.output_to_files(
                best_params=best_params,
                losses=all_losses,
                algorithm_str=self.algorithm_str,
                hyper_param_str=f"n{n_samples}",
            )

        return best_params, best_params_history, all_losses, wall_time_indices

    def estimate_baseline_statistics(
        self,
        n_samples: int = 1000,
        n_runs: int = 20,
        seed_start: int = 0,
    ) -> dict:
        """Estimate baseline statistics over multiple independent runs.

        This method runs random search multiple times with different seeds
        and computes statistics over the mean losses of each run. Useful for
        establishing a robust random baseline for comparison.

        Args:
            n_samples (int): Number of random samples per run. Defaults to 1000.
            n_runs (int): Number of independent runs to perform. Defaults to 20.
            seed_start (int): Starting seed value. Seeds will be seed_start, seed_start+1, ...
                Defaults to 0.

        Returns:
            dict: Dictionary with statistics:
                - mean: Mean of per-run means
                - std: Standard deviation of per-run means
                - min: Minimum per-run mean
                - max: Maximum per-run mean
                - median: Median of per-run means
                - run_means: List of mean losses for each run
        """
        print(
            f"Estimating random baseline over {n_runs} runs ({n_samples} samples each)..."
        )

        run_means = []
        for i in range(n_runs):
            seed = seed_start + i
            _, _, losses, _ = self.optimize(
                save_to_file=False,
                return_best_params_history=False,
                random_seed=seed,
                n_samples=n_samples,
            )
            run_mean = float(jnp.mean(losses))
            run_means.append(run_mean)
            print(f"  Run {i + 1}/{n_runs} (seed={seed}): mean = {run_mean:.6f}")

        mean_baseline = sum(run_means) / len(run_means)
        std_baseline = (
            sum((x - mean_baseline) ** 2 for x in run_means) / len(run_means)
        ) ** 0.5

        stats = {
            "mean": mean_baseline,
            "std": std_baseline,
            "min": min(run_means),
            "max": max(run_means),
            "median": sorted(run_means)[len(run_means) // 2],
            "run_means": run_means,
        }

        print(f"\nRandom baseline statistics over {n_runs} runs:")
        print(f"  Mean: {stats['mean']:.6f} ± {stats['std']:.6f}")
        print(f"  Min:  {stats['min']:.6f}")
        print(f"  Max:  {stats['max']:.6f}")
        print(f"  Median: {stats['median']:.6f}")

        return stats
