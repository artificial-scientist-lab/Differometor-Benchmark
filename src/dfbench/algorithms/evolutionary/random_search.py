import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float

from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)
from dfbench.core.objective import Objective


class RandomSearch(OptimizationAlgorithm):
    """Random Search optimization algorithm.

    Samples random parameters uniformly within the problem's bounds and evaluates them.
    Useful as a baseline for comparing more sophisticated optimization algorithms.

    All history tracking, printing, and checkpointing is handled by the
    `Objective` wrapper. The algorithm loop is minimal.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("random_search").
        algorithm_type (AlgorithmType): Type classification (EVOLUTIONARY).
        _problem (ContinuousProblem): The optimization problem instance.
        _batch_size (int): Number of samples to evaluate in parallel per batch.

    Note:
        This algorithm requires the problem to have a `bounds` attribute and uses
        `problem.objective_function` which expects bounded parameters.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = RandomSearch(problem, batch_size=100)
        >>> objective = optimizer.optimize(
        ...     n_samples=10000,
        ...     max_time=120,
        ... )
    """

    algorithm_str: str = "random_search"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self,
        problem: ContinuousProblem,
        batch_size: int = 100,
        verbose: int = 0,
        save_params_history: bool = True,
        save_batched_losses: bool = True,
        save_batched_params: bool = False,
    ) -> None:
        """Initialize Random Search optimizer.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
                Must have a `bounds` attribute (shape [2, n_params]).
            batch_size (int): Number of samples to evaluate in parallel per batch.
                Defaults to 100.
            verbose (int): Verbosity level (0=silent, 1+=prints). Defaults to 0.
            save_params_history: Whether to save parameter history. Defaults to True.
            save_batched_losses: Whether to save full batched losses (vs reduced).
                Defaults to True for detailed analysis.
            save_batched_params: Whether to save full batched params (memory heavy).
                Defaults to False.
        """
        self._problem = problem
        self._batch_size = batch_size
        self._verbose = verbose
        self._save_params_history = save_params_history
        self._save_batched_losses = save_batched_losses
        self._save_batched_params = save_batched_params

        # Validate that the problem has bounds
        if not hasattr(problem, "bounds"):
            raise ValueError(
                "RandomSearch requires the problem to have a 'bounds' attribute. "
                "The bounds should be a numpy array of shape [2, n_params] with "
                "[lower_bounds, upper_bounds]."
            )

    def optimize(
        self,
        random_seed: int | None = None,
        max_time: float | None = None,
        n_samples: int = 10000,
        verbose: int | None = None,
        print_every: int = 100,
        plot_loss: bool = False,
        save_run_to_file: bool = False,
    ) -> Objective:
        """Run Random Search optimization.

        Args:
            random_seed (int | None): Random seed for reproducibility. Defaults to None.
            max_time (float | None): Time budget in seconds. None for unlimited.
            n_samples (int): Total number of random samples to evaluate. Defaults to 10000.
            verbose: Verbosity level (0=silent, 1+=prints via Objective).
            print_every: Print summary every N evaluations.
            plot_loss: If True, call obj.output_to_files for plotting.
            save_run_to_file: If True, call obj.save_run_data for checkpointing.

        Returns:
            The Objective instance with all logged data.
        """
        # Set random seed
        seed = random_seed if random_seed is not None else 0
        key = random.PRNGKey(seed)

        # Get bounds
        lower, upper = self._problem.bounds[0], self._problem.bounds[1]

        obj = Objective(
            self._problem,
            unbounded=False,
            max_time=max_time,
            max_evals=n_samples,
            save_params_history=self._save_params_history,
            save_batched_losses_history=self._save_batched_losses,
            save_batched_history=self._save_batched_params,
            print_every=print_every,
            verbose=verbose if verbose is not None else self._verbose,
            algorithm_str=self.algorithm_str,
        )

        # Warmup JIT
        if self._verbose >= 1:
            print(f"Warming up JIT compilation...")
        _ = obj.vmap_value(jnp.zeros((self._batch_size, self._problem.n_params)))

        obj.start_logging()

        while not obj.budget_exceeded:
            # Generate random samples
            key, subkey = random.split(key)
            random_params = random.uniform(
                subkey,
                shape=(self._batch_size, self._problem.n_params),
                minval=lower,
                maxval=upper,
            )

            # Evaluate batch
            losses = obj.vmap_value(random_params)

        # Outputs
        if plot_loss:
            obj.output_to_files()
        if save_run_to_file:
            obj.save_run_data()

        return obj

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
            obj = self.optimize(
                random_seed=seed,
                n_samples=n_samples,
            )
            run_mean = float(jnp.mean(jnp.array(obj.loss_history)))
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

        print(f"\nBaseline statistics:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std:  {stats['std']:.6f}")
        print(f"  Min:  {stats['min']:.6f}")
        print(f"  Max:  {stats['max']:.6f}")

        return stats
