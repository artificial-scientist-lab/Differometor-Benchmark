import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


class RandomSearch(OptimizationAlgorithm):
    """Random Search optimization algorithm.

    Samples random parameters uniformly within the problem's bounds and evaluates them.
    Useful as a baseline for comparing more sophisticated optimization algorithms.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("random_search").
        algorithm_type (AlgorithmType): Type classification (EVOLUTIONARY).
        batch_size (int): Number of samples to evaluate in parallel per batch.

    Example:
        >>> from dfbench import Objective
        >>> from dfbench.problems import VoyagerProblem
        >>> problem = VoyagerProblem()
        >>> obj = Objective(problem, unbounded=False, max_time=120)
        >>> optimizer = RandomSearch(batch_size=1)
        >>> result = optimizer.optimize(
        ...     problem_objective=obj,
        ...     max_iterations=100,
        ... )
    """

    algorithm_str: str = "random_search"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self,
        batch_size: int = 1,
    ) -> None:
        """Initialize Random Search optimizer.

        Args:
            batch_size (int): Number of samples to evaluate in parallel per batch.
                Defaults to 1.
        """
        self.batch_size = batch_size

    def optimize(
        self,
        problem_objective: Objective,
        max_iterations: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Run Random Search optimization.

        Args:
            problem_objective: The Objective instance wrapping the problem.
            max_iterations: Maximum number of batches to evaluate. If None, runs until budget exceeded.
            random_seed (int | None): Random seed for reproducibility. Defaults to None.
        """
        obj = problem_objective
        problem = obj.problem

        random_seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)

        # Get bounds
        lower, upper = problem.bounds[0], problem.bounds[1]

        # Warmup JIT
        obj.warmup_vmap_value(batch_size=self.batch_size)

        obj.start_logging()

        iteration = 0
        while not obj.budget_exceeded:
            if max_iterations is not None and iteration >= max_iterations:
                break

            # Generate random samples
            key, subkey = random.split(key)
            random_params = random.uniform(
                subkey,
                shape=(self.batch_size, problem.n_params),
                minval=lower,
                maxval=upper,
            )

            # Evaluate batch
            losses = obj.vmap_value(random_params)

            iteration += 1
