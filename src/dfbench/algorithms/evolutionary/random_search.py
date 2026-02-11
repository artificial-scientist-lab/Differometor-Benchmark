import secrets

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

    All history tracking, printing, and checkpointing is handled by the
    `Objective` wrapper. The algorithm loop is minimal.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("random_search").
        algorithm_type (AlgorithmType): Type classification (EVOLUTIONARY).
        batch_size (int): Number of samples to evaluate in parallel per batch.

    Example:
        >>> from dfbench import Objective
        >>> from dfbench.problems import VoyagerProblem
        >>> problem = VoyagerProblem()
        >>> obj = Objective(problem, unbounded=False, max_time=120)
        >>> optimizer = RandomSearch(batch_size=100)
        >>> result = optimizer.optimize(
        ...     problem_objective=obj,
        ...     max_iterations=100,
        ... )
    """

    algorithm_str: str = "random_search"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self,
        batch_size: int = 100,
    ) -> None:
        """Initialize Random Search optimizer.

        Args:
            batch_size (int): Number of samples to evaluate in parallel per batch.
                Defaults to 100.
        """
        self.batch_size = batch_size

    def optimize(
        self,
        problem_objective: Objective,
        max_iterations: int | None = None,
        random_seed: int | None = None,
    ) -> Objective:
        """Run Random Search optimization.

        Args:
            problem_objective: The Objective instance wrapping the problem.
            max_iterations: Maximum number of batches to evaluate. If None, runs until budget exceeded.
            random_seed (int | None): Random seed for reproducibility. Defaults to None.

        Returns:
            The Objective instance with all logged data.
        """
        obj = problem_objective
        problem = obj.problem

        self.setup_objective(obj, unbounded=False, random_seed=random_seed)

        if random_seed is None:
            random_seed = secrets.randbits(32)
        obj.set_seed(random_seed)
        np.random.seed(random_seed)
        key = random.PRNGKey(random_seed)
        print(f"Random seed: {random_seed}")

        # Get bounds
        lower, upper = problem.bounds[0], problem.bounds[1]

        # Warmup JIT
        _ = obj.vmap_value(jnp.zeros((self.batch_size, problem.n_params)))

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

        return obj
