from abc import ABC, abstractmethod
from typing import Callable

from jaxtyping import Array, Float
from enum import Enum


class AlgorithmType(Enum):
    """Classification of optimization algorithm types.

    Used to categorize algorithms for benchmarking and comparison.

    Values:
        GRADIENT_BASED: Algorithms using gradient information (e.g., Adam, SA-GD).
        EVOLUTIONARY: Population-based algorithms (e.g., PSO, Random Search).
        SURROGATE_BASED: Algorithms using surrogate models (e.g., Bayesian Optimization).
        DIFFUSION_BASED: Generative diffusion-based optimization (experimental).
    """

    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    SURROGATE_BASED = "surrogate_based"
    DIFFUSION_BASED = "diffusion_based"


class ContinuousProblem(ABC):
    """Abstract base class for continuous optimization problems.

    Defines the interface that all optimization problems must implement.
    Problems provide objective functions and bounds for the parameter space.

    Attributes:
        name (str): Human-readable name for the problem.
        objective_function (Callable): Objective to minimize. Expects parameters
            in the BOUNDED space (within `bounds`). Used by evolutionary and
            surrogate-based algorithms that search directly in parameter space.
        sigmoid_objective_function (Callable): Objective for UNBOUNDED optimization.
            Expects parameters in (-∞, +∞) and applies sigmoid bounding internally.
            Used by gradient-based algorithms (Adam, SA-GD) for unconstrained search.

    Note:
        The two objective functions serve different optimization strategies:
        - `objective_function`: For bounded search (PSO, BO, Random Search)
        - `sigmoid_objective_function`: For unbounded search with internal bounding
          (gradient descent methods that work best in unconstrained space)

    Example:
        >>> class MyProblem(ContinuousProblem):
        ...     @property
        ...     def bounds(self):
        ...         return np.array([[0, 0], [1, 1]])  # 2 params in [0, 1]
        ...
        ...     @property
        ...     def optimization_pairs(self):
        ...         return [("component1", "param1"), ("component1", "param2")]
    """

    name: str

    objective_function: Callable[[Float[Array, "{self.n_params}"]], Float]

    sigmoid_objective_function: Callable[[Float[Array, "{self.n_params}"]], Float]

    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def bounds(
        self,
    ) -> Float[Array, "2 {self.n_params}"]:
        """Parameter bounds as [lower_bounds, upper_bounds].

        Returns:
            Array of shape (2, n_params) where bounds[0] are lower bounds
            and bounds[1] are upper bounds for each parameter.
        """
        pass

    @property
    @abstractmethod
    def optimization_pairs(
        self,
    ) -> list[tuple[str, str]]:
        """List of (component_name, property_name) tuples being optimized.

        Returns:
            List of tuples identifying each parameter in the optimization.
        """
        pass

    @property
    def n_params(self) -> int:
        """Number of parameters to optimize."""
        return len(self.optimization_pairs)

    def output_to_files(
        self,
        best_params: Float[Array, "{self.n_params}"] = None,
        losses: Float[Array, "iterations"] = None,
        population_losses: Float[Array, "iterations pop"] = None,
        algorithm_str: str = "",
        hyper_param_str: str = "",
        hyper_param_str_in_filename: bool = True,
    ) -> None:
        """Save optimization results to files.

        Args:
            best_params: Best parameters found during optimization.
            losses: Loss values at each iteration.
            population_losses: Per-member losses for population-based algorithms.
            algorithm_str: Algorithm identifier for filename.
            hyper_param_str: Hyperparameter string for filename.
            hyper_param_str_in_filename: Whether to include hyperparams in filename.
        """
        pass


class OptimizationAlgorithm(ABC):
    """Abstract base class for optimization algorithms.

    Defines the interface that all optimization algorithms must implement.

    Attributes:
        algorithm_str (str): Unique identifier string for the algorithm
            (e.g., "adam", "evox_pso", "botorch_bo").
        algorithm_type (AlgorithmType): Classification of algorithm type.
        _problem (ContinuousProblem): The optimization problem instance
            (conventionally stored with underscore prefix).

    Note:
        All algorithms must implement:
        - `__init__(problem, ...)`: Initialize with a problem instance
        - `optimize(...)`: Run optimization and return results

        The `optimize` method should return at minimum:
        - best_params: Best parameters found
        - best_params_history: History of best params (or None)
        - losses: Loss at each iteration
        - wall_time_indices: Indices at wall time checkpoints (or None)
    """

    algorithm_str: str
    algorithm_type: AlgorithmType

    @abstractmethod
    def __init__(self, problem: ContinuousProblem, *args, **kwargs):
        """Initialize the algorithm with an optimization problem.

        Args:
            problem: The continuous optimization problem to solve.
            *args: Algorithm-specific positional arguments.
            **kwargs: Algorithm-specific keyword arguments.
        """
        pass

    @abstractmethod
    def optimize(self, save_to_file: bool = True, *args, **kwargs):
        """Run the optimization algorithm.

        Args:
            save_to_file: Whether to save results to disk.
            *args: Algorithm-specific positional arguments.
            **kwargs: Algorithm-specific keyword arguments (e.g., wall_times,
                random_seed, return_best_params_history).

        Returns:
            tuple: At minimum (best_params, best_params_history, losses, wall_time_indices).
                Some algorithms may return additional values.
        """
        pass
