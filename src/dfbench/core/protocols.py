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
    GENERATIVE = "generative"


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
        - `optimize(...)`: Run optimization and return an Objective instance

        The returned Objective contains all run data:
        - best_params, best_params_bounded: Best parameters found
        - loss_history, params_history: Full optimization history
        - time_steps: Timestamps at each evaluation
        - Budget tracking: eval_count, time_elapsed, etc.
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
    def optimize(self, *args, **kwargs):
        """Run the optimization algorithm.

        Args:
            *args: Algorithm-specific positional arguments.
            **kwargs: Algorithm-specific keyword arguments (e.g., max_time,
                random_seed, learning_rate).

        Returns:
            Objective: Instance containing all optimization run data including
                best_params, loss_history, params_history, time_steps, and
                budget tracking state.
        """
        pass
