from abc import ABC, abstractmethod
from typing import Callable

from jaxtyping import Array, Float


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
