from abc import ABC, abstractmethod
from typing import Callable

from jaxtyping import Array, Float

from dfbench.core.search_space import SearchSpace


class ContinuousProblem(ABC):
    """Abstract base class for continuous optimization problems.

    Defines the interface that all optimization problems must implement.
    Problems provide objective functions and bounds for the parameter space.

    Attributes:
        name (str): Human-readable name for the problem.
        objective_function (Callable): Objective to minimize. Expects parameters
            in the BOUNDED space (within `bounds`). The Objective wrapper owns
            any mapping needed for algorithms that optimize in unbounded space.
    """

    name: str = "unkown_problem"

    objective_function: Callable[[Float[Array, "{self.n_params}"]], Float]

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
    def search_space(self) -> SearchSpace:
        """Explicit schema for this problem's implicit parameter domain.

        Returns:
            SearchSpace instance describing the parameter space.
        """
        pass

    @property
    def n_params(self) -> int:
        """Number of parameters to optimize."""
        return len(self.optimization_pairs)
