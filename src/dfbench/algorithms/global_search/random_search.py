from __future__ import annotations

from jaxtyping import Float, Array

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.search_space import SearchSpace
from dfbench.core import ParameterConfig


class RandomSearch(OptimizationAlgorithm):
    """Random Search optimization algorithm.

    Samples random parameters uniformly within the problem's bounds and evaluates them.
    Useful as a baseline for comparing more sophisticated optimization algorithms.

    Attributes:
        algorithm_str: Identifier string for this algorithm ("random_search").
        algorithm_type: Type classification (EVOLUTIONARY).
        seed: Random seed for reproducibility.
    """

    algorithm_str: str = "random_search"
    algorithm_type: AlgorithmType = AlgorithmType.GLOBAL_SEARCH

    def __init__(
        self,
        problem_space: SearchSpace,
        seed: int | None = None,
        init_params: ParameterConfig | None = None,
        **kwargs,
    ) -> None:
        """Initialize Random Search optimizer.

        Args:
            batch_size: Number of samples to evaluate in parallel per batch.
                Defaults to 1.
            seed: Random seed for reproducibility. Defaults to None.
        """
        self.seed = seed
        self.space = problem_space

    def ask(
        self,
        n_samples: int = 1,
    ) -> ParameterConfig:
        """Sample a random parameter configuration from the search space.

        Returns:
            A ParameterConfig object containing the sampled parameters.
        """
        return ParameterConfig.from_values(
            values=self.space.sample(n=n_samples),
            search_space=self.space,
            unbounded=False,
        )

    def tell(
        self,
        params: Float[Array, "..."],
        loss: Float[Array, "..."] | Float,
    ):
        pass
