from abc import ABC, abstractmethod
from enum import Enum
import secrets

import jax
import numpy as np
from jaxtyping import Array, Float

from dfbench.core.objective import Objective


class AlgorithmType(Enum):
    """Classification of optimization algorithm types.

    Used to categorize algorithms for benchmarking and comparison.

    Values:
        GRADIENT_BASED: Algorithms using gradient information (e.g., Adam, SA-GD).
        EVOLUTIONARY: Population-based algorithms (e.g., PSO, Random Search).
        DERIVATIVE_FREE: Direct-search and local derivative-free algorithms.
        GLOBAL_SEARCH: Stochastic global-search algorithms.
        SURROGATE_BASED: Algorithms using surrogate models (e.g., Bayesian Optimization).
        GENERATIVE: Generative-model-based optimization algorithms.
    """

    GRADIENT_BASED = "gradient_based"
    EVOLUTIONARY = "evolutionary"
    SURROGATE_BASED = "surrogate_based"
    GLOBAL_SEARCH = "global_search"
    DERIVATIVE_FREE = "derivative_free"
    GENERATIVE = "generative"


class OptimizationAlgorithm(ABC):
    """Abstract base class for optimization algorithms.

    Defines the interface that all optimization algorithms must implement.
    Algorithm blueprint in `optimize()`.

    Attributes:
        algorithm_str (str): Unique identifier string for the algorithm
            (e.g., "adam", "evox_pso", "botorch_bo").
        algorithm_type (AlgorithmType): Classification of algorithm type.
        _problem (ContinuousProblem): The optimization problem instance
            (conventionally stored with underscore prefix).

    Note:
        All algorithms must implement:
        - `__init__(...)`: Initialize algorithm-specific meta-parameters
        - `optimize(...)`: Run optimization and mutate the provided Objective

        The returned Objective contains all run data:
        - `best_params`, `best_params_bounded`: Best parameters found
        - `loss_history`, `params_history`: Full optimization history
        - `time_steps`: Timestamps at each evaluation
        - Budget tracking: `eval_count`, `time_elapsed`, etc.
    """

    # Set this!
    algorithm_str: str
    algorithm_type: AlgorithmType

    _random_seed: int | None = None

    @abstractmethod
    def __init__(self, **kwargs):
        """Initialize the algorithm with an optimization problem.

        Args:
            **kwargs: Algorithm-specific meta-parameters like `batch_size` for vmapping.
        """
        pass

    @abstractmethod
    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        **kwargs,
    ) -> None:
        """Run the optimization algorithm following the standard blueprint pattern.

        Subclasses must override this method and implement their algorithm-specific logic
        in step 6 (main optimization loop).

        The Objective instance is mutated in place — all logged data (losses, params,
        gradients, timestamps) is recorded directly into it. There is no return value;
        the caller already holds the reference.

        Args:
            objective: Pre-configured Objective instance for function evaluations.
            init_params: Initial parameters. If None, initialized randomly.
            random_seed: Random seed for reproducibility. If None, uses system entropy.
            **kwargs: Algorithm-specific hyperparameters (learning_rate, patience, etc.).
                For algorithms where one iteration != one evaluation (population-based,
                surrogate-based, etc.), add ``max_iterations`` here to cap algorithm
                iterations independently of the eval budget. For single-eval-per-step
                algorithms (e.g. Adam, L-BFGS), ``max_iterations`` is redundant with
                ``max_evals`` on the Objective and should not be added.
        """
        # 1. Setup references
        obj = objective
        problem = obj.problem  # noqa: F841, just for clarity and convention in the algorithm code

        # 2. Setup objective and resolve/apply random seed
        random_seed, key = self.prepare(obj, unbounded=False, random_seed=random_seed)
        # For frameworks that also need seeding:
        # torch.manual_seed(random_seed)

        # 3. Initialize parameters
        if init_params is None:
            # Bounded optimization
            params = (
                obj.random_params_bounded()
            )  # If n_samples = 1, returns shape (n_params,)
            # Unbounded optimization
            params = (
                obj.random_params_unbounded()
            )  # If unbounded = True was given to prepare()
            # Batched optimization
            batched_params = obj.random_params_bounded(  # noqa: F841, just for convention in the algorithm code
                n_samples=10
            )  # Use self.batch_size from __init__() here
        else:
            params = init_params  # noqa: F841, just for convention in the algorithm code

        # 4. JIT warmup (optional but recommended, else much time is lost during
        # the first evaluation). Warm up only the paths your algorithm will use.
        obj.warmup_value()  # loss-only single-point evaluation
        obj.warmup_value_and_grad()  # single-point loss + gradient
        obj.warmup_grad()  # gradient-only single-point evaluation
        # Or for batched algorithms:
        obj.warmup_vmap_value(batch_size=10)
        obj.warmup_vmap_value_and_grad(batch_size=10)
        obj.warmup_vmap_grad(batch_size=10)

        # 5. Start logging
        obj.start_logging()

        # 6. Optimization
        # This tracks evals and time. Once exhausted the objective won't log anymore.
        # Obviously, you can add iteration-based stopping criteria here as well.

        # --------- Initializing algorithm logic here ---------

        while not obj.budget_exceeded:
            ...  # --------- Looped algorithm logic here ---------
            # Loss printing is also done by the Objective. The frequency can be set in its __init__().

        # 7. Done — the Objective instance now contains all logged data.
        # What data is logged is decided by the user initializing the Objective.
        # Plotting and saving to file can be done afterwards by calling methods on the Objective.
        # If automatic file saving was enabled by the user, the data is already saved to file and can be loaded from there as well.
        # Please take a look at the Objective class documentation or docstring for a guide and details.

    def prepare(
        self,
        obj: Objective,
        unbounded: bool,
        random_seed: int | None = None,
        algorithm_str: str | None = None,
        **kwargs,
    ) -> tuple[int, jax.Array]:
        """Set up the Objective and resolve/apply the random seed.

        Configures `obj.unbounded`, `obj.algorithm_str`, and seeds all relevant
        RNG sources (`np.random`, JAX). Call this at the top of `optimize()`
        instead of managing seed resolution manually.

        Args:
            obj (Objective): The Objective instance to set up.
            unbounded (bool): Whether the algorithm needs unbounded parameter space.
            random_seed (int | None): Seed for reproducibility. If None, one is generated
                via system entropy and stored in self._random_seed.
            algorithm_str (str | None): Optional algorithm identifier. If None, uses self.algorithm_str.
            **kwargs: Additional Objective attributes to set.

        Returns:
            tuple[int, jax.Array]: Resolved integer seed and a JAX PRNGKey.
                Use the seed for framework-specific seeding (e.g. torch.manual_seed)
                and the key for JAX-based random operations.
        """
        obj.set_space_mode(unbounded)
        if algorithm_str:
            obj.algorithm_str = algorithm_str
        elif self.algorithm_str:
            obj.algorithm_str = self.algorithm_str
        else:
            # If neither is provided, default to the class name in lowercase
            obj.algorithm_str = self.__class__.__name__.lower()

        if random_seed is None:
            random_seed = secrets.randbits(32)
        self._random_seed = random_seed
        obj.set_seed(random_seed)
        np.random.seed(random_seed)
        key = jax.random.PRNGKey(random_seed)
        print(f"Random seed: {random_seed}")

        for k, v in kwargs.items():
            try:
                setattr(obj, k, v)
            except AttributeError:
                print(
                    f"Warning: Objective has no attribute '{k}' to set with value {v}"
                )
        return random_seed, key
