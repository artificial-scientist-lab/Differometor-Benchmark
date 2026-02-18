from abc import ABC, abstractmethod
from enum import Enum
import secrets

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from dfbench.core.objective import Objective


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
        - `__init__(problem, ...)`: Initialize with a problem instance
        - `optimize(...)`: Run optimization and return an Objective instance

        The returned Objective contains all run data:
        - `best_params`, `best_params_bounded`: Best parameters found
        - `loss_history`, `params_history`: Full optimization history
        - `time_steps`: Timestamps at each evaluation
        - Budget tracking: `eval_count`, `time_elapsed`, etc.
    """

    # Set this!
    algorithm_str: str
    algorithm_type: AlgorithmType

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
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        **kwargs
    ) -> Objective:
        """Run the optimization algorithm following the standard blueprint pattern.

        Subclasses must override this method and implement their algorithm-specific logic
        in step 6 (main optimization loop).

        Args:
            problem_objective: Pre-configured Objective instance for function evaluations.
            init_params: Initial parameters. If None, initialized randomly.
            random_seed: Random seed for reproducibility. If None, uses system entropy.
            **kwargs: Algorithm-specific hyperparameters (learning_rate, patience, etc.).
                Also add max_iterations (!= max_evals. These are algorithm-specific iterations) 
                or patience here.

        Returns:
            Objective instance containing complete optimization history.
        """
        # 1. Setup references
        obj = problem_objective
        problem = obj.problem
        
        self.setup_objective(obj, unbounded=False, random_seed=random_seed)
        
        # 2. Set random seed
        if random_seed is None:
            random_seed = secrets.randbits(32)  # Use system entropy for true randomness
        obj.set_seed(random_seed)
        np.random.seed(random_seed)
        key = jax.random.PRNGKey(random_seed)  # Set all package's seeds (np, jax, torch, etc.)
        print(f"Random seed: {random_seed}")  # Log for reproducibility
        
        # 3. Initialize parameters
        if init_params is None:
            # Bounded optimization
            params = obj.random_params_bounded()  # If n_samples = 1, returns shape (n_params,)
            # Unbounded optimization
            params = obj.random_params_unbounded()  # If unbounded = True was given to setup_objective()
            # Batched optimization
            batched_params = obj.random_params_bounded(n_samples=10) # Use self.batch_size from __init__() here
        else:
            params = init_params
        
        # 4. JIT warmup (optional but recommended, else much time is lost during the first evaluation)
        _ = obj.value(params)
        _ = obj.value_and_grad(params)  # For gradientients and loss
        _ = obj.grad(params)  # Loss won't get logged
            # Or for batched:
        _ = obj.vmap_value()
        _ = obj.vmap_value_and_grad()
        _ = obj.vmap_grad()
        
        # 5. Start logging
        obj.start_logging()
        
        # 6. Optimization
        # This tracks evals and time. Once exhausted the objective won't log anymore.
        # Obviously, you can add iteration-based stopping criteria here as well.
        
        # --------- Initializing algorithm logic here ---------
        
        while not obj.budget_exceeded:
            
            ... # --------- Looped algorithm logic here ---------
            # Loss printing is also done by the Objective. The frequency can be set in its __init__().

        # 7. Return the Objective instance (IMPORTANT -- contains all logged data)
        # What data is logged is decided by the user intializing the Objective.
        # Plotting and saving to file can be done afterwards by calling methods on the returned Objective.
        # If automatic file saving was enabled by the user, the data is already saved to file and can be loaded from there as well.
        # Please take a look at the Objective class documentation or docstring for a guide and details.
        return obj
    
    def setup_objective(
        self, 
        obj: Objective,
        unbounded: bool, 
        algorithm_str: str | None = None,
        random_seed: int | None = None, 
        **kwargs) -> None:
        """Helps set up the Objective with parameters that have to be set for all algorithms.
        Please decide if it should be unbounded or not and give a random_seed.
        Could also be done manually.

        Args:
            obj (Objective): The Objective instance to set up.
            unbounded (bool): Whether the algorithm needs unbounded parameter space.
            algorithm_str (str | None): Optional algorithm identifier. If None, uses self.algorithm_str.
            random_seed (int | None): Random seed given to the Objective for reproducable 
            random parameter generation across algorithms.

        Returns:
            None
        """
        obj.unbounded = unbounded
        if algorithm_str:
            obj.algorithm_str = algorithm_str
        elif self.algorithm_str:
            obj.algorithm_str = self.algorithm_str
        else:
            # If neither is provided, default to the class name in lowercase
            obj.algorithm_str = self.__class__.__name__.lower()
            
        obj.set_seed(random_seed)
        for k, v in kwargs.items():
            try:
                setattr(obj, k, v)
            except AttributeError:
                print(f"Warning: Objective has no attribute '{k}' to set with value {v}")
        return obj
