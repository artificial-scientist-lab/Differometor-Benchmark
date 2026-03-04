import jax
import jax.numpy as jnp
import numpy as np
import torch
from typing import Literal, get_args
from evox.algorithms import (
    OpenES,
    XNES,
    SeparableNES,
    DES,
    SNES,
    ARS,
    ASEBO,
    PersistentES,
    NoiseReuseES,
    GuidedES,
    ESMC,
    CMAES,
)
from evox.core import Problem as EvoxProblem
from evox.workflows import EvalMonitor, StdWorkflow
from jaxtyping import Array, Float

from dfbench import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
    j2t,
    t2j,
)
from dfbench.core.objective import Objective


ESVariant = Literal[
    "OpenES",
    "XNES",
    "SeparableNES",
    "DES",
    "SNES",
    "ARS",
    "ASEBO",
    "PersistentES",
    "NoiseReuseES",
    "GuidedES",
    "ESMC",
    "CMAES",
]


class EvoxES(OptimizationAlgorithm):
    """EvoX-based Evolution Strategy algorithm.

    Implements Evolution Strategies using the EvoX library with PyTorch backend.
    Handles batched evaluation of population to manage memory efficiently.
    Supports multiple ES variants through the variant parameter.

    All history tracking is handled by the `Objective` wrapper.

    Attributes:
        algorithm_str (str): Identifier string (e.g., "evox_cmaes", "evox_openes").
        algorithm_type (AlgorithmType): Type classification (EVOLUTIONARY).
        _batch_size (int): Number of individuals to evaluate per batch.
        _variant (str): ES variant name (e.g., "CMAES", "OpenES").

    Note:
        This algorithm uses `problem.objective_function` with the problem's bounds.
        The population searches directly in the bounded parameter space.

    Example:
        >>> problem = VoyagerProblem()
        >>> obj = Objective(problem, ...)
        >>> optimizer = EvoxES(batch_size=50, variant="CMAES")
        >>> result = optimizer.optimize(
        ...     problem_objective=obj,
        ...     max_iterations=1000,
        ...     pop_size=100,
        ... )
    """

    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self,
        batch_size: int = 1,
        variant: ESVariant = "CMAES",
    ) -> None:
        """Initialize EvoX Evolution Strategy.

        Args:
            batch_size (int): Number of individuals to evaluate simultaneously in each batch.
                Reduce this value if encountering out-of-memory errors. Defaults to 1.
            variant (ESVariant): ES variant to use. Options:
                - 'CMAES': Covariance Matrix Adaptation Evolution Strategy (default)
                - 'OpenES': OpenAI Evolution Strategy
                - 'XNES': Exponential Natural Evolution Strategy
                - 'SeparableNES': Separable Natural Evolution Strategy
                - 'DES': Distributed Evolution Strategy
                - 'SNES': Separable Natural Evolution Strategy
                - 'ARS': Augmented Random Search
                - 'ASEBO': Adaptive Sampling Evolution-Based Optimization
                - 'PersistentES': Persistent Evolution Strategy
                - 'NoiseReuseES': Noise Reuse Evolution Strategy
                - 'GuidedES': Guided Evolution Strategy
                - 'ESMC': Evolution Strategy with Monte Carlo
                Defaults to 'CMAES'.
        """
        self._batch_size = batch_size
        self._variant: ESVariant = variant

        # Validate variant at runtime
        valid_variants = get_args(ESVariant)
        if self._variant not in valid_variants:
            raise ValueError(
                f"Unknown ES variant: '{variant}'. "
                f"Valid options are: {', '.join(valid_variants)}"
            )

        # Set algorithm_str based on variant
        self.algorithm_str = f"evox_{self._variant.lower()}"

    def optimize(
        self,
        problem_objective: Objective,
        max_iterations: int | None = None,
        init_params_pop: Float[Array, "pop_size n_params"] | None = None,
        random_seed: int | None = None,
        pop_size: int = 100,
        n_generations: int = 10000,
        **es_kwargs,
    ) -> None:
        """Run ES optimization.

        Args:
            problem_objective: The Objective instance wrapping the problem.
            max_iterations: Maximum number of iterations (generations). None for unlimited.
            init_params_pop: Initial population of parameters. Not supported by most
                ES variants (mean is typically initialized instead). Defaults to None.
            random_seed: Random seed for reproducibility. Defaults to None.
            pop_size: Number of individuals in the population. Defaults to 100.
            n_generations: Number of generations to run. Defaults to 10000.
            **es_kwargs: Variant-specific keyword arguments passed to the EvoX algorithm.
        """
        obj = problem_objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)

        # Get bounds from problem
        if not hasattr(problem, "bounds"):
            raise ValueError(
                f"Problem {type(problem).__name__} must have a 'bounds' attribute."
            )
        problem_bounds = problem.bounds
        lb_np = np.asarray(problem_bounds[0])
        ub_np = np.asarray(problem_bounds[1])

        # Define the problem in EvoX that delegates to Objective
        batch_size = self._batch_size

        class ESProblem(EvoxProblem):
            def __init__(self, objective: Objective):
                super().__init__()
                self._obj = objective
                self.batch_size = batch_size

            def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
                jpop = t2j(pop)
                n_individuals = jpop.shape[0]
                all_losses = []

                for i in range(0, n_individuals, self.batch_size):
                    batch = jpop[i : i + self.batch_size]
                    batch_losses = self._obj.vmap_value(batch)
                    all_losses.append(batch_losses)

                losses = jnp.concatenate(all_losses, axis=0)
                return j2t(losses).float()

        # Warmup JIT
        _ = obj.vmap_value(jnp.zeros((self._batch_size, problem.n_params)))

        es_problem = ESProblem(obj)

        # Map variant names to algorithm classes
        variant_map = {
            "CMAES": CMAES,
            "OpenES": OpenES,
            "XNES": XNES,
            "SeparableNES": SeparableNES,
            "DES": DES,
            "SNES": SNES,
            "ARS": ARS,
            "ASEBO": ASEBO,
            "PersistentES": PersistentES,
            "NoiseReuseES": NoiseReuseES,
            "GuidedES": GuidedES,
            "ESMC": ESMC,
        }

        AlgorithmClass = variant_map[self._variant]

        # Default sigma value for most variants
        default_sigma = float(np.mean(0.3 * (ub_np - lb_np)))

        def get_center_init():
            return torch.from_numpy(
                np.random.uniform(lb_np, ub_np, size=problem.n_params)
            ).float()

        # Initialize based on variant-specific requirements
        if self._variant == "CMAES":
            if "mean_init" not in es_kwargs:
                es_kwargs["mean_init"] = get_center_init()
            if "sigma" not in es_kwargs:
                es_kwargs["sigma"] = default_sigma
            algorithm = AlgorithmClass(pop_size=pop_size, **es_kwargs)

        elif self._variant == "OpenES":
            if "center_init" not in es_kwargs:
                es_kwargs["center_init"] = get_center_init()
            if "learning_rate" not in es_kwargs:
                es_kwargs["learning_rate"] = 0.05
            if "noise_stdev" not in es_kwargs:
                es_kwargs["noise_stdev"] = default_sigma
            algorithm = AlgorithmClass(pop_size=pop_size, **es_kwargs)

        elif self._variant == "SNES":
            if "center_init" not in es_kwargs:
                es_kwargs["center_init"] = get_center_init()
            if "sigma" not in es_kwargs:
                es_kwargs["sigma"] = default_sigma
            algorithm = AlgorithmClass(pop_size=pop_size, **es_kwargs)

        elif self._variant == "XNES":
            if "init_mean" not in es_kwargs:
                es_kwargs["init_mean"] = get_center_init()
            if "init_covar" not in es_kwargs:
                es_kwargs["init_covar"] = torch.eye(problem.n_params).float() * (
                    default_sigma**2
                )
            algorithm = AlgorithmClass(pop_size=pop_size, **es_kwargs)

        elif self._variant == "SeparableNES":
            if "init_mean" not in es_kwargs:
                es_kwargs["init_mean"] = get_center_init()
            if "init_std" not in es_kwargs:
                es_kwargs["init_std"] = (
                    torch.ones(problem.n_params).float() * default_sigma
                )
            algorithm = AlgorithmClass(pop_size=pop_size, **es_kwargs)

        elif self._variant == "DES":
            if "center_init" not in es_kwargs:
                es_kwargs["center_init"] = get_center_init()
            if "sigma_init" not in es_kwargs:
                es_kwargs["sigma_init"] = default_sigma
            algorithm = AlgorithmClass(pop_size=pop_size, **es_kwargs)

        elif self._variant in [
            "ARS",
            "ASEBO",
            "PersistentES",
            "NoiseReuseES",
            "GuidedES",
            "ESMC",
        ]:
            if "center_init" not in es_kwargs:
                es_kwargs["center_init"] = get_center_init()
            if "sigma" not in es_kwargs:
                es_kwargs["sigma"] = default_sigma
            algorithm = AlgorithmClass(pop_size=pop_size, **es_kwargs)

        else:
            raise ValueError(f"Unknown ES variant: {self._variant}")

        # If initial population is provided
        if init_params_pop is not None:
            if isinstance(init_params_pop, jax.Array):
                init_pop_torch = j2t(init_params_pop)
            else:
                init_pop_torch = init_params_pop
            if hasattr(algorithm, "pop"):
                algorithm.pop = init_pop_torch

        # Create workflow
        monitor = EvalMonitor()
        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=es_problem,
            monitor=monitor,
        )

        # Initialize workflow BEFORE starting the timer
        # This does JIT compilation and initial population evaluation
        # We start logging AFTER this so init overhead doesn't count against time budget
        workflow.init_step()

        obj.start_logging()

        # Run generations
        iteration = 0
        while not obj.budget_exceeded and iteration < n_generations:
            if max_iterations is not None and iteration >= max_iterations:
                break
            workflow.step()
            iteration += 1
