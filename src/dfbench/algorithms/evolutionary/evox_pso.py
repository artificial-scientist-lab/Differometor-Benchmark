import jax
import jax.numpy as jnp
import torch
from typing import Literal, get_args
from evox.algorithms import PSO, CLPSO, CSO, DMSPSOEL, FSPSO, SLPSOGS, SLPSOUS
from evox.core import Problem as EvoxProblem
from evox.workflows import EvalMonitor, StdWorkflow
from jaxtyping import Array, Float

from dfbench import (
    OptimizationAlgorithm,
    AlgorithmType,
    j2t,
    t2j,
)
from dfbench.core.objective import Objective


PSOVariant = Literal["PSO", "CLPSO", "CSO", "DMSPSOEL", "FSPSO", "SLPSOGS", "SLPSOUS"]


class EvoxPSO(OptimizationAlgorithm):
    """EvoX-based Particle Swarm Optimization algorithm.

    Implements PSO using the EvoX library with PyTorch backend. Handles batched
    evaluation of particles to manage memory efficiently. Supports multiple PSO
    variants through the variant parameter.

    All history tracking is handled by the `Objective` wrapper.

    Attributes:
        algorithm_str (str): Identifier string (e.g., "evox_pso", "evox_clpso").
        algorithm_type (AlgorithmType): Type classification (EVOLUTIONARY).
        _batch_size (int): Number of particles to evaluate per batch.
        _variant (str): PSO variant name (uppercase, e.g., "PSO", "CLPSO").

    Note:
        This algorithm uses `problem.objective_function` with the problem's bounds.
        The swarm searches directly in the bounded parameter space.

    Example:
        >>> problem = VoyagerProblem()
        >>> obj = Objective(problem, ...)
        >>> optimizer = EvoxPSO(batch_size=50, variant="CLPSO")
        >>> result = optimizer.optimize(
        ...     objective=obj,
        ...     max_iterations=1000,
        ...     pop_size=200,
        ... )
    """

    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self,
        batch_size: int = 1,
        variant: PSOVariant = "PSO",
    ) -> None:
        """Initialize EvoX Particle Swarm Optimization.

        Args:
            batch_size (int): Number of particles to evaluate simultaneously in each batch.
                Reduce this value if encountering out-of-memory errors. Defaults to 1.
            variant (PSOVariant): PSO variant to use. Options:
                - 'PSO': Standard Particle Swarm Optimization (default)
                - 'CLPSO': Comprehensive Learning PSO
                - 'CSO': Competitive Swarm Optimizer
                - 'DMSPSOEL': Dynamic Multi-Swarm PSO with Elite Learning
                - 'FSPSO': Fitness-Sharing PSO
                - 'SLPSOGS': Social Learning PSO with Gaussian Sampling
                - 'SLPSOUS': Social Learning PSO with Uniform Sampling
                Defaults to 'PSO'.
        """
        self._batch_size = batch_size
        self._variant: PSOVariant = variant.upper()  # type: ignore[assignment]

        # Validate variant at runtime
        valid_variants = get_args(PSOVariant)
        if self._variant not in valid_variants:
            raise ValueError(
                f"Unknown PSO variant: '{variant}'. "
                f"Valid options are: {', '.join(valid_variants)}"
            )

        # Set algorithm_str based on variant
        self.algorithm_str = f"evox_{self._variant.lower()}"

    def optimize(
        self,
        objective: Objective,
        max_iterations: int | None = None,
        init_params_pop: Float[Array, "pop_size n_params"] | None = None,
        random_seed: int | None = None,
        pop_size: int = 100,
        n_generations: int = 10000,
        **pso_kwargs,
    ) -> None:
        """Run PSO optimization.

        Args:
            objective: The Objective instance wrapping the problem.
            max_iterations: Maximum number of iterations (generations). None for unlimited.
            init_params_pop: Initial population of parameters. If None, randomly
                initialized within bounds. Defaults to None.
            random_seed: Random seed for reproducibility. Defaults to None.
            pop_size: Number of particles in the swarm. Defaults to 100.
            n_generations: Number of generations to run. Defaults to 10000.
            **pso_kwargs: Variant-specific keyword arguments passed to the EvoX algorithm.
        """
        obj = objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)

        # Get bounds from problem
        if not hasattr(problem, "bounds"):
            raise ValueError(
                f"Problem {type(problem).__name__} must have a 'bounds' attribute."
            )
        problem_bounds = problem.bounds
        lb = j2t(problem_bounds[0])
        ub = j2t(problem_bounds[1])

        # Define the problem in EvoX that delegates to Objective
        batch_size = self._batch_size

        class PSOProblem(EvoxProblem):
            def __init__(self, objective: Objective):
                super().__init__()
                self._obj = objective
                self.batch_size = batch_size

            def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
                jpop = t2j(pop)
                n_particles = jpop.shape[0]
                all_losses = []

                for i in range(0, n_particles, self.batch_size):
                    batch = jpop[i : i + self.batch_size]
                    batch_losses = self._obj.vmap_value(batch)
                    all_losses.append(batch_losses)

                losses = jnp.concatenate(all_losses, axis=0)
                return j2t(losses)

        # Warmup JIT
        obj.warmup_vmap_value(batch_size=self._batch_size)

        pso_problem = PSOProblem(obj)

        # Map variant names to algorithm classes
        variant_map = {
            "PSO": PSO,
            "CLPSO": CLPSO,
            "CSO": CSO,
            "DMSPSOEL": DMSPSOEL,
            "FSPSO": FSPSO,
            "SLPSOGS": SLPSOGS,
            "SLPSOUS": SLPSOUS,
        }

        # Initiate algorithm
        AlgorithmClass = variant_map[self._variant]
        algorithm = AlgorithmClass(pop_size=pop_size, lb=lb, ub=ub, **pso_kwargs)

        # If initial population is provided
        if init_params_pop is not None:
            if isinstance(init_params_pop, jax.Array):
                init_pop_torch = j2t(init_params_pop)
            else:
                init_pop_torch = init_params_pop
            algorithm.pop = init_pop_torch

        # Create workflow
        monitor = EvalMonitor()
        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=pso_problem,
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
