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
    j2t_numpy as j2t,
    t2j_numpy as t2j,
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
        _problem (ContinuousProblem): The optimization problem instance.
        _batch_size (int): Number of individuals to evaluate per batch.
        _variant (str): ES variant name (e.g., "CMAES", "OpenES").

    Note:
        This algorithm uses `problem.objective_function` with the problem's bounds.
        The population searches directly in the bounded parameter space.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = EvoxES(problem, batch_size=50, variant="CMAES")
        >>> objective = optimizer.optimize(
        ...     pop_size=100,
        ...     max_time=120,
        ... )
    """

    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self,
        problem: ContinuousProblem,
        batch_size: int = 5,
        variant: ESVariant = "CMAES",
        verbose: int = 0,
        save_params_history: bool = True,
        save_batched_losses: bool = True,
        save_batched_params: bool = False,
    ) -> None:
        """Initialize EvoX Evolution Strategy.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
            batch_size (int): Number of individuals to evaluate simultaneously in each batch.
                Reduce this value if encountering out-of-memory errors. Defaults to 5.
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
            verbose (int): Verbosity level (0=silent, 1+=prints). Defaults to 0.
            save_params_history: Whether to save parameter history. Defaults to True.
            save_batched_losses: Whether to save full batched losses (vs reduced).
                Defaults to True for detailed analysis.
            save_batched_params: Whether to save full batched params (memory heavy).
                Defaults to False.
        """
        self._problem = problem
        self._batch_size = batch_size
        self._variant: ESVariant = variant
        self._verbose = verbose
        self._save_params_history = save_params_history
        self._save_batched_losses = save_batched_losses
        self._save_batched_params = save_batched_params

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
        init_params_pop: Float[Array, "{pop_size} {self._problem.n_params}"]
        | None = None,
        random_seed: int | None = None,
        max_time: float | None = None,
        pop_size: int = 100,
        n_generations: int = 10000,
        verbose: int | None = None,
        print_every: int = 100,
        plot_loss: bool = False,
        save_run_to_file: bool = False,
        **es_kwargs,
    ) -> Objective:
        """Run ES optimization.

        Args:
            init_params_pop: Initial population of parameters. Not supported by most
                ES variants (mean is typically initialized instead). Defaults to None.
            random_seed: Random seed for reproducibility. Defaults to None.
            max_time: Time budget in seconds. None for unlimited.
            pop_size: Number of individuals in the population. Defaults to 100.
            n_generations: Number of generations to run. Defaults to 10000.
            verbose: Verbosity level (0=silent, 1+=prints via Objective).
            print_every: Print summary every N evaluations.
            plot_loss: If True, call obj.output_to_files for plotting.
            save_run_to_file: If True, call obj.save_run_data for checkpointing.
            **es_kwargs: Variant-specific keyword arguments passed to the EvoX algorithm.

        Returns:
            The Objective instance with all logged data.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # Get bounds from problem
        if not hasattr(self._problem, "bounds"):
            raise ValueError(
                f"Problem {type(self._problem).__name__} must have a 'bounds' attribute."
            )
        problem_bounds = self._problem.bounds
        lb_np = np.asarray(problem_bounds[0])
        ub_np = np.asarray(problem_bounds[1])

        # Create Objective wrapper
        obj = Objective(
            self._problem,
            unbounded=False,
            max_time=max_time,
            max_evals=n_generations * pop_size,
            save_params_history=self._save_params_history,
            save_batched_losses_history=self._save_batched_losses,
            save_batched_history=self._save_batched_params,
            print_every=print_every,
            verbose=verbose if verbose is not None else self._verbose,
            algorithm_str=self.algorithm_str,
        )

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
        if self._verbose >= 1:
            print(f"Warming up JIT compilation...")
        _ = obj.vmap_value(jnp.zeros((self._batch_size, self._problem.n_params)))

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
                np.random.uniform(lb_np, ub_np, size=self._problem.n_params)
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
                es_kwargs["init_covar"] = torch.eye(self._problem.n_params).float() * (
                    default_sigma**2
                )
            algorithm = AlgorithmClass(pop_size=pop_size, **es_kwargs)

        elif self._variant == "SeparableNES":
            if "init_mean" not in es_kwargs:
                es_kwargs["init_mean"] = get_center_init()
            if "init_std" not in es_kwargs:
                es_kwargs["init_std"] = (
                    torch.ones(self._problem.n_params).float() * default_sigma
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
        gen = 0
        while not obj.budget_exceeded and gen < n_generations:
            workflow.step()
            gen += 1

        # Outputs
        if plot_loss:
            obj.output_to_files()
        if save_run_to_file:
            obj.save_run_data()

        return obj
