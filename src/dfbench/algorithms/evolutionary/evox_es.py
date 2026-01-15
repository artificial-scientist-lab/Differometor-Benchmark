import jax
import jax.numpy as jnp
import numpy as np
import torch
import time
from collections import deque
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
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from dfbench import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
    j2t_numpy as j2t,
    t2j_numpy as t2j,
)


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

    Attributes:
        algorithm_str (str): Identifier string (e.g., "evox_cmaes", "evox_openes").
        algorithm_type (AlgorithmType): Type classification (EVOLUTIONARY).
        _problem (ContinuousProblem): The optimization problem instance.
        _batch_size (int): Number of individuals to evaluate per batch.
        _variant (str): ES variant name (e.g., "CMAES", "OpenES").
        _es_problem (EvoxProblem): EvoX problem wrapper for the objective function.

    Note:
        This algorithm uses `problem.objective_function` with the problem's bounds.
        The population searches directly in the bounded parameter space.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = EvoxES(problem, batch_size=50, variant="CMAES")
        >>> best_params, history, losses, wall_indices, pop_losses = optimizer.optimize(
        ...     pop_size=100,
        ...     wall_times=[30, 60, 120],
        ... )
    """

    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self,
        problem: ContinuousProblem,
        batch_size: int = 5,
        variant: ESVariant = "CMAES",
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
        """
        self._problem = problem
        self._batch_size = batch_size
        self._variant: ESVariant = variant  # type: ignore[assignment]

        # Validate variant at runtime (Literal provides static checking)
        valid_variants = get_args(ESVariant)
        if self._variant not in valid_variants:
            raise ValueError(
                f"Unknown ES variant: '{variant}'. "
                f"Valid options are: {', '.join(valid_variants)}"
            )

        # Set algorithm_str based on variant
        self.algorithm_str = f"evox_{self._variant.lower()}"

        # Define the problem in EvoX so it can be optimized
        class ESProblem(EvoxProblem):
            def __init__(self, batch_size):
                super().__init__()
                self.batch_size = batch_size
                # vmap for a single batch
                self.vectorized_objective = jax.vmap(
                    problem.objective_function, in_axes=0
                )
                # warmup the function to compile it
                _ = self.vectorized_objective(
                    jnp.zeros((self.batch_size, problem.n_params))
                )

            def evaluate(self, pop: torch.Tensor) -> torch.Tensor:
                # EvoX works in torch, this project in JAX
                jpop = t2j(pop)

                # Split population into batches to avoid OOM
                n_individuals = jpop.shape[0]
                all_losses = []

                for i in range(0, n_individuals, self.batch_size):
                    batch = jpop[i : i + self.batch_size]
                    batch_losses = self.vectorized_objective(batch)
                    all_losses.append(batch_losses)

                # Concatenate all batch results
                losses = jnp.concatenate(all_losses, axis=0)
                return j2t(losses)

        # ...and initiate it.
        self._es_problem = ESProblem(self._batch_size)

    @jaxtyped(typechecker=typechecker)
    def optimize(
        self,
        save_to_file: bool = True,
        init_params_pop: Float[Array, "{pop_size} {self._problem.n_params}"]
        | None = None,
        return_best_params_history: bool = False,
        random_seed: int | None = None,
        wall_times: list[int | float] | None = None,
        pop_size: int = 100,
        n_generations: int | None = None,
        **es_kwargs,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_gens {self._problem.n_params}"] | Float[Array, "0"],
        Float[Array, "n_gens"],
        list[int] | None,
        Float[Array, "n_gens {pop_size}"],
    ]:
        """Run ES optimization.

        Args:
            save_to_file (bool): Whether to save optimization results to file. Defaults to True.
            init_params_pop (Float[Array, "pop_size n_params"] | None): Initial population of
                parameters. Not supported by most ES variants (mean is typically initialized instead).
                Defaults to None.
            return_best_params_history (bool): Whether to track best parameters at each
                generation. Defaults to False.
            random_seed (int | None): Random seed for reproducibility. Controls both initial
                population generation and random coefficients during optimization. Defaults to None.
            wall_times (list[int | float] | None): List of wall-time checkpoints (in seconds).
                The algorithm runs until the maximum checkpoint. At each checkpoint,
                the current generation index is recorded. Checkpoints are automatically
                sorted ascending; returned indices follow this sorted order.
                If None, runs for n_generations. Defaults to None.
            pop_size (int): Number of individuals in the population. Defaults to 100.
            n_generations (int | None): Number of generations to run. Required if wall_times
                is None. Can be combined with wall_times as an additional stopping criterion.
                Defaults to None.
            **es_kwargs: Variant-specific keyword arguments passed to the EvoX algorithm constructor.
                Parameter options by variant:
                - CMAES: mean_init (torch.Tensor, initial mean, required), sigma (float,
                  step size, required), weights (torch.Tensor | None, recombination weights)
                - OpenES: center_init (torch.Tensor, initial center, required), sigma (float,
                  noise standard deviation, required), optimizer_type (str, "adam" or "sgd")
                - XNES: center_init (torch.Tensor), sigma_init (float)
                - SeparableNES: center_init (torch.Tensor), sigma_init (float)
                - DES: center_init (torch.Tensor), sigma_init (float)
                - SNES: center_init (torch.Tensor), sigma_init (float)
                - ARS: center_init (torch.Tensor), sigma_init (float), lr (float, learning rate)
                - ASEBO: center_init (torch.Tensor), sigma_init (float)
                - PersistentES: center_init (torch.Tensor), sigma (float), alpha (float)
                - NoiseReuseES: center_init (torch.Tensor), sigma (float), k (int, reuse count)
                - GuidedES: center_init (torch.Tensor), sigma (float), surrogate (callable)
                - ESMC: center_init (torch.Tensor), sigma (float), num_samples (int)
                Refer to EvoX documentation for complete parameter details.

        Returns:
            tuple: A 5-tuple containing:
                - best_params (Float[Array, "n_params"]): Best parameters found.
                - best_params_history (Float[Array, "n_gens n_params"]): History of best
                  parameters per generation. Empty array if return_best_params_history=False.
                - losses (Float[Array, "n_gens"]): Best loss at each generation.
                - wall_time_indices (list[int] | None): Generation indices corresponding to
                  each wall_times checkpoint (in sorted ascending order). None if wall_times is None.
                - population_losses (Float[Array, "n_gens pop_size"]): Loss for each individual
                  at each generation (ES-specific). Could contain NaN if population sizes vary.
        """
        # Set random seed if provided (affects both initialization and step randomness)
        # Use numpy random to match AdamGD initialization
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        # Initiate monitor for loss tracking etc.
        monitor = EvalMonitor()

        # Get bounds from problem
        if not hasattr(self._problem, "bounds"):
            raise ValueError(
                f"Problem {type(self._problem).__name__} must have a 'bounds' attribute."
            )
        problem_bounds = self._problem.bounds
        lb_np = np.asarray(problem_bounds[0])
        ub_np = np.asarray(problem_bounds[1])

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

        # Initiate algorithm with hyper params using the selected variant
        AlgorithmClass = variant_map[self._variant]

        # CMAES requires special initialization
        if self._variant == "CMAES":
            if "mean_init" not in es_kwargs:
                # Initialize mean uniformly within bounds (using numpy to match AdamGD for now)
                mean_init = torch.from_numpy(
                    np.random.uniform(lb_np, ub_np, size=self._problem.n_params)
                ).float()
                es_kwargs["mean_init"] = mean_init
            if "sigma" not in es_kwargs:
                # Set sigma to 0.3 * (ub - lb)
                sigma = float(np.mean(0.3 * (ub_np - lb_np)))
                es_kwargs["sigma"] = sigma
            algorithm = AlgorithmClass(pop_size=pop_size, **es_kwargs)
        else:
            # Most other ES variants need bounds and center initialization
            lb = j2t(problem_bounds[0])
            ub = j2t(problem_bounds[1])

            # Check if center_init is provided, otherwise initialize uniformly within bounds
            if "center_init" not in es_kwargs and "mean_init" not in es_kwargs:
                # Use numpy uniform random to match AdamGD initialization
                center_init = torch.from_numpy(
                    np.random.uniform(lb_np, ub_np, size=self._problem.n_params)
                ).float()
                es_kwargs["center_init"] = center_init

            # Provide sigma_init if not given
            if "sigma_init" not in es_kwargs and "sigma" not in es_kwargs and self._variant != "SNES":
                sigma_init = float(np.mean(0.3 * (ub_np - lb_np)))
                es_kwargs["sigma_init"] = sigma_init

            try:
                algorithm = AlgorithmClass(pop_size=pop_size, lb=lb, ub=ub, **es_kwargs)
            except TypeError:
                # Some variants don't accept lb/ub
                algorithm = AlgorithmClass(pop_size=pop_size, **es_kwargs)

        # If initial population is provided, set it before init_step (if supported)
        if init_params_pop is not None:
            # Convert to torch if needed
            if isinstance(init_params_pop, jax.Array):
                init_pop_torch = j2t(init_params_pop)
            else:
                init_pop_torch = init_params_pop

            # Try to set population if the algorithm supports it
            if hasattr(algorithm, "pop"):
                algorithm.pop = init_pop_torch

        # This results in the workflow
        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=self._es_problem,
            monitor=monitor,
        )

        # Initialize: evaluates population and sets initial best values
        workflow.init_step()

        # Executing the algorithm itself
        best_params_history = []  # Shape: (n_steps, n_params)

        # Capture initial best params after init_step
        if return_best_params_history:
            best_params = t2j(monitor.topk_solutions)[0]
            best_params_history.append(best_params)

        # Initialize wall_time_indices tracking
        wall_time_indices: list[int] | None = None
        wall_times_remaining: deque[int | float] | None = None
        if wall_times is not None:
            wall_time_indices = []
            wall_times_remaining = deque(sorted(wall_times))
            max_wall_time = wall_times_remaining[-1]

        # If there is no time limit:
        if wall_times is None:
            for _ in range(n_generations):
                workflow.step()
                if return_best_params_history:
                    best_params = t2j(monitor.topk_solutions)[0]
                    best_params_history.append(best_params)
        else:
            start_time = time.time()
            gen = 0  # Generation counter (init_step counts as generation 0)
            # With both time limit and generation limit:
            if n_generations is not None:
                for _ in range(n_generations):
                    elapsed = time.time() - start_time

                    # Record generation index at wall_times checkpoints
                    while wall_times_remaining and elapsed >= wall_times_remaining[0]:
                        wall_time_indices.append(gen)
                        wall_times_remaining.popleft()

                    if elapsed >= max_wall_time:
                        break

                    workflow.step()
                    gen += 1
                    if return_best_params_history:
                        best_params = t2j(monitor.topk_solutions)[0]
                        best_params_history.append(best_params)
            # With only time limit:
            else:
                while True:
                    elapsed = time.time() - start_time

                    # Record generation index at wall_times checkpoints
                    while wall_times_remaining and elapsed >= wall_times_remaining[0]:
                        wall_time_indices.append(gen)
                        wall_times_remaining.popleft()

                    if elapsed >= max_wall_time:
                        break

                    workflow.step()
                    gen += 1
                    if return_best_params_history:
                        best_params = t2j(monitor.topk_solutions)[0]
                        best_params_history.append(best_params)

            # Fill remaining wall_times that weren't reached with final generation
            while wall_times_remaining:
                wall_time_indices.append(gen)
                wall_times_remaining.popleft()

        # Extract results from monitor
        best_params = t2j(monitor.topk_solutions)[0]
        best_params_history = jnp.array(best_params_history)

        # Handle fit_history: it's a list of fitness values per generation
        # Each generation may have different number of individuals, so we need to pad to uniform shape
        if len(monitor.fit_history) > 0:
            # Convert to numpy first, then find max length
            fit_history_np = [np.asarray(f) for f in monitor.fit_history]
            max_len = max(len(f) for f in fit_history_np)

            # Pad each generation's fitness array to max_len with NaN
            padded_history = []
            for f in fit_history_np:
                if len(f) < max_len:
                    padded = np.full(max_len, np.nan)
                    padded[: len(f)] = f
                    padded_history.append(padded)
                else:
                    padded_history.append(f)

            population_losses = jnp.array(padded_history)
            # Compute losses as min (ignoring NaN values if present)
            losses = jnp.nanmin(population_losses, axis=1)
        else:
            # Edge case: empty history
            population_losses = jnp.array([])
            losses = jnp.array([])

        print("Best params history shape:")
        print(best_params_history.shape)

        hyper_param_str = f"_gen{n_generations}_pop{pop_size}"

        if save_to_file:
            self._problem.output_to_files(
                best_params=best_params,
                losses=losses,
                population_losses=population_losses,
                algorithm_str=self.algorithm_str,
                hyper_param_str=hyper_param_str,
            )

        return (
            best_params,
            best_params_history,
            losses,
            wall_time_indices,
            population_losses,
        )