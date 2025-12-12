import jax
import jax.numpy as jnp
import numpy as np
import torch
import time
from collections import deque
from evox.algorithms import PSO, CLPSO, CSO, DMSPSOEL, FSPSO, SLPSOGS, SLPSOUS
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


class EvoxPSO(OptimizationAlgorithm):
    """EvoX-based Particle Swarm Optimization algorithm.

    Implements PSO using the EvoX library with PyTorch backend. Handles batched
    evaluation of particles to manage memory efficiently. Supports multiple PSO
    variants through the variant parameter.

    Attributes:
        algorithm_str (str): Identifier string (e.g., "evox_pso", "evox_clpso").
        algorithm_type (AlgorithmType): Type classification (EVOLUTIONARY).
        _problem (ContinuousProblem): The optimization problem instance.
        _batch_size (int): Number of particles to evaluate per batch.
        _variant (str): PSO variant name (uppercase, e.g., "PSO", "CLPSO").
        _pso_problem (EvoxProblem): EvoX problem wrapper for the objective function.

    Note:
        This algorithm uses `problem.objective_function` with the problem's bounds.
        The swarm searches directly in the bounded parameter space.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = EvoxPSO(problem, batch_size=50, variant="CLPSO")
        >>> best_params, history, losses, wall_indices, pop_losses = optimizer.optimize(
        ...     pop_size=200,
        ...     wall_times=[30, 60, 120],
        ... )
    """

    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self, problem: ContinuousProblem, batch_size: int = 5, variant: str = "PSO"
    ) -> None:
        """Initialize EvoX Particle Swarm Optimization.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
            batch_size (int): Number of particles to evaluate simultaneously in each batch.
                Reduce this value if encountering out-of-memory errors. Defaults to 5.
            variant (str): PSO variant to use. Options:
                - 'PSO': Standard Particle Swarm Optimization (default)
                - 'CLPSO': Comprehensive Learning PSO
                - 'CSO': Competitive Swarm Optimizer
                - 'DMSPSOEL': Dynamic Multi-Swarm PSO with Elite Learning
                - 'FSPSO': Fitness-Sharing PSO
                - 'SLPSOGS': Social Learning PSO with Gaussian Sampling
                - 'SLPSOUS': Social Learning PSO with Uniform Sampling
                Defaults to 'PSO'.
        """
        self._problem = problem
        self._batch_size = batch_size
        self._variant = variant.upper()  # Normalize to uppercase

        # Validate variant
        valid_variants = [
            "PSO",
            "CLPSO",
            "CSO",
            "DMSPSOEL",
            "FSPSO",
            "SLPSOGS",
            "SLPSOUS",
        ]
        if self._variant not in valid_variants:
            raise ValueError(
                f"Unknown PSO variant: '{variant}'. "
                f"Valid options are: {', '.join(valid_variants)}"
            )

        # Set algorithm_str based on variant
        self.algorithm_str = f"evox_{self._variant.lower()}"

        # Define the problem in EvoX so it can be optimized
        class PSOProblem(EvoxProblem):
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
                n_particles = jpop.shape[0]
                all_losses = []

                for i in range(0, n_particles, self.batch_size):
                    batch = jpop[i : i + self.batch_size]
                    batch_losses = self.vectorized_objective(batch)
                    all_losses.append(batch_losses)

                # Concatenate all batch results
                losses = jnp.concatenate(all_losses, axis=0)
                return j2t(losses)

        # ...and initiate it.
        self._pso_problem = PSOProblem(self._batch_size)

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
        **pso_kwargs,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_gens {self._problem.n_params}"] | Float[Array, "0"],
        Float[Array, "n_gens"],
        list[int] | None,
        Float[Array, "n_gens {pop_size}"],
    ]:
        """Run PSO optimization.

        Args:
            save_to_file (bool): Whether to save optimization results to file. Defaults to True.
            init_params_pop (Float[Array, "pop_size n_params"] | None): Initial population of
                parameters. If None, randomly initialized within bounds. Defaults to None.
            return_best_params_history (bool): Whether to track best parameters at each
                generation. Defaults to False.
            random_seed (int | None): Random seed for reproducibility. Controls both initial
                population generation and random coefficients during optimization. Defaults to None.
            wall_times (list[int | float] | None): List of wall-time checkpoints (in seconds).
                The algorithm runs until the maximum checkpoint. At each checkpoint,
                the current generation index is recorded. Checkpoints are automatically
                sorted ascending; returned indices follow this sorted order.
                If None, runs for n_generations. Defaults to None.
            pop_size (int): Number of particles in the swarm. Defaults to 100.
            n_generations (int | None): Number of generations to run. Required if wall_times
                is None. Can be combined with wall_times as an additional stopping criterion.
                Defaults to None.
            **pso_kwargs: Variant-specific keyword arguments passed to the EvoX algorithm constructor.
                Parameter options by variant:
                - PSO: w (float, inertia weight, default 0.6), phi_p (float,
                  cognitive coefficient, default 2.5), phi_g (float, social coefficient,
                  default 0.8)
                - CLPSO: inertia_weight (float, default 0.5), const_coefficient (float,
                  default 1.5), learning_probability (float, default 0.05)
                - CSO: phi (float, inertia weight, default 0.0), mean (torch.Tensor | None),
                  stdev (torch.Tensor | None)
                - DMSPSOEL: dynamic_sub_swarm_size (int, default 10), dynamic_sub_swarms_num
                  (int, default 5), following_sub_swarm_size (int, default 10),
                  regrouped_iteration_num (int, default 50), max_iteration (int, default 100),
                  inertia_weight (float, default 0.7), pbest_coefficient (float, default 1.5),
                  lbest_coefficient (float, default 1.5), rbest_coefficient (float, default 1.0),
                  gbest_coefficient (float, default 1.0). Note: pop_size is ignored for DMSPSOEL;
                  it's calculated as dynamic_sub_swarm_size * dynamic_sub_swarms_num + following_sub_swarm_size.
                - FSPSO: inertia_weight (float, default 0.6), cognitive_coefficient (float,
                  default 2.5), social_coefficient (float, default 0.8), mean (torch.Tensor | None),
                  stdev (torch.Tensor | None), mutate_rate (float, default 0.01)
                - SLPSOGS: social_influence_factor (float, default 0.2),
                  demonstrator_choice_factor (float, default 0.7)
                - SLPSOUS: social_influence_factor (float, default 0.2),
                  demonstrator_choice_factor (float, default 0.7)
                Refer to EvoX documentation for complete parameter details.

        Returns:
            tuple: A 5-tuple containing:
                - best_params (Float[Array, "n_params"]): Best parameters found.
                - best_params_history (Float[Array, "n_gens n_params"]): History of best
                  parameters per generation. Empty array if return_best_params_history=False.
                - losses (Float[Array, "n_gens"]): Best loss at each generation.
                - wall_time_indices (list[int] | None): Generation indices corresponding to
                  each wall_times checkpoint (in sorted ascending order). None if wall_times is None.
                - population_losses (Float[Array, "n_gens pop_size"]): Loss for each particle
                  at each generation (PSO-specific). Could contain NaN if population sizes vary (CSO).
        """
        # Set random seed if provided (affects both initialization and step randomness)
        if random_seed is not None:
            torch.manual_seed(random_seed)

        # Initiate monitor for loss tracking etc.
        monitor = EvalMonitor()

        # Get bounds from problem
        if not hasattr(self._problem, "bounds"):
            raise ValueError(
                f"Problem {type(self._problem).__name__} must have a 'bounds' attribute."
            )
        problem_bounds = self._problem.bounds
        # problem.bounds is expected to be shape [2, n_params] with [lower, upper]
        lb = j2t(problem_bounds[0])
        ub = j2t(problem_bounds[1])

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

        # Initiate algorithm with hyper params using the selected variant
        AlgorithmClass = variant_map[self._variant]
        algorithm = AlgorithmClass(pop_size=pop_size, lb=lb, ub=ub, **pso_kwargs)

        # If initial population is provided, set it before init_step
        if init_params_pop is not None:
            # Convert to torch if needed
            if isinstance(init_params_pop, jax.Array):
                init_pop_torch = j2t(init_params_pop)
            else:
                init_pop_torch = init_params_pop
            # Only override the population - init_step will handle the rest
            algorithm.pop = init_pop_torch

        # This results in the workflow
        workflow = StdWorkflow(
            algorithm=algorithm,
            problem=self._pso_problem,
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
        # Each generation may have different number of particles, so we need to pad to uniform shape
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
