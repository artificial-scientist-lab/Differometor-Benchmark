"""Bayesian Optimization using BoTorch (PyTorch-based, GPU-accelerated)."""

import jax.numpy as jnp
import numpy as np
import time
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from collections import deque
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from differometor.utils import sigmoid_bounding
from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)
from dfbench.core.utils import t2j_numpy as t2j


class BotorchBO(OptimizationAlgorithm):
    """Bayesian Optimization using BoTorch (PyTorch-based, GPU-accelerated).

    Implements Bayesian Optimization using a Gaussian Process surrogate model
    and Expected Improvement acquisition function. Uses BoTorch for GPU-accelerated
    GP fitting and acquisition optimization.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("botorch_bo").
        algorithm_type (AlgorithmType): Type classification (SURROGATE_BASED).
        _problem (ContinuousProblem): The optimization problem instance.
        max_iterations (int): Maximum number of BO iterations (excluding initial samples).
        device (torch.device): PyTorch device (cuda if available, else cpu).
        dtype (torch.dtype): PyTorch dtype for tensors (float64 for numerical stability).

    Note:
        This algorithm searches in the bounded parameter space using `problem.objective_function`.
        When `use_problem_bounds=True` (default), it uses the problem's native bounds.
        Otherwise, it uses the provided `lb`/`ub` parameters (defaulting to [-10, 10]).

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = BotorchBO(problem, max_iterations=100)
        >>> best_params, history, losses, wall_indices = optimizer.optimize(
        ...     wall_times=[30, 60, 120],
        ...     n_initial=10,
        ... )
    """

    algorithm_str: str = "botorch_bo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self, problem: ContinuousProblem, max_iterations: int = 100) -> None:
        """Initialize BoTorch Bayesian Optimization.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
                Must have `objective_function` and `bounds` attributes.
            max_iterations (int): Maximum number of BO iterations (excluding initial
                random samples). Defaults to 100.
        """
        self._problem = problem
        self.max_iterations = max_iterations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

    def _create_evaluate_fn(self, use_problem_bounds: bool):
        """Create the evaluation function based on bounds mode.

        Args:
            use_problem_bounds: If True, uses objective_function (bounded space).
                If False, uses sigmoid_objective_function (unbounded space).

        Returns:
            Callable that evaluates a torch tensor and returns loss as torch tensor.
        """
        if use_problem_bounds:
            # Use objective_function which expects params in bounded space
            objective_fn = self._problem.objective_function
        else:
            # Use sigmoid_objective_function for unbounded search
            objective_fn = self._problem.sigmoid_objective_function

        def evaluate(x: torch.Tensor) -> torch.Tensor:
            y = float(objective_fn(t2j(x)))
            return torch.tensor([y], device=self.device, dtype=self.dtype)

        # Warmup JIT compilation
        _ = objective_fn(jnp.zeros(self._problem.n_params))

        return evaluate

    @jaxtyped(typechecker=typechecker)
    def optimize(
        self,
        save_to_file: bool = True,
        use_problem_bounds: bool = True,
        init_params: Float[Array, "{self._problem.n_params}"] | None = None,
        return_best_params_history: bool = False,
        random_seed: int | None = None,
        wall_times: list[int | float] | None = None,
        lb: Float[Array, "{self._problem.n_params}"] | None = None,
        ub: Float[Array, "{self._problem.n_params}"] | None = None,
        n_initial: int = 5,
        **bo_kwargs,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_iters {self._problem.n_params}"] | None,
        Float[Array, "n_iters"],
        list[int] | None,
    ]:
        """Run Bayesian Optimization.

        Args:
            save_to_file (bool): Whether to save optimization results to file. Defaults to True.
            use_problem_bounds (bool): If True, use bounds from `problem.bounds` and
                `problem.objective_function`. If False, use `lb`/`ub` parameters with
                `problem.sigmoid_objective_function`. Defaults to True.
            init_params (Float[Array, "n_params"] | None): Initial parameters to include
                in the training set. If None, only random samples are used. Defaults to None.
            return_best_params_history (bool): Whether to track best parameters at each
                iteration. Defaults to False.
            random_seed (int | None): Random seed for reproducibility. Controls initial
                sample generation and acquisition optimization. Defaults to None.
            wall_times (list[int | float] | None): List of wall-time checkpoints (in seconds).
                The algorithm runs until the maximum checkpoint. At each checkpoint,
                the current iteration index is recorded. If None, runs for max_iterations.
                Defaults to None.
            lb (Float[Array, "n_params"] | None): Lower bounds for each parameter.
                Ignored if use_problem_bounds=True. Defaults to -10 for all parameters.
            ub (Float[Array, "n_params"] | None): Upper bounds for each parameter.
                Ignored if use_problem_bounds=True. Defaults to 10 for all parameters.
            n_initial (int): Number of initial random samples before fitting GP.
                Defaults to 5.
            **bo_kwargs: Additional keyword arguments (reserved for future use).

        Returns:
            tuple: A 4-tuple containing:
                - best_params (Float[Array, "n_params"]): Best parameters found.
                - best_params_history (Float[Array, "n_iters n_params"] | None): History of
                  best parameters per iteration. None if return_best_params_history=False.
                - losses (Float[Array, "n_iters"]): Loss at each iteration (including initial samples).
                - wall_time_indices (list[int] | None): Iteration indices corresponding to
                  each wall_times checkpoint (in sorted ascending order). None if wall_times is None.
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        # Determine bounds based on use_problem_bounds flag
        if use_problem_bounds:
            if not hasattr(self._problem, "bounds"):
                raise ValueError(
                    "use_problem_bounds=True requires the problem to have a 'bounds' attribute. "
                    f"Problem {type(self._problem).__name__} does not have this attribute."
                )
            problem_bounds = self._problem.bounds
            if isinstance(problem_bounds, np.ndarray):
                lb_np, ub_np = problem_bounds[0], problem_bounds[1]
            else:
                lb_np, ub_np = np.array(problem_bounds[0]), np.array(problem_bounds[1])
            print(
                f"Using problem bounds: lb shape={lb_np.shape}, ub shape={ub_np.shape}"
            )
        else:
            if lb is None:
                lb_np = np.full(self._problem.n_params, -10.0)
            else:
                lb_np = np.array(lb)
            if ub is None:
                ub_np = np.full(self._problem.n_params, 10.0)
            else:
                ub_np = np.array(ub)

        # Create evaluation function based on bounds mode
        evaluate = self._create_evaluate_fn(use_problem_bounds)

        # Convert bounds to torch tensors
        bounds = torch.tensor(
            np.array([lb_np, ub_np]), device=self.device, dtype=self.dtype
        )
        bounds_range = bounds[1] - bounds[0]

        # Normalization helpers: map between original space and unit cube [0, 1]^d
        def normalize(x: torch.Tensor) -> torch.Tensor:
            return (x - bounds[0]) / bounds_range

        def unnormalize(x: torch.Tensor) -> torch.Tensor:
            return x * bounds_range + bounds[0]

        # Unit cube bounds for GP
        unit_bounds = torch.tensor(
            [[0.0] * self._problem.n_params, [1.0] * self._problem.n_params],
            device=self.device,
            dtype=self.dtype,
        )

        # Initialize tracking variables
        best_params = jnp.zeros(self._problem.n_params)
        best_params_history: list = []
        best_loss = float("inf")
        losses: list = []

        # Initialize wall_time tracking
        wall_time_indices: list[int] | None = None
        wall_times_remaining: deque[int | float] | None = None
        max_wall_time: float | None = None
        if wall_times is not None:
            wall_time_indices = []
            wall_times_remaining = deque(sorted(wall_times))
            max_wall_time = wall_times_remaining[-1]

        # Generate initial samples (stored in normalized [0,1] space)
        if init_params is not None:
            train_X_orig = torch.tensor(
                np.array(init_params).reshape(1, -1),
                device=self.device,
                dtype=self.dtype,
            )
            train_X = normalize(train_X_orig)
            train_Y = evaluate(train_X_orig[0]).unsqueeze(0)
        else:
            train_X = torch.empty(
                0, self._problem.n_params, device=self.device, dtype=self.dtype
            )
            train_Y = torch.empty(0, 1, device=self.device, dtype=self.dtype)

        # Add random initial points (in normalized space)
        n_random = max(0, n_initial - len(train_X))
        if n_random > 0:
            random_X = torch.rand(
                n_random, self._problem.n_params, device=self.device, dtype=self.dtype
            )
            random_Y = torch.stack([evaluate(unnormalize(x)) for x in random_X])
            train_X = (
                torch.cat([train_X, random_X], dim=0) if len(train_X) > 0 else random_X
            )
            train_Y = (
                torch.cat([train_Y, random_Y], dim=0) if len(train_Y) > 0 else random_Y
            )

        # Record initial losses and update best
        for idx, y in enumerate(train_Y):
            loss = float(y.item())
            losses.append(loss)
            if loss < best_loss - 1e-4:
                best_loss = loss
                best_params = t2j(unnormalize(train_X[idx]))

            if return_best_params_history:
                best_params_history.append(best_params)

        def run_iteration(
            iteration: int,
            train_X: torch.Tensor,
            train_Y: torch.Tensor,
            current_best_loss: float,
        ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
            """Run a single BO iteration.

            Args:
                iteration: Current iteration number (for logging).
                train_X: Training inputs in normalized [0,1] space.
                train_Y: Training outputs (losses).
                current_best_loss: Current best loss for EI computation.

            Returns:
                Updated train_X, train_Y, new loss, and candidate in original space.
            """
            # Fit GP model (minimize loss, so negate Y for EI which maximizes)
            gp = SingleTaskGP(train_X, -train_Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            # Optimize acquisition function in unit cube
            acq = LogExpectedImprovement(gp, best_f=-current_best_loss)
            candidate, _ = optimize_acqf(
                acq,
                bounds=unit_bounds,
                q=1,
                num_restarts=5,
                raw_samples=20,
            )

            # Evaluate candidate in original space
            candidate_orig = unnormalize(candidate[0])
            new_Y = evaluate(candidate_orig)
            train_X = torch.cat([train_X, candidate], dim=0)
            train_Y = torch.cat([train_Y, new_Y.unsqueeze(0)], dim=0)

            loss = float(new_Y.item())
            return train_X, train_Y, loss, candidate_orig

        # Main optimization loop
        if wall_times is not None:
            start_time = time.time()
            i = len(losses)

            while (time.time() - start_time) < max_wall_time:
                elapsed = time.time() - start_time

                # Record iteration index at wall_times checkpoints
                while wall_times_remaining and elapsed >= wall_times_remaining[0]:
                    wall_time_indices.append(i)
                    wall_times_remaining.popleft()

                train_X, train_Y, loss, candidate = run_iteration(
                    i, train_X, train_Y, best_loss
                )

                if i % 10 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                if loss < best_loss - 1e-4:
                    best_loss = loss
                    best_params = t2j(candidate)
                    print(f"Iteration {i}: New best loss = {loss}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                losses.append(loss)
                i += 1

            # Fill remaining wall_times that weren't reached
            while wall_times_remaining:
                wall_time_indices.append(i - 1 if i > 0 else 0)
                wall_times_remaining.popleft()

        else:
            for i in range(len(losses), self.max_iterations + n_initial):
                train_X, train_Y, loss, candidate = run_iteration(
                    i, train_X, train_Y, best_loss
                )

                if i % 10 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                if loss < best_loss - 1e-4:
                    best_loss = loss
                    best_params = t2j(candidate)
                    print(f"Iteration {i}: New best loss = {loss}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                losses.append(loss)

        losses_array = jnp.array(losses)
        best_params_history_array = (
            jnp.array(best_params_history) if return_best_params_history else None
        )

        # Transform unbounded parameters back to bounded space if using sigmoid objective
        if not use_problem_bounds:
            best_params = sigmoid_bounding(best_params, self._problem.bounds)
            if return_best_params_history:
                best_params_history_array = jnp.array(
                    [sigmoid_bounding(p, self._problem.bounds) for p in best_params_history]
                )

        if save_to_file:
            self._problem.output_to_files(
                best_params=best_params,
                losses=losses_array,
                population_losses=None,
                algorithm_str=self.algorithm_str,
                hyper_param_str=f"n_initial{n_initial}",
                hyper_param_str_in_filename=True,
            )

        return best_params, best_params_history_array, losses_array, wall_time_indices
