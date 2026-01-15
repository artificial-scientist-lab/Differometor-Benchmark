"""State-of-the-art Bayesian Optimization using BoTorch with batch acquisition."""

import jax
import jax.numpy as jnp
import numpy as np
import time
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement as qLogEI
from botorch.optim import optimize_acqf
from botorch.generation import gen_candidates_scipy
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from collections import deque
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)
from dfbench.core.utils import t2j_numpy as t2j


class BotorchBO(OptimizationAlgorithm):
    """State-of-the-art Bayesian Optimization using BoTorch with batch acquisition.

    Implements Bayesian Optimization using a Gaussian Process surrogate model
    and batch Expected Improvement acquisition function (qLogEI). Uses BoTorch for 
    GPU-accelerated GP fitting and acquisition optimization with Sobol initialization.

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

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = BotorchBO(problem, max_iterations=100)
        >>> best_params, history, losses, wall_indices = optimizer.optimize(
        ...     wall_times=[30, 60, 120],
        ...     n_initial=10,
        ...     batch_size=5,
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

    def _evaluate_y(
        self,
        X: Float[torch.Tensor, "... d"],
        bounds: Float[torch.Tensor, "2 d"],
        max_retries: int = 3,
        perturbation_scale: float = 1e-6,
    ) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]:
        """Evaluate objective function at given input(s).

        If NaN/Inf is encountered, attempts to perturb the point and retry.
        Returns both values and a validity mask so invalid points can be
        filtered from GP training data.

        Args:
            X: Input(s) in normalized [0,1] space.
            bounds: Original bounds for unnormalization.
            max_retries: Number of retries with perturbation for NaN values.
            perturbation_scale: Scale of random perturbation for retries.

        Returns:
            Tuple of (negated objective values, validity mask).
            Invalid entries have NaN values and False in mask.
        """
        unnormalized_X = unnormalize(X, bounds)
        X_jax = t2j(unnormalized_X)

        # Handle batch dimension
        if X_jax.ndim == 1:
            Y_jax = self._problem.objective_function(X_jax)
            Y_torch = torch.tensor([Y_jax.item()], device=X.device, dtype=X.dtype)
        else:
            batched_obj = jax.vmap(self._problem.objective_function)
            Y_jax = batched_obj(X_jax)
            Y_torch = torch.from_numpy(np.array(Y_jax)).to(device=X.device, dtype=X.dtype)

        # Track validity
        invalid_mask = torch.isnan(Y_torch) | torch.isinf(Y_torch)

        # Retry invalid points with small perturbations
        if torch.any(invalid_mask) and max_retries > 0:
            invalid_indices = torch.where(invalid_mask)[0]
            print(f"Warning: {len(invalid_indices)} NaN/Inf values detected, retrying with perturbation...")

            for idx in invalid_indices:
                for retry in range(max_retries):
                    # Perturb the point slightly (in normalized space)
                    X_perturbed = X[idx].clone()
                    perturbation = torch.randn_like(X_perturbed) * perturbation_scale * (retry + 1)
                    X_perturbed = torch.clamp(X_perturbed + perturbation, 0.0, 1.0)

                    # Re-evaluate
                    unnorm_perturbed = unnormalize(X_perturbed, bounds)
                    X_jax_perturbed = t2j(unnorm_perturbed)
                    Y_retry = self._problem.objective_function(X_jax_perturbed)
                    Y_retry_torch = torch.tensor(Y_retry.item(), device=X.device, dtype=X.dtype)

                    if torch.isfinite(Y_retry_torch):
                        Y_torch[idx] = Y_retry_torch
                        invalid_mask[idx] = False
                        break

            # Report remaining invalid points
            remaining_invalid = torch.sum(invalid_mask).item()
            if remaining_invalid > 0:
                print(f"Warning: {remaining_invalid} points still invalid after retries, will be excluded from GP")

        valid_mask = ~invalid_mask
        return -Y_torch, valid_mask  # Negate for maximization

    def _fit_model(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
    ) -> SingleTaskGP:
        """Fit GP model to current data.

        Args:
            train_X: Training inputs of shape (n, d).
            train_Y: Training targets of shape (n, 1).

        Returns:
            Trained Gaussian Process model.
        """
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        return model

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
        n_initial: int = 10,
        batch_size: int = 1,
        **bo_kwargs,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_iters {self._problem.n_params}"] | None,
        Float[Array, "n_iters"],
        list[int] | None,
    ]:
        """Run Bayesian Optimization with batch acquisition.

        Args:
            save_to_file (bool): Whether to save optimization results to file. Defaults to True.
            use_problem_bounds (bool): If True, use bounds from `problem.bounds` and
                `problem.objective_function`. Defaults to True.
            init_params (Float[Array, "n_params"] | None): Initial parameters to include
                in the training set. If None, only Sobol samples are used. Defaults to None.
            return_best_params_history (bool): Whether to track best parameters at each
                iteration. Defaults to False.
            random_seed (int | None): Random seed for reproducibility. Controls Sobol
                sample generation and acquisition optimization. Defaults to None.
            wall_times (list[int | float] | None): List of wall-time checkpoints (in seconds).
                The algorithm runs until the maximum checkpoint. At each checkpoint,
                the current iteration index is recorded. If None, runs for max_iterations.
                Defaults to None.
            lb (Float[Array, "n_params"] | None): Lower bounds for each parameter.
                Ignored if use_problem_bounds=True. Defaults to -10 for all parameters.
            ub (Float[Array, "n_params"] | None): Upper bounds for each parameter.
                Ignored if use_problem_bounds=True. Defaults to 10 for all parameters.
            n_initial (int): Number of initial Sobol samples before fitting GP.
                Defaults to 10.
            batch_size (int): Number of points to acquire per iteration. Defaults to 1.
            **bo_kwargs: Additional keyword arguments for acquisition optimization
                (raw_samples, num_restarts, etc.).

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
        
        if use_problem_bounds:
            if not hasattr(self._problem, "bounds"):
                raise ValueError(
                    "use_problem_bounds=True requires the problem to have a 'bounds' attribute."
                )
            problem_bounds = self._problem.bounds
            if isinstance(problem_bounds, np.ndarray):
                lb_np, ub_np = problem_bounds[0], problem_bounds[1]
            else:
                lb_np, ub_np = np.array(problem_bounds[0]), np.array(problem_bounds[1])
        else:
            lb_np = np.full(self._problem.n_params, -10.0) if lb is None else np.array(lb)
            ub_np = np.full(self._problem.n_params, 10.0) if ub is None else np.array(ub)

        # Convert to torch tensors
        problem_bounds_torch = torch.tensor(
            np.array([lb_np, ub_np]), device=self.device, dtype=self.dtype
        )

        # Unit cube bounds for GP [0, 1]^d
        unit_bounds = torch.stack([
            torch.zeros(self._problem.n_params, dtype=self.dtype, device=self.device),
            torch.ones(self._problem.n_params, dtype=self.dtype, device=self.device),
        ], dim=0)

        # Warmup JIT compilation
        _ = self._problem.objective_function(jnp.zeros(self._problem.n_params))

        # Initialize tracking
        best_params = jnp.zeros(self._problem.n_params)
        best_params_history: list = []
        best_loss = float("inf")
        losses: list = []

        # Wall-time tracking
        wall_time_indices: list[int] | None = None
        wall_times_remaining: deque[int | float] | None = None
        max_wall_time: float | None = None
        if wall_times is not None:
            wall_time_indices = []
            wall_times_remaining = deque(sorted(wall_times))
            max_wall_time = wall_times_remaining[-1]

        # Generate initial Sobol samples
        sobol = torch.quasirandom.SobolEngine(
            dimension=self._problem.n_params, scramble=True, seed=random_seed
        )
        train_X = sobol.draw(n=n_initial).to(dtype=self.dtype, device=self.device)
        
        # Include init_params if provided
        if init_params is not None:
            init_X_unnorm = torch.tensor(
                np.array(init_params).reshape(1, -1),
                device=self.device,
                dtype=self.dtype,
            )
            init_X_norm = normalize(init_X_unnorm, problem_bounds_torch)
            train_X = torch.cat([init_X_norm, train_X], dim=0)

        # Evaluate initial samples (returns Y values and validity mask)
        train_Y_raw, valid_mask = self._evaluate_y(train_X, problem_bounds_torch)
        train_Y_raw = train_Y_raw.unsqueeze(-1)

        # Record all losses (including invalid as NaN for tracking)
        for idx, y in enumerate(train_Y_raw):
            if valid_mask[idx]:
                loss = -float(y.item())  # Convert back to minimization
                if loss < best_loss - 1e-4:
                    best_loss = loss
                    best_params = t2j(unnormalize(train_X[idx], problem_bounds_torch))
            
            # Store best_loss so far (not current loss)
            losses.append(best_loss if best_loss != float("inf") else float("nan"))
                
            if return_best_params_history:
                best_params_history.append(best_params)

        # Filter to only valid points for GP training
        train_X = train_X[valid_mask]
        train_Y = train_Y_raw[valid_mask]

        if len(train_Y) == 0:
            raise ValueError("All initial evaluations returned NaN/Inf. Check problem bounds and objective function.")

        # Acquisition optimization options
        acqf_options = {
            "raw_samples": bo_kwargs.get("raw_samples", 512),
            "num_restarts": bo_kwargs.get("num_restarts", 4),
            "retry_on_optimization_warning": False,
            "options": {
                "nonnegative": False,
                "sample_around_best": True,
                "sample_around_best_sigma": 0.1,
                "maxiter": 300,
                "batch_limit": 64,
            },
        }

        def run_iteration(
            iteration: int,
            train_X: torch.Tensor,
            train_Y: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, list[float], list[bool]]:
            """Run a single BO iteration with batch acquisition.

            Args:
                iteration: Current iteration number.
                train_X: Training inputs in normalized [0,1] space.
                train_Y: Training outputs (negated losses for maximization).

            Returns:
                Updated train_X, train_Y, list of losses for this batch, and validity flags.
            """
            # Fit GP model
            model = self._fit_model(train_X, train_Y)
            model.eval()

            # Optimize acquisition function
            acqf = qLogEI(model, train_Y.max())
            candidates, _ = optimize_acqf(
                acqf,
                bounds=unit_bounds,
                q=batch_size,
                gen_candidates=gen_candidates_scipy,
                **acqf_options,
            )

            # Evaluate candidates (returns Y values and validity mask)
            new_Y_raw, valid_mask_batch = self._evaluate_y(candidates, problem_bounds_torch)
            new_Y_raw = new_Y_raw.unsqueeze(-1)
            
            # Build batch_losses list (NaN for invalid)
            batch_losses = []
            batch_valid = []
            for j, y in enumerate(new_Y_raw):
                if valid_mask_batch[j]:
                    batch_losses.append(-float(y.item()))
                    batch_valid.append(True)
                else:
                    batch_losses.append(float("nan"))
                    batch_valid.append(False)

            # Update training data with only valid points
            valid_candidates = candidates[valid_mask_batch]
            valid_Y = new_Y_raw[valid_mask_batch]
            if len(valid_Y) > 0:
                train_X = torch.cat([train_X, valid_candidates], dim=0)
                train_Y = torch.cat([train_Y, valid_Y], dim=0)

            return train_X, train_Y, batch_losses, batch_valid

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

                train_X, train_Y, batch_losses, batch_valid = run_iteration(i, train_X, train_Y)

                # Process batch results
                for j, loss in enumerate(batch_losses):
                    if batch_valid[j]:
                        if i % 10 == 0 and j == 0:
                            print(f"Iteration {i}: Loss = {loss}")

                        if loss < best_loss - 1e-4:
                            best_loss = loss
                            # Count valid entries up to this point to get correct index
                            valid_count = sum(batch_valid[:j+1])
                            candidate_idx = len(train_X) - sum(batch_valid) + valid_count - 1
                            best_params = t2j(unnormalize(train_X[candidate_idx], problem_bounds_torch))
                            print(f"Iteration {i}: New best loss = {loss}")

                    if return_best_params_history:
                        best_params_history.append(best_params)

                    # Store best_loss so far (not current loss)
                    losses.append(best_loss)
                    i += 1

            # Fill remaining wall_times
            while wall_times_remaining:
                wall_time_indices.append(i - 1 if i > 0 else 0)
                wall_times_remaining.popleft()

        else:
            i = len(losses)
            total_iterations = self.max_iterations + n_initial

            while i < total_iterations:
                train_X, train_Y, batch_losses, batch_valid = run_iteration(i, train_X, train_Y)

                # Process batch results
                for j, loss in enumerate(batch_losses):
                    if batch_valid[j]:
                        if i % 10 == 0 and j == 0:
                            print(f"Iteration {i}: Loss = {loss}")

                        if loss < best_loss - 1e-4:
                            best_loss = loss
                            # Count valid entries up to this point to get correct index
                            valid_count = sum(batch_valid[:j+1])
                            candidate_idx = len(train_X) - sum(batch_valid) + valid_count - 1
                            best_params = t2j(unnormalize(train_X[candidate_idx], problem_bounds_torch))
                            print(f"Iteration {i}: New best loss = {loss}")

                    if return_best_params_history:
                        best_params_history.append(best_params)

                    # Store best_loss so far (not current loss)
                    losses.append(best_loss)
                    i += 1
                    
                    if i >= total_iterations:
                        break

        losses_array = jnp.array(losses)
        best_params_history_array = (
            jnp.array(best_params_history) if return_best_params_history else None
        )

        if save_to_file:
            self._problem.output_to_files(
                best_params=best_params,
                losses=losses_array,
                population_losses=None,
                algorithm_str=self.algorithm_str,
                hyper_param_str=f"n_initial{n_initial}_batch{batch_size}",
                hyper_param_str_in_filename=True,
            )

        return best_params, best_params_history_array, losses_array, wall_time_indices
