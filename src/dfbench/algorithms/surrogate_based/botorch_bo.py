"""State-of-the-art Bayesian Optimization using BoTorch with batch acquisition."""

import jax.numpy as jnp
import numpy as np
import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement as qLogEI
from botorch.optim import optimize_acqf
from botorch.generation import gen_candidates_scipy
from botorch.utils.transforms import unnormalize, normalize
from gpytorch.mlls import ExactMarginalLogLikelihood
from jaxtyping import Array, Float

from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)
from dfbench.core.utils import t2j_numpy as t2j
from dfbench.core.objective import Objective


class BotorchBO(OptimizationAlgorithm):
    """State-of-the-art Bayesian Optimization using BoTorch with batch acquisition.

    Implements Bayesian Optimization using a Gaussian Process surrogate model
    and batch Expected Improvement acquisition function (qLogEI). Uses BoTorch for
    GPU-accelerated GP fitting and acquisition optimization with Sobol initialization.

    All history tracking is handled by the `Objective` wrapper.

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
        >>> objective = optimizer.optimize(
        ...     max_time=120,
        ...     n_initial=10,
        ...     batch_size=5,
        ... )
    """

    algorithm_str: str = "botorch_bo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(
        self,
        problem: ContinuousProblem,
        max_iterations: int = 100,
        verbose: int = 0,
        save_params_history: bool = True,
        save_batched_losses: bool = True,
        save_batched_params: bool = False,
    ) -> None:
        """Initialize BoTorch Bayesian Optimization.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
                Must have `objective_function` and `bounds` attributes.
            max_iterations (int): Maximum number of BO iterations (excluding initial
                random samples). Defaults to 100.
            verbose (int): Verbosity level (0=silent, 1+=prints). Defaults to 0.
            save_params_history: Whether to save parameter history. Defaults to True.
            save_batched_losses: Whether to save full batched losses (vs reduced).
                Defaults to True for detailed analysis.
            save_batched_params: Whether to save full batched params (memory heavy).
                Defaults to False.
        """
        self._problem = problem
        self.max_iterations = max_iterations
        self._verbose = verbose
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self._save_params_history = save_params_history
        self._save_batched_losses = save_batched_losses
        self._save_batched_params = save_batched_params

    def _evaluate_y(
        self,
        X: Float[torch.Tensor, "... d"],
        bounds: Float[torch.Tensor, "2 d"],
        obj: Objective,
        max_retries: int = 3,
        perturbation_scale: float = 1e-6,
    ) -> tuple[Float[torch.Tensor, "..."], Float[torch.Tensor, "..."]]:
        """Evaluate objective function at given input(s) through Objective wrapper.

        If NaN/Inf is encountered, attempts to perturb the point and retry.
        Returns both values and a validity mask so invalid points can be
        filtered from GP training data.

        Args:
            X: Input(s) in normalized [0,1] space.
            bounds: Original bounds for unnormalization.
            obj: Objective wrapper for evaluation tracking.
            max_retries: Number of retries with perturbation for NaN values.
            perturbation_scale: Scale of random perturbation for retries.

        Returns:
            Tuple of (negated objective values, validity mask).
            Invalid entries have NaN values and False in mask.
        """
        unnormalized_X = unnormalize(X, bounds)
        X_jax = t2j(unnormalized_X)

        # Handle batch dimension - use Objective for tracking
        if X_jax.ndim == 1:
            Y_jax = obj.value(X_jax)
            Y_torch = torch.tensor([Y_jax.item()], device=X.device, dtype=X.dtype)
        else:
            Y_jax = obj.vmap_value(X_jax)
            Y_torch = torch.from_numpy(np.array(Y_jax)).to(
                device=X.device, dtype=X.dtype
            )

        # Track validity
        invalid_mask = torch.isnan(Y_torch) | torch.isinf(Y_torch)

        # Retry invalid points with small perturbations
        if torch.any(invalid_mask) and max_retries > 0:
            invalid_indices = torch.where(invalid_mask)[0]
            print(
                f"Warning: {len(invalid_indices)} NaN/Inf values detected, retrying with perturbation..."
            )

            for idx in invalid_indices:
                for retry in range(max_retries):
                    X_perturbed = X[idx].clone()
                    perturbation = (
                        torch.randn_like(X_perturbed) * perturbation_scale * (retry + 1)
                    )
                    X_perturbed = torch.clamp(X_perturbed + perturbation, 0.0, 1.0)

                    unnorm_perturbed = unnormalize(X_perturbed, bounds)
                    X_jax_perturbed = t2j(unnorm_perturbed)
                    Y_retry = obj.value(X_jax_perturbed)
                    Y_retry_torch = torch.tensor(
                        Y_retry.item(), device=X.device, dtype=X.dtype
                    )

                    if torch.isfinite(Y_retry_torch):
                        Y_torch[idx] = Y_retry_torch
                        invalid_mask[idx] = False
                        break

            remaining_invalid = torch.sum(invalid_mask).item()
            if remaining_invalid > 0:
                print(
                    f"Warning: {remaining_invalid} points still invalid after retries"
                )

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

    def optimize(
        self,
        use_problem_bounds: bool = True,
        init_params: Float[Array, "{self._problem.n_params}"] | None = None,
        random_seed: int | None = None,
        max_time: float | None = None,
        lb: Float[Array, "{self._problem.n_params}"] | None = None,
        ub: Float[Array, "{self._problem.n_params}"] | None = None,
        n_initial: int = 10,
        batch_size: int = 1,
        verbose: int | None = None,
        print_every: int = 10,
        plot_loss: bool = False,
        save_run_to_file: bool = False,
        **bo_kwargs,
    ) -> Objective:
        """Run Bayesian Optimization with batch acquisition.

        Args:
            use_problem_bounds: If True, use bounds from `problem.bounds`.
                Defaults to True.
            init_params: Initial parameters to include in the training set.
                If None, only Sobol samples are used. Defaults to None.
            random_seed: Random seed for reproducibility. Defaults to None.
            max_time: Time budget in seconds. None for unlimited.
            lb: Lower bounds for each parameter. Ignored if use_problem_bounds=True.
            ub: Upper bounds for each parameter. Ignored if use_problem_bounds=True.
            n_initial: Number of initial Sobol samples before fitting GP.
                Defaults to 10.
            batch_size: Number of points to acquire per iteration. Defaults to 1.
            verbose: Verbosity level (0=silent, 1+=prints via Objective).
            print_every: Print summary every N evaluations.
            plot_loss: If True, call obj.output_to_files for plotting.
            save_run_to_file: If True, call obj.save_run_data for checkpointing.
            **bo_kwargs: Additional keyword arguments for acquisition optimization.

        Returns:
            The Objective instance with all logged data.
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
            lb_np = (
                np.full(self._problem.n_params, -10.0) if lb is None else np.array(lb)
            )
            ub_np = (
                np.full(self._problem.n_params, 10.0) if ub is None else np.array(ub)
            )

        problem_bounds_torch = torch.tensor(
            np.array([lb_np, ub_np]), device=self.device, dtype=self.dtype
        )

        unit_bounds = torch.stack(
            [
                torch.zeros(
                    self._problem.n_params, dtype=self.dtype, device=self.device
                ),
                torch.ones(
                    self._problem.n_params, dtype=self.dtype, device=self.device
                ),
            ],
            dim=0,
        )

        # Create Objective wrapper
        obj = Objective(
            self._problem,
            unbounded=False,
            max_time=max_time,
            max_evals=(self.max_iterations + n_initial) * batch_size,
            save_params_history=self._save_params_history,
            save_batched_losses_history=self._save_batched_losses,
            save_batched_history=self._save_batched_params,
            print_every=print_every,
            verbose=verbose if verbose is not None else self._verbose,
            algorithm_str=self.algorithm_str,
        )

        # Warmup JIT (both single and batched evaluations)
        if self._verbose >= 1:
            print(f"Warming up JIT compilation...")
        _ = obj.value(jnp.zeros(self._problem.n_params))
        _ = obj.vmap_value(jnp.zeros((2, self._problem.n_params)))

        obj.start_logging()

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

        # Evaluate initial samples
        train_Y_raw, valid_mask = self._evaluate_y(train_X, problem_bounds_torch, obj)
        train_Y_raw = train_Y_raw.unsqueeze(-1)

        # Filter to only valid points for GP training
        train_X = train_X[valid_mask]
        train_Y = train_Y_raw[valid_mask]

        if len(train_Y) == 0:
            raise ValueError(
                "All initial evaluations returned NaN/Inf. Check problem bounds and objective function."
            )

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

        # Main optimization loop
        iteration = 0
        while not obj.budget_exceeded and iteration < self.max_iterations:
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

            # Evaluate candidates
            new_Y_raw, valid_mask_batch = self._evaluate_y(
                candidates, problem_bounds_torch, obj
            )
            new_Y_raw = new_Y_raw.unsqueeze(-1)

            # Update training data with only valid points
            valid_candidates = candidates[valid_mask_batch]
            valid_Y = new_Y_raw[valid_mask_batch]
            if len(valid_Y) > 0:
                train_X = torch.cat([train_X, valid_candidates], dim=0)
                train_Y = torch.cat([train_Y, valid_Y], dim=0)

            iteration += 1

        # Outputs
        if plot_loss:
            obj.output_to_files()
        if save_run_to_file:
            obj.save_run_data()

        return obj
