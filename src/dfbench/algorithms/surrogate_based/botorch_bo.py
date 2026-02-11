"""State-of-the-art Bayesian Optimization using BoTorch with batch acquisition."""

import secrets

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

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.utils import t2j
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
        device (torch.device): PyTorch device (cuda if available, else cpu).
        dtype (torch.dtype): PyTorch dtype for tensors (float64 for numerical stability).

    Note:
        This algorithm searches in the bounded parameter space using `problem.objective_function`.
        Bounds are always taken from `problem.bounds`.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = BotorchBO()
        >>> objective = optimizer.optimize(
        ...     problem_objective=objective,
        ...     max_iterations=100,
        ...     n_initial=10,
        ...     batch_size=5,
        ... )
    """

    algorithm_str: str = "botorch_bo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        """Initialize BoTorch Bayesian Optimization.
        
        No configuration parameters needed - all settings are provided
        at optimization time via the optimize() method.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64

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
        problem_objective: Objective,
        max_iterations: int | None = None,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        n_initial: int = 10,
        batch_size: int = 1,
        **bo_kwargs,
    ) -> Objective:
        """Run Bayesian Optimization with batch acquisition.

        Args:
            problem_objective: The Objective instance wrapping the problem.
            max_iterations: Maximum number of BO iterations (excluding initial samples).
                Required parameter.
            init_params: Initial parameters to include in the training set.
                If None, only Sobol samples are used. Defaults to None.
            random_seed: Random seed for reproducibility. Defaults to None.
            n_initial: Number of initial Sobol samples before fitting GP.
                Defaults to 10.
            batch_size: Number of points to acquire per iteration. Defaults to 1.
            **bo_kwargs: Additional keyword arguments for acquisition optimization.

        Returns:
            The Objective instance with all logged data.
        """
        obj = problem_objective
        problem = obj.problem

        if max_iterations is None:
            raise ValueError("max_iterations is required")

        self.setup_objective(obj, unbounded=False, random_seed=random_seed)

        if random_seed is None:
            random_seed = secrets.randbits(32)
        obj.set_seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        print(f"Random seed: {random_seed}")

        # Get bounds from problem
        lb_np = np.asarray(problem.bounds[0])
        ub_np = np.asarray(problem.bounds[1])

        problem_bounds_torch = torch.tensor(
            np.array([lb_np, ub_np]), device=self.device, dtype=self.dtype
        )

        unit_bounds = torch.stack(
            [
                torch.zeros(
                    problem.n_params, dtype=self.dtype, device=self.device
                ),
                torch.ones(
                    problem.n_params, dtype=self.dtype, device=self.device
                ),
            ],
            dim=0,
        )

        # Warmup JIT (vmap_value is used for batch evaluation in _evaluate_y)
        _ = obj.vmap_value(jnp.zeros((1, problem.n_params)))

        obj.start_logging()

        # Generate initial Sobol samples
        sobol = torch.quasirandom.SobolEngine(
            dimension=problem.n_params, scramble=True, seed=random_seed
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
        while not obj.budget_exceeded and iteration < max_iterations:
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

        return obj
