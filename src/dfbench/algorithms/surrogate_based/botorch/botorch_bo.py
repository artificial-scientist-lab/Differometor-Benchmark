"""State-of-the-art Bayesian Optimization using BoTorch with batch acquisition."""

import numpy as np
try:
    import torch
except ImportError as exc:
    raise ImportError(
        "torch is required for this algorithm. Install with:  uv add 'dfbench[bo]'"
    ) from exc
from botorch.acquisition import qLogExpectedImprovement as qLogEI
from botorch.utils.transforms import normalize
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective
from dfbench.algorithms.utils.gp import fit_gp, optimize_acqfn
from dfbench.algorithms.utils.misc import evaluate_y
from dfbench.algorithms.utils.initial_design import create_initial_design


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
        >>> optimizer = BotorchBO(batch_size=5)
        >>> objective = optimizer.optimize(
        ...     objective=objective,
        ...     max_iterations=100,
        ...     n_initial=10,
        ...     acquisition_batch_size=5,
        ... )
    """

    algorithm_str: str = "botorch_bo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self, batch_size: int = 1) -> None:
        """Initialize BoTorch Bayesian Optimization.

        Args:
            batch_size: Number of candidates evaluated per ``vmap_value`` call.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self.batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        max_iterations: int | None = None,
        init_params: Float[Array, "n_params"] | None = None,
        num_restarts: int = 20,
        raw_samples: int | None = None,
        random_seed: int | None = None,
        n_initial: int = 10,
        acquisition_batch_size: int = 1,
        **bo_kwargs,
    ) -> None:
        """Run Bayesian Optimization with batch acquisition.

        Args:
            objective: The Objective instance wrapping the problem.
            max_iterations: Optional cap on BO iterations (excluding initial samples).
                When ``None`` the algorithm runs until ``obj.budget_exceeded``.
            init_params: Initial parameters to include in the training set.
                If None, only Sobol samples are used. Defaults to None.
            random_seed: Random seed for reproducibility. Defaults to None.
            n_initial: Number of initial Sobol samples before fitting GP.
                Defaults to 10.
            acquisition_batch_size: Number of points to acquire per iteration.
                Defaults to 1.
            **bo_kwargs: Additional keyword arguments for acquisition optimization.
        """
        if acquisition_batch_size < 1:
            raise ValueError("acquisition_batch_size must be at least 1.")

        obj = objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)

        # Get bounds from problem
        lb_np = np.asarray(problem.bounds[0])
        ub_np = np.asarray(problem.bounds[1])

        problem_bounds_torch = torch.tensor(
            np.array([lb_np, ub_np]), device=self.device, dtype=self.dtype
        )

        unit_bounds = torch.stack(
            [
                torch.zeros(problem.n_params, dtype=self.dtype, device=self.device),
                torch.ones(problem.n_params, dtype=self.dtype, device=self.device),
            ],
            dim=0,
        )

        # Warmup JIT (vmap_value is used for batch evaluation in _evaluate_y)
        obj.warmup_vmap_value(batch_size=self.batch_size)

        obj.start_logging()

        # Create initial design
        train_X = create_initial_design(
            dimensions=problem.n_params,
            n_initial=n_initial,
            random_seed=random_seed,
            sampler="sobol",
        ).to(device=self.device, dtype=self.dtype)

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
        train_Y_raw, valid_mask = evaluate_y(
            X=train_X,
            bounds=problem_bounds_torch,
            obj=obj,
            batch_size=self.batch_size,
        )
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
        while not obj.budget_exceeded and (
            max_iterations is None or iteration < max_iterations
        ):
            # Fit GP model
            model = fit_gp(train_X, train_Y)
            model.eval()

            # Optimize acquisition function
            # acqf = qLogNEI(
            #     model=model,
            #     X_baseline=train_X,
            # )
            acqf = qLogEI(
                model=model,
                best_f=train_Y.min(),
            )

            candidates, _ = optimize_acqfn(
                acquisition_function=acqf,
                bounds=unit_bounds,
                q=acquisition_batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
                seed=random_seed,
                acqf_options=acqf_options,
            )

            # Evaluate candidates
            new_Y_raw, valid_mask_batch = evaluate_y(
                candidates,
                problem_bounds_torch,
                obj,
                batch_size=self.batch_size,
            )
            new_Y_raw = new_Y_raw.unsqueeze(-1)

            # Update training data with only valid points
            valid_candidates = candidates[valid_mask_batch]
            valid_Y = new_Y_raw[valid_mask_batch]
            if len(valid_Y) > 0:
                train_X = torch.cat([train_X, valid_candidates], dim=0)
                train_Y = torch.cat([train_Y, valid_Y], dim=0)

            iteration += 1
