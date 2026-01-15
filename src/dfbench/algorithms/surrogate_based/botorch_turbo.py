"""Trust Region Bayesian Optimization (TuRBO) using BoTorch.

TuRBO maintains a local trust region around the best point and adapts its size
based on optimization progress. This makes it particularly effective for
high-dimensional optimization problems where global BO may struggle.

Reference:
    Eriksson, David, et al. "Scalable global optimization via local Bayesian
    optimization." Advances in Neural Information Processing Systems. 2019.
"""

import math
import warnings
import jax
import jax.numpy as jnp
import numpy as np
import time
import torch
from dataclasses import dataclass, field
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qLogExpectedImprovement as qLogEI
from botorch.optim import optimize_acqf
from botorch.generation import MaxPosteriorSampling
from botorch.utils.transforms import unnormalize, normalize
from botorch.exceptions.errors import ModelFittingError
from botorch.exceptions.warnings import OptimizationWarning
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
import gpytorch
from collections import deque
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker
from typing import Literal

from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)
from dfbench.core.utils import t2j_numpy as t2j


@dataclass
class TurboState:
    """State for Trust Region Bayesian Optimization.

    Maintains the trust region length, success/failure counters, and best value.
    The trust region expands after successive improvements and shrinks after
    consecutive failures. Optimization restarts when the trust region becomes
    too small.

    Attributes:
        dim (int): Dimensionality of the search space.
        batch_size (int): Number of points acquired per iteration.
        length (float): Current trust region side length (in [0,1] space).
        length_min (float): Minimum trust region length before restart.
        length_max (float): Maximum trust region length.
        failure_counter (int): Count of consecutive failures.
        failure_tolerance (int): Number of failures before shrinking.
        success_counter (int): Count of consecutive successes.
        success_tolerance (int): Number of successes before expanding.
        best_value (float): Best objective value found so far.
        restart_triggered (bool): Whether a restart has been triggered.
    """

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7  # ~0.0078
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = field(init=False)
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = float("-inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        """Post-initialize the failure tolerance based on dimension and batch size."""
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )


def update_turbo_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    """Update the TuRBO state based on new observations.

    Expands the trust region after consecutive successes, shrinks after
    consecutive failures, and triggers restart when the region becomes too small.

    Args:
        state: Current TuRBO state.
        Y_next: New objective values from the latest batch (maximization).

    Returns:
        Updated TuRBO state.
    """
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    # Expand trust region after enough successes
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    # Shrink trust region after enough failures
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())

    # Trigger restart if trust region is too small
    if state.length < state.length_min:
        state.restart_triggered = True

    return state


class BotorchTuRBO(OptimizationAlgorithm):
    """Trust Region Bayesian Optimization using BoTorch.

    Implements TuRBO-1 which maintains a single trust region centered on the
    best point found. The trust region adapts its size based on optimization
    progress: expanding after successes and shrinking after failures.

    This is particularly effective for high-dimensional problems where standard
    BO struggles due to over-exploration.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("botorch_turbo").
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
        >>> optimizer = BotorchTurbo(problem, max_iterations=100)
        >>> best_params, history, losses, wall_indices = optimizer.optimize(
        ...     wall_times=[30, 60, 120],
        ...     n_initial=10,
        ...     batch_size=4,
        ...     acqf="ts",  # Thompson Sampling or "ei" for Expected Improvement
        ... )
    """

    algorithm_str: str = "botorch_turbo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self, problem: ContinuousProblem, max_iterations: int = 100) -> None:
        """Initialize BoTorch TuRBO Optimization.

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
        self.max_cholesky_size = float("inf")  # Always use Cholesky

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
        use_turbo_constraints: bool = True,
    ) -> SingleTaskGP:
        """Fit GP model with TuRBO-specific kernel settings.

        Uses a Matern 5/2 kernel with ARD and constrained lengthscales as
        recommended in the TuRBO paper. Falls back to simpler GP if fitting fails.

        Args:
            train_X: Training inputs of shape (n, d).
            train_Y: Training targets of shape (n, 1), normalized.
            use_turbo_constraints: Whether to use TuRBO's tight constraints.
                If fitting fails, will retry with relaxed constraints.

        Returns:
            Trained Gaussian Process model.
        """
        dim = train_X.shape[-1]

        def create_turbo_model() -> tuple[SingleTaskGP, ExactMarginalLogLikelihood]:
            """Create GP with TuRBO-specific constraints."""
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=dim,
                    lengthscale_constraint=Interval(0.005, 4.0),
                )
            )
            model = SingleTaskGP(
                train_X,
                train_Y,
                covar_module=covar_module,
                likelihood=likelihood,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            return model, mll

        def create_simple_model() -> tuple[SingleTaskGP, ExactMarginalLogLikelihood]:
            """Create simple GP without tight constraints (fallback)."""
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            return model, mll

        # Try fitting with TuRBO constraints first
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)
                
                if use_turbo_constraints:
                    try:
                        model, mll = create_turbo_model()
                        fit_gpytorch_mll(mll)
                        return model
                    except ModelFittingError:
                        print("Warning: TuRBO GP fitting failed, trying with relaxed constraints...")
                
                # Fallback: try simple GP without tight constraints
                try:
                    model, mll = create_simple_model()
                    fit_gpytorch_mll(mll)
                    return model
                except ModelFittingError:
                    print("Warning: Simple GP fitting also failed, using unfitted model...")
                    # Return unfitted model as last resort
                    model, _ = create_simple_model()
                    return model

    def _generate_batch(
        self,
        state: TurboState,
        model: SingleTaskGP,
        X: torch.Tensor,
        Y: torch.Tensor,
        batch_size: int,
        n_candidates: int | None = None,
        num_restarts: int = 10,
        raw_samples: int = 512,
        acqf: Literal["ts", "ei"] = "ts",
    ) -> torch.Tensor:
        """Generate a new batch of candidate points within the trust region.

        Constructs a trust region around the current best point, scaled by the
        GP lengthscales. Uses either Thompson Sampling (TS) or Expected
        Improvement (EI) to select candidates.

        Args:
            state: Current TuRBO state with trust region parameters.
            model: Fitted GP model.
            X: All evaluated points in [0,1] space.
            Y: Corresponding objective values (normalized).
            batch_size: Number of points to generate.
            n_candidates: Number of candidates for Thompson sampling.
            num_restarts: Number of restarts for acquisition optimization.
            raw_samples: Number of raw samples for initialization.
            acqf: Acquisition function type ("ts" or "ei").

        Returns:
            Batch of new candidate points in [0,1] space.
        """
        assert acqf in ("ts", "ei"), f"acqf must be 'ts' or 'ei', got {acqf}"
        assert X.min() >= 0.0 and X.max() <= 1.0, "X must be in [0,1]"
        assert torch.all(torch.isfinite(Y)), "Y must be finite"

        dim = X.shape[-1]
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * dim))

        # Find center of trust region (best point)
        x_center = X[Y.argmax(), :].clone()

        # Scale trust region by GP lengthscales for anisotropic regions
        # Handle different kernel structures (ScaleKernel wrapping various base kernels)
        try:
            # Try TuRBO-style: ScaleKernel(MaternKernel) or ScaleKernel(RBFKernel)
            base_kernel = model.covar_module.base_kernel
            if hasattr(base_kernel, 'lengthscale'):
                lengthscale = base_kernel.lengthscale
            elif hasattr(base_kernel, 'base_kernel') and hasattr(base_kernel.base_kernel, 'lengthscale'):
                lengthscale = base_kernel.base_kernel.lengthscale
            else:
                # Fallback to uniform weights
                lengthscale = torch.ones(dim, device=self.device, dtype=self.dtype)
            
            weights = lengthscale.squeeze().detach()
            # Ensure weights is 1D with correct dimension
            if weights.dim() == 0:
                weights = weights.expand(dim)
            elif len(weights) != dim:
                weights = torch.ones(dim, device=self.device, dtype=self.dtype)
                
            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        except (AttributeError, RuntimeError):
            # If anything fails, use uniform weights
            weights = torch.ones(dim, device=self.device, dtype=self.dtype)

        # Compute trust region bounds
        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "ts":
            # Thompson Sampling
            sobol = torch.quasirandom.SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Create perturbation mask - probability of perturbing each dimension
            prob_perturb = min(20.0 / dim, 1.0)
            mask = (
                torch.rand(n_candidates, dim, dtype=self.dtype, device=self.device)
                <= prob_perturb
            )
            # Ensure at least one dimension is perturbed
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            if len(ind) > 0:
                mask[ind, torch.randint(0, dim, size=(len(ind),), device=self.device)] = True

            # Create candidates from center + perturbations
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            # Sample from posterior
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        elif acqf == "ei":
            # Log Expected Improvement
            ei = qLogEI(model, Y.max())
            X_next, _ = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next

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
        n_initial: int | None = None,
        batch_size: int = 4,
        acqf: Literal["ts", "ei"] = "ts",
        n_restarts: int = 1,
        **turbo_kwargs,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_iters {self._problem.n_params}"] | None,
        Float[Array, "n_iters"],
        list[int] | None,
    ]:
        """Run TuRBO optimization with adaptive trust regions.

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
            n_initial (int | None): Number of initial Sobol samples before fitting GP.
                Defaults to 2 * dim.
            batch_size (int): Number of points to acquire per iteration. Defaults to 4.
            acqf (Literal["ts", "ei"]): Acquisition function type. "ts" for Thompson
                Sampling (faster, better for high-dim), "ei" for Expected Improvement.
                Defaults to "ts".
            n_restarts (int): Number of TuRBO restarts when trust region shrinks too much.
                Defaults to 1 (no restarts, runs until convergence or max_iterations).
            **turbo_kwargs: Additional keyword arguments:
                - n_candidates (int): Number of candidates for TS (default: min(5000, max(2000, 200*dim)))
                - num_restarts (int): Restarts for acqf optimization (default: 10)
                - raw_samples (int): Raw samples for acqf optimization (default: 512)
                - length_init (float): Initial trust region length (default: 0.8)
                - length_min (float): Minimum trust region length (default: 0.5^7)
                - length_max (float): Maximum trust region length (default: 1.6)
                - success_tolerance (int): Successes before expanding (default: 10)

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

        dim = self._problem.n_params

        # Set default n_initial based on dimension
        if n_initial is None:
            n_initial = 2 * dim

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
            lb_np = np.full(dim, -10.0) if lb is None else np.array(lb)
            ub_np = np.full(dim, 10.0) if ub is None else np.array(ub)

        # Convert to torch tensors
        problem_bounds_torch = torch.tensor(
            np.array([lb_np, ub_np]), device=self.device, dtype=self.dtype
        )

        # Warmup JIT compilation
        _ = self._problem.objective_function(jnp.zeros(dim))

        # TuRBO-specific parameters
        n_candidates = turbo_kwargs.get("n_candidates", min(5000, max(2000, 200 * dim)))
        num_restarts = turbo_kwargs.get("num_restarts", 10)
        raw_samples = turbo_kwargs.get("raw_samples", 512)
        length_init = turbo_kwargs.get("length_init", 0.8)
        length_min = turbo_kwargs.get("length_min", 0.5**7)
        length_max = turbo_kwargs.get("length_max", 1.6)
        success_tolerance = turbo_kwargs.get("success_tolerance", 10)

        # Initialize tracking
        best_params = jnp.zeros(dim)
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

        def get_initial_points(n_pts: int, seed: int | None = None) -> torch.Tensor:
            """Generate initial Sobol points in [0,1]^d."""
            sobol = torch.quasirandom.SobolEngine(
                dimension=dim, scramble=True, seed=seed
            )
            return sobol.draw(n=n_pts).to(dtype=self.dtype, device=self.device)

        def run_turbo_instance(
            start_idx: int,
            init_X: torch.Tensor | None = None,
        ) -> tuple[int, bool]:
            """Run a single TuRBO instance until restart is triggered or budget exhausted.

            Returns:
                Tuple of (current iteration index, whether to continue).
            """
            nonlocal best_params, best_loss, losses, best_params_history
            nonlocal wall_times_remaining, wall_time_indices

            # Generate initial points
            train_X = get_initial_points(n_initial, random_seed)

            # Include init_params if provided
            if init_X is not None:
                train_X = torch.cat([init_X, train_X], dim=0)

            # Evaluate initial points (returns Y values and validity mask)
            train_Y_raw, valid_mask = self._evaluate_y(train_X, problem_bounds_torch)
            train_Y_raw = train_Y_raw.unsqueeze(-1)

            # Record all losses (including invalid as NaN for tracking)
            for idx, y in enumerate(train_Y_raw):
                if valid_mask[idx]:
                    loss = -float(y.item())
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

            # Initialize TuRBO state with best valid value
            state = TurboState(
                dim=dim,
                batch_size=batch_size,
                length=length_init,
                length_min=length_min,
                length_max=length_max,
                success_tolerance=success_tolerance,
                best_value=train_Y.max().item(),
            )

            i = start_idx + len(train_Y)

            def should_continue() -> bool:
                if wall_times is not None:
                    return (time.time() - start_time) < max_wall_time
                else:
                    return i < total_iterations

            while should_continue() and not state.restart_triggered:
                if wall_times is not None:
                    elapsed = time.time() - start_time
                    # Record iteration index at wall_times checkpoints
                    while wall_times_remaining and elapsed >= wall_times_remaining[0]:
                        wall_time_indices.append(i)
                        wall_times_remaining.popleft()

                # Normalize Y for GP fitting
                Y_mean = train_Y.mean()
                Y_std = train_Y.std()
                if Y_std < 1e-6:
                    Y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
                train_Y_normalized = (train_Y - Y_mean) / Y_std

                # Fit GP and generate batch
                model = self._fit_model(train_X, train_Y_normalized)
                model.eval()

                X_next = self._generate_batch(
                    state=state,
                    model=model,
                    X=train_X,
                    Y=train_Y_normalized,
                    batch_size=batch_size,
                    n_candidates=n_candidates,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    acqf=acqf,
                )

                # Evaluate new candidates (returns Y values and validity mask)
                Y_next_raw, valid_mask_next = self._evaluate_y(X_next, problem_bounds_torch)
                Y_next_raw = Y_next_raw.unsqueeze(-1)

                # Update state only with valid values
                if torch.any(valid_mask_next):
                    valid_Y_for_state = Y_next_raw[valid_mask_next]
                    state = update_turbo_state(state, valid_Y_for_state)
                else:
                    # All invalid - count as failure
                    state.failure_counter += 1
                    if state.failure_counter >= state.failure_tolerance:
                        state.length /= 2.0
                        state.failure_counter = 0
                    if state.length < state.length_min:
                        state.restart_triggered = True

                # Update training data with only valid points
                valid_X_next = X_next[valid_mask_next]
                valid_Y_next = Y_next_raw[valid_mask_next]
                if len(valid_Y_next) > 0:
                    train_X = torch.cat([train_X, valid_X_next], dim=0)
                    train_Y = torch.cat([train_Y, valid_Y_next], dim=0)

                # Process batch results (track all for logging, even invalid)
                for j, y in enumerate(Y_next_raw):
                    if valid_mask_next[j]:
                        loss = -float(y.item())

                        if i % 10 == 0 and j == 0:
                            print(
                                f"Iteration {i}: Loss = {loss:.4e}, "
                                f"TR length = {state.length:.2e}"
                            )

                        if loss < best_loss - 1e-4:
                            best_loss = loss
                            # Find the index in the valid subset
                            valid_idx_count = int(valid_mask_next[:j+1].sum().item()) - 1
                            actual_train_idx = len(train_X) - len(valid_Y_next) + valid_idx_count
                            best_params = t2j(
                                unnormalize(train_X[actual_train_idx], problem_bounds_torch)
                            )
                            print(f"Iteration {i}: New best loss = {loss:.4e}")

                    # Store best_loss so far (not current loss)
                    losses.append(best_loss)

                    if return_best_params_history:
                        best_params_history.append(best_params)

                    i += 1

                    if not wall_times and i >= total_iterations:
                        break

            if state.restart_triggered:
                print(f"TuRBO restart triggered at iteration {i} (TR length < {length_min:.2e})")

            return i, should_continue() and state.restart_triggered

        # Main optimization
        if wall_times is not None:
            start_time = time.time()
            total_iterations = None  # Not used for wall-time mode

            current_idx = 0
            restart_count = 0

            while (time.time() - start_time) < max_wall_time and restart_count < n_restarts:
                init_X = None
                if restart_count == 0 and init_params is not None:
                    init_X_unnorm = torch.tensor(
                        np.array(init_params).reshape(1, -1),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    init_X = normalize(init_X_unnorm, problem_bounds_torch)

                current_idx, should_restart = run_turbo_instance(current_idx, init_X)
                restart_count += 1

                if not should_restart:
                    break

            # Fill remaining wall_times
            while wall_times_remaining:
                wall_time_indices.append(len(losses) - 1 if losses else 0)
                wall_times_remaining.popleft()

        else:
            total_iterations = self.max_iterations + n_initial
            current_idx = 0
            restart_count = 0

            while current_idx < total_iterations and restart_count < n_restarts:
                init_X = None
                if restart_count == 0 and init_params is not None:
                    init_X_unnorm = torch.tensor(
                        np.array(init_params).reshape(1, -1),
                        device=self.device,
                        dtype=self.dtype,
                    )
                    init_X = normalize(init_X_unnorm, problem_bounds_torch)

                current_idx, should_restart = run_turbo_instance(current_idx, init_X)
                restart_count += 1

                if not should_restart:
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
                hyper_param_str=f"n_initial{n_initial}_batch{batch_size}_acqf{acqf}",
                hyper_param_str_in_filename=True,
            )

        return best_params, best_params_history_array, losses_array, wall_time_indices
