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
import jax.numpy as jnp
import numpy as np
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
from jaxtyping import Array, Float
from typing import Literal

from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)
from dfbench.core.utils import t2j_numpy as t2j
from dfbench.core.objective import Objective


@dataclass
class TurboState:
    """State for Trust Region Bayesian Optimization.

    Maintains the trust region length, success/failure counters, and best value.
    """

    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = field(init=False)
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = float("-inf")
    restart_triggered: bool = False

    def __post_init__(self) -> None:
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )


def update_turbo_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    """Update the TuRBO state based on new observations."""
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())

    if state.length < state.length_min:
        state.restart_triggered = True

    return state


class BotorchTuRBO(OptimizationAlgorithm):
    """Trust Region Bayesian Optimization using BoTorch.

    Implements TuRBO-1 which maintains a single trust region centered on the
    best point found. The trust region adapts its size based on optimization
    progress: expanding after successes and shrinking after failures.

    All history tracking is handled by the `Objective` wrapper.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("botorch_turbo").
        algorithm_type (AlgorithmType): Type classification (SURROGATE_BASED).
        _problem (ContinuousProblem): The optimization problem instance.
        max_iterations (int): Maximum number of BO iterations.
        device (torch.device): PyTorch device (cuda if available, else cpu).
        dtype (torch.dtype): PyTorch dtype for tensors.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = BotorchTuRBO(problem, max_iterations=100)
        >>> objective = optimizer.optimize(
        ...     max_time=120,
        ...     n_initial=10,
        ...     batch_size=4,
        ...     acqf="ts",
        ... )
    """

    algorithm_str: str = "botorch_turbo"
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
        """Initialize BoTorch TuRBO Optimization.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
            max_iterations (int): Maximum number of BO iterations. Defaults to 100.
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
        self.max_cholesky_size = float("inf")
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
        """Evaluate objective function at given input(s) through Objective wrapper."""
        unnormalized_X = unnormalize(X, bounds)
        X_jax = t2j(unnormalized_X)

        if X_jax.ndim == 1:
            Y_jax = obj.value(X_jax)
            Y_torch = torch.tensor([Y_jax.item()], device=X.device, dtype=X.dtype)
        else:
            Y_jax = obj.vmap_value(X_jax)
            Y_torch = torch.from_numpy(np.array(Y_jax)).to(
                device=X.device, dtype=X.dtype
            )

        invalid_mask = torch.isnan(Y_torch) | torch.isinf(Y_torch)

        if torch.any(invalid_mask) and max_retries > 0:
            invalid_indices = torch.where(invalid_mask)[0]
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

        valid_mask = ~invalid_mask
        return -Y_torch, valid_mask

    def _fit_model(
        self,
        train_X: torch.Tensor,
        train_Y: torch.Tensor,
        use_turbo_constraints: bool = True,
    ) -> SingleTaskGP:
        """Fit GP model with TuRBO-specific kernel settings."""
        dim = train_X.shape[-1]

        def create_turbo_model():
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

        def create_simple_model():
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            return model, mll

        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)

                if use_turbo_constraints:
                    try:
                        model, mll = create_turbo_model()
                        fit_gpytorch_mll(mll)
                        return model
                    except ModelFittingError:
                        pass

                try:
                    model, mll = create_simple_model()
                    fit_gpytorch_mll(mll)
                    return model
                except ModelFittingError:
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
        """Generate a new batch of candidate points within the trust region."""
        assert acqf in ("ts", "ei")
        assert X.min() >= 0.0 and X.max() <= 1.0
        assert torch.all(torch.isfinite(Y))

        dim = X.shape[-1]
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * dim))

        x_center = X[Y.argmax(), :].clone()

        try:
            base_kernel = model.covar_module.base_kernel
            if hasattr(base_kernel, "lengthscale"):
                lengthscale = base_kernel.lengthscale
            elif hasattr(base_kernel, "base_kernel") and hasattr(
                base_kernel.base_kernel, "lengthscale"
            ):
                lengthscale = base_kernel.base_kernel.lengthscale
            else:
                lengthscale = torch.ones(dim, device=self.device, dtype=self.dtype)

            weights = lengthscale.squeeze().detach()
            if weights.dim() == 0:
                weights = weights.expand(dim)
            elif len(weights) != dim:
                weights = torch.ones(dim, device=self.device, dtype=self.dtype)

            weights = weights / weights.mean()
            weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        except (AttributeError, RuntimeError):
            weights = torch.ones(dim, device=self.device, dtype=self.dtype)

        tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

        if acqf == "ts":
            sobol = torch.quasirandom.SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
            pert = tr_lb + (tr_ub - tr_lb) * pert

            prob_perturb = min(20.0 / dim, 1.0)
            mask = (
                torch.rand(n_candidates, dim, dtype=self.dtype, device=self.device)
                <= prob_perturb
            )
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            if len(ind) > 0:
                mask[
                    ind, torch.randint(0, dim, size=(len(ind),), device=self.device)
                ] = True

            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]

            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = thompson_sampling(X_cand, num_samples=batch_size)

        elif acqf == "ei":
            ei = qLogEI(model, Y.max())
            X_next, _ = optimize_acqf(
                ei,
                bounds=torch.stack([tr_lb, tr_ub]),
                q=batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next

    def optimize(
        self,
        use_problem_bounds: bool = True,
        init_params: Float[Array, "{self._problem.n_params}"] | None = None,
        random_seed: int | None = None,
        max_time: float | None = None,
        lb: Float[Array, "{self._problem.n_params}"] | None = None,
        ub: Float[Array, "{self._problem.n_params}"] | None = None,
        n_initial: int | None = None,
        batch_size: int = 4,
        acqf: Literal["ts", "ei"] = "ts",
        n_restarts: int = 1,
        verbose: int | None = None,
        print_every: int = 10,
        plot_loss: bool = False,
        save_run_to_file: bool = False,
        **turbo_kwargs,
    ) -> Objective:
        """Run TuRBO optimization with adaptive trust regions.

        Args:
            use_problem_bounds: If True, use bounds from `problem.bounds`.
            init_params: Initial parameters to include in the training set.
            random_seed: Random seed for reproducibility.
            max_time: Time budget in seconds. None for unlimited.
            lb: Lower bounds for each parameter. Ignored if use_problem_bounds=True.
            ub: Upper bounds for each parameter. Ignored if use_problem_bounds=True.
            n_initial: Number of initial Sobol samples. Defaults to 2 * dim.
            batch_size: Number of points to acquire per iteration. Defaults to 4.
            acqf: Acquisition function type ("ts" or "ei"). Defaults to "ts".
            n_restarts: Number of TuRBO restarts. Defaults to 1.
            verbose: Verbosity level (0=silent, 1+=prints via Objective).
            print_every: Print summary every N evaluations.
            plot_loss: If True, call obj.output_to_files for plotting.
            save_run_to_file: If True, call obj.save_run_data for checkpointing.
            **turbo_kwargs: Additional keyword arguments.

        Returns:
            The Objective instance with all logged data.
        """
        if random_seed is not None:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        dim = self._problem.n_params

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

        problem_bounds_torch = torch.tensor(
            np.array([lb_np, ub_np]), device=self.device, dtype=self.dtype
        )

        # TuRBO-specific parameters
        n_candidates = turbo_kwargs.get("n_candidates", min(5000, max(2000, 200 * dim)))
        num_restarts = turbo_kwargs.get("num_restarts", 10)
        raw_samples = turbo_kwargs.get("raw_samples", 512)
        length_init = turbo_kwargs.get("length_init", 0.8)
        length_min = turbo_kwargs.get("length_min", 0.5**7)
        length_max = turbo_kwargs.get("length_max", 1.6)
        success_tolerance = turbo_kwargs.get("success_tolerance", 10)

        # Create Objective wrapper
        obj = Objective(
            self._problem,
            unbounded=False,
            max_time=max_time,
            max_evals=(self.max_iterations + n_initial) * batch_size * n_restarts,
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
        _ = obj.value(jnp.zeros(dim))
        _ = obj.vmap_value(jnp.zeros((2, dim)))

        obj.start_logging()

        def run_turbo_instance(init_X: torch.Tensor | None = None):
            """Run a single TuRBO instance until restart is triggered or budget exhausted."""
            sobol = torch.quasirandom.SobolEngine(
                dimension=dim, scramble=True, seed=random_seed
            )
            train_X = sobol.draw(n=n_initial).to(dtype=self.dtype, device=self.device)

            if init_X is not None:
                train_X = torch.cat([init_X, train_X], dim=0)

            train_Y_raw, valid_mask = self._evaluate_y(
                train_X, problem_bounds_torch, obj
            )
            train_Y_raw = train_Y_raw.unsqueeze(-1)

            train_X = train_X[valid_mask]
            train_Y = train_Y_raw[valid_mask]

            if len(train_Y) == 0:
                raise ValueError(
                    "All initial evaluations returned NaN/Inf."
                )

            state = TurboState(
                dim=dim,
                batch_size=batch_size,
                length=length_init,
                length_min=length_min,
                length_max=length_max,
                success_tolerance=success_tolerance,
                best_value=train_Y.max().item(),
            )

            iteration = 0
            while not obj.budget_exceeded and not state.restart_triggered and iteration < self.max_iterations:
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

                # Evaluate new candidates
                Y_next_raw, valid_mask_next = self._evaluate_y(
                    X_next, problem_bounds_torch, obj
                )
                Y_next_raw = Y_next_raw.unsqueeze(-1)

                # Update state only with valid values
                if torch.any(valid_mask_next):
                    valid_Y_for_state = Y_next_raw[valid_mask_next]
                    state = update_turbo_state(state, valid_Y_for_state)
                else:
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

                iteration += 1

            return not obj.budget_exceeded and state.restart_triggered

        # Main optimization with restarts
        restart_count = 0

        while not obj.budget_exceeded and restart_count < n_restarts:
            init_X = None
            if restart_count == 0 and init_params is not None:
                init_X_unnorm = torch.tensor(
                    np.array(init_params).reshape(1, -1),
                    device=self.device,
                    dtype=self.dtype,
                )
                init_X = normalize(init_X_unnorm, problem_bounds_torch)

            should_restart = run_turbo_instance(init_X)
            restart_count += 1

            if not should_restart:
                break

        # Outputs
        if plot_loss:
            obj.output_to_files()
        if save_run_to_file:
            obj.save_run_data()

        return obj
