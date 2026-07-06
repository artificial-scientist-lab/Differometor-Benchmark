"""Trust Region Bayesian Optimization (TuRBO) using BoTorch.

TuRBO maintains a local trust region around the best point and adapts its size
based on optimization progress. This makes it particularly effective for
high-dimensional optimization problems where global BO may struggle.

Reference:
    Eriksson, David, et al. "Scalable global optimization via local Bayesian
    optimization." Advances in Neural Information Processing Systems. 2019.
"""

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from botorch.models import SingleTaskGP
from botorch.acquisition import qLogExpectedImprovement as qLogEI
from botorch.generation import MaxPosteriorSampling
from botorch.utils.transforms import normalize
from jaxtyping import Array, Float
from typing import Literal

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective

from dfbench.algorithms.utils.gp import fit_gp, optimize_acqfn
from dfbench.algorithms.utils.misc import evaluate_y
from dfbench.algorithms.utils.initial_design import create_initial_design


@dataclass
class TurboState:
    """State for Trust Region Bayesian Optimization.

    Maintains the trust region length, success/failure counters, and best value.
    """

    dim: int
    acquisition_batch_size: int
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
        if self.acquisition_batch_size < 1:
            raise ValueError("acquisition_batch_size must be at least 1.")

        self.failure_tolerance = math.ceil(
            max(
                4.0 / self.acquisition_batch_size,
                float(self.dim) / self.acquisition_batch_size,
            )
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
        device (torch.device): PyTorch device (cuda if available, else cpu).
        dtype (torch.dtype): PyTorch dtype for tensors.
        max_cholesky_size (float): Maximum Cholesky matrix size for GP fitting.
    """

    algorithm_str: str = "botorch_turbo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self, batch_size: int = 1) -> None:
        """Initialize BoTorch TuRBO Optimization.

        Args:
            batch_size: Number of candidates evaluated per ``vmap_value`` call.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self.max_cholesky_size = float("inf")
        self.batch_size = batch_size

    def _generate_batch(
        self,
        state: TurboState,
        model: SingleTaskGP,
        X: torch.Tensor,
        Y: torch.Tensor,
        acquisition_batch_size: int,
        n_candidates: int | None = None,
        num_restarts: int = 10,
        raw_samples: int | None = None,
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
        bounds = torch.stack([tr_lb, tr_ub])

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
                X_next = thompson_sampling(X_cand, num_samples=acquisition_batch_size)

        elif acqf == "ei":
            ei = qLogEI(
                model=model,
                best_f=Y.max(),
            )
            X_next, _ = optimize_acqfn(
                acquisition_function=ei,
                bounds=bounds,
                q=acquisition_batch_size,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )

        return X_next

    def optimize(
        self,
        objective: Objective,
        max_iterations: int | None = None,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        n_initial: int | None = None,
        acquisition_batch_size: int = 1,
        num_restarts: int = 10,
        raw_samples: int | None = None,
        acqf: Literal["ts", "ei"] = "ts",
        n_restarts: int | None = None,
        **turbo_kwargs,
    ) -> None:
        """Run TuRBO optimization with adaptive trust regions.

        Args:
            objective: The Objective instance wrapping the problem.
            max_iterations: Optional cap on BO iterations per TuRBO instance.
                When ``None`` the algorithm runs until ``obj.budget_exceeded``
                (or a TuRBO restart triggers).
            init_params: Initial parameters to include in the training set.
            random_seed: Random seed for reproducibility.
            n_initial: Number of initial Sobol samples. Defaults to 2 * dim.
            batch_size: Number of points to acquire per iteration. Defaults to 1.
            num_restarts: Number of random restarts for multistart optimization.
            raw_samples: Number of raw samples for initialization.
            acquisition_batch_size: Number of points to acquire per iteration.
                Defaults to 1.
            acqf: Acquisition function type ("ts" or "ei"). Defaults to "ts".
            n_restarts: Maximum number of TuRBO restarts. When ``None``,
                restart until the objective budget is exhausted. Defaults to
                ``None``.
            **turbo_kwargs: Additional keyword arguments.
        """
        if acquisition_batch_size < 1:
            raise ValueError("acquisition_batch_size must be at least 1.")

        obj = objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)

        dim = problem.n_params

        if n_initial is None:
            n_initial = 2 * dim

        # Get bounds from problem
        lb_np = np.asarray(problem.bounds[0])
        ub_np = np.asarray(problem.bounds[1])

        problem_bounds_torch = torch.tensor(
            np.array([lb_np, ub_np]), device=self.device, dtype=self.dtype
        )

        # TuRBO-specific parameters
        n_candidates = turbo_kwargs.get("n_candidates", min(5000, max(2000, 200 * dim)))
        length_init = turbo_kwargs.get("length_init", 0.8)
        length_min = turbo_kwargs.get("length_min", 0.5**7)
        length_max = turbo_kwargs.get("length_max", 1.6)
        success_tolerance = turbo_kwargs.get("success_tolerance", 10)

        # Warmup JIT (vmap_value is used for batch evaluation in _evaluate_y)
        obj.warmup_vmap_value(batch_size=self.batch_size)

        obj.start_logging()

        def run_turbo_instance(init_X: torch.Tensor | None = None):
            """Run a single TuRBO instance until restart is triggered or budget exhausted."""

            train_X = create_initial_design(
                dimensions=problem.n_params,
                n_initial=n_initial,
                random_seed=random_seed,
            ).to(device=self.device, dtype=self.dtype)

            if init_X is not None:
                train_X = torch.cat([init_X, train_X], dim=0)

            train_Y_raw, valid_mask = evaluate_y(
                X=train_X,
                bounds=problem_bounds_torch,
                obj=obj,
                batch_size=self.batch_size,
            )
            train_Y_raw = train_Y_raw.unsqueeze(-1)

            train_X = train_X[valid_mask]
            train_Y = train_Y_raw[valid_mask]

            if len(train_Y) == 0:
                raise ValueError("All initial evaluations returned NaN/Inf.")

            state = TurboState(
                dim=dim,
                acquisition_batch_size=acquisition_batch_size,
                length=length_init,
                length_min=length_min,
                length_max=length_max,
                success_tolerance=success_tolerance,
                best_value=train_Y.max().item(),
            )

            iteration = 0
            while (
                not obj.budget_exceeded
                and not state.restart_triggered
                and (max_iterations is None or iteration < max_iterations)
            ):
                # Normalize Y for GP fitting
                Y_mean = train_Y.mean()
                Y_std = train_Y.std()
                if Y_std < 1e-6:
                    Y_std = torch.tensor(1.0, device=self.device, dtype=self.dtype)
                train_Y_normalized = (train_Y - Y_mean) / Y_std

                # Fit GP and generate batch
                model = fit_gp(
                    train_X=train_X,
                    train_Y=train_Y_normalized,
                    tr_modeling=True,
                    use_turbo_constraints=True,
                )
                model.eval()

                X_next = self._generate_batch(
                    state=state,
                    model=model,
                    X=train_X,
                    Y=train_Y_normalized,
                    acquisition_batch_size=acquisition_batch_size,
                    n_candidates=n_candidates,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    acqf=acqf,
                )

                # Evaluate new candidates
                Y_next_raw, valid_mask_next = evaluate_y(
                    X=X_next,
                    bounds=problem_bounds_torch,
                    obj=obj,
                    batch_size=self.batch_size,
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

        while not obj.budget_exceeded and (
            n_restarts is None or restart_count < n_restarts
        ):
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
