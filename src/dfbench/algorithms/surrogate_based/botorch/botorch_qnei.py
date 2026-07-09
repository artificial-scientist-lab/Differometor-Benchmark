"""qNEI: Noisy Expected Improvement via BoTorch.

Uses ``qLogNoisyExpectedImprovement`` (the numerically stable log variant)
which accounts for observation noise in the acquisition function, making it
more robust for real-world noisy objectives than standard qEI.

Reference:
    Letham et al., "Noisy Expected Improvement", NeurIPS 2019.
    Ament et al., "Unexpected Improvements to Expected Improvement", 2023.

Operates in **bounded** parameter space.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "torch is required for this algorithm. Install with:  uv add 'dfbench[bo]'"
    ) from exc
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective
from dfbench.algorithms.surrogate_based.botorch._botorch_common import (
    DEVICE,
    DTYPE,
    evaluate_objective,
    fit_gp,
    get_problem_bounds_torch,
    sobol_initial_samples,
    unit_bounds_torch,
)

try:
    from botorch.acquisition import qLogNoisyExpectedImprovement as qLogNEI_acqf
    from botorch.optim import optimize_acqf
    from botorch.generation import gen_candidates_scipy
    from botorch.utils.transforms import normalize

    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


class BotorchQNEI(OptimizationAlgorithm):
    """Noisy Expected Improvement BO via BoTorch.

    Wraps ``qLogNoisyExpectedImprovement`` which conditions on the training data
    to compute the improvement relative to observed (noisy) best, rather than
    assuming a noiseless baseline. Uses the log variant for numerical stability.

    Operates in **bounded** parameter space.

    Attributes:
        algorithm_str: ``"botorch_qnei"``
        algorithm_type: ``SURROGATE_BASED``
    """

    algorithm_str: str = "botorch_qnei"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        if not _BOTORCH_AVAILABLE:
            raise ImportError(
                "BoTorch is required for BotorchQNEI. Install with: uv add 'dfbench[bo]'"
            )
        self.device = DEVICE
        self.dtype = DTYPE

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        n_initial: int = 10,
        batch_size: int = 1,
        max_iterations: int | None = None,
        prune_baseline: bool = True,
        **bo_kwargs,
    ) -> None:
        """Run qNEI.

        Args:
            objective: Objective wrapper (mutated in place).
            init_params: Optional starting point (bounded).
            random_seed: Seed for reproducibility.
            n_initial: Sobol initialisation budget.
            batch_size: Candidates per iteration.
            max_iterations: Optional cap on BO iterations after initialisation.
                When ``None`` the algorithm runs until ``obj.budget_exceeded``.
            prune_baseline: Whether to prune the baseline set for qNEI.
            **bo_kwargs: Forwarded to ``optimize_acqf``.
        """
        obj = objective
        dim = obj.n_params

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)

        bounds = get_problem_bounds_torch(obj.bounds, self.device, self.dtype)
        u_bounds = unit_bounds_torch(dim, self.device, self.dtype)

        acqf_opts = {
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

        # JIT warmup
        obj.warmup_vmap_value(batch_size=1)
        obj.start_logging()

        # Initial Sobol
        train_X = sobol_initial_samples(
            dim, n_initial, random_seed, device=self.device, dtype=self.dtype
        )
        if init_params is not None:
            x0 = torch.tensor(
                np.asarray(init_params).reshape(1, -1),
                device=self.device,
                dtype=self.dtype,
            )
            train_X = torch.cat([normalize(x0, bounds), train_X])

        Y_init, valid = evaluate_objective(train_X, bounds, obj)
        train_X = train_X[valid]
        train_Y = Y_init[valid].unsqueeze(-1)

        if len(train_Y) == 0:
            raise ValueError("All initial evaluations returned NaN/Inf.")

        iteration = 0
        while not obj.budget_exceeded and (
            max_iterations is None or iteration < max_iterations
        ):
            model = fit_gp(train_X, train_Y)
            model.eval()

            acqf = qLogNEI_acqf(
                model,
                X_baseline=train_X,
                prune_baseline=prune_baseline,
            )
            candidates, _ = optimize_acqf(
                acqf,
                bounds=u_bounds,
                q=batch_size,
                gen_candidates=gen_candidates_scipy,
                **acqf_opts,
            )

            Y_new, vm = evaluate_objective(candidates, bounds, obj)
            Y_new = Y_new.unsqueeze(-1)

            if vm.any():
                train_X = torch.cat([train_X, candidates[vm]])
                train_Y = torch.cat([train_Y, Y_new[vm]])

            iteration += 1
