"""qKG — Knowledge Gradient acquisition via BoTorch.

Knowledge Gradient maximises the expected *increase in value of the best
posterior mean* after observing a new point, providing a one-step Bayes-optimal
lookahead policy.

Reference:
    Wu & Frazier, "The Parallel Knowledge Gradient Method for Batch Bayesian
    Optimization", NeurIPS 2016.

Operates in **bounded** parameter space.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import torch
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
    from botorch.acquisition import qKnowledgeGradient
    from botorch.optim import optimize_acqf
    from botorch.generation import gen_candidates_scipy
    from botorch.utils.transforms import normalize

    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


class BotorchqKG(OptimizationAlgorithm):
    """Knowledge Gradient BO via BoTorch.

    Uses ``qKnowledgeGradient`` for one-step Bayes-optimal lookahead.
    Computationally more expensive per iteration than EI-family methods but
    can be more sample-efficient.

    Operates in **bounded** parameter space.

    Attributes:
        algorithm_str: ``"botorch_qkg"``
        algorithm_type: ``SURROGATE_BASED``
    """

    algorithm_str: str = "botorch_qkg"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        if not _BOTORCH_AVAILABLE:
            raise ImportError(
                "BoTorch is required for BotorchqKG. Install with: "
                "uv pip install botorch"
            )
        self.device = DEVICE
        self.dtype = DTYPE

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        n_initial: int = 10,
        batch_size: int = 1,
        max_iterations: int | None = None,
        num_fantasies: int = 16,
        **bo_kwargs,
    ) -> None:
        """Run knowledge-gradient BO.

        Args:
            problem_objective: Objective wrapper (mutated in place).
            init_params: Optional starting point (bounded).
            random_seed: Seed for reproducibility.
            n_initial: Sobol initialisation budget.
            batch_size: Candidates per iteration.
            max_iterations: BO iterations after initialisation. Required.
            num_fantasies: Number of fantasy models for KG estimation.
            **bo_kwargs: Forwarded to ``optimize_acqf``.
        """
        if max_iterations is None:
            raise ValueError("max_iterations is required")

        obj = problem_objective
        problem = obj.problem
        dim = problem.n_params

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)

        bounds = get_problem_bounds_torch(problem, self.device, self.dtype)
        u_bounds = unit_bounds_torch(dim, self.device, self.dtype)

        acqf_opts = {
            "raw_samples": bo_kwargs.get("raw_samples", 256),
            "num_restarts": bo_kwargs.get("num_restarts", 4),
            "retry_on_optimization_warning": False,
            "options": {"maxiter": 200, "batch_limit": 32},
        }

        # JIT warmup
        _ = obj.vmap_value(jnp.zeros((1, dim)))
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
        while not obj.budget_exceeded and iteration < max_iterations:
            model = fit_gp(train_X, train_Y)
            model.eval()

            acqf = qKnowledgeGradient(
                model,
                num_fantasies=num_fantasies,
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
