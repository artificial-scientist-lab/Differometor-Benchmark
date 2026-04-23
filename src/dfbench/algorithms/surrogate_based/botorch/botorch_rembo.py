"""REMBO — Random EMbedding Bayesian Optimization via BoTorch.

Projects the ambient space into a low-dimensional random subspace, runs
standard GP-BO there, and projects candidates back. The key idea is that
many high-dimensional objectives have low effective dimensionality.

Reference:
    Wang et al., "Bayesian Optimization in a Billion Dimensions via Random
    Embeddings", JAIR 2016.

This is a thin wrapper that changes the candidate-generation geometry, not
an entire bespoke surrogate stack. It reuses the BoTorch GP and qLogEI
infrastructure.

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
    from botorch.acquisition import qLogExpectedImprovement as qLogEI
    from botorch.optim import optimize_acqf
    from botorch.generation import gen_candidates_scipy
    from botorch.utils.transforms import normalize, unnormalize

    _BOTORCH_AVAILABLE = True
except ImportError:
    _BOTORCH_AVAILABLE = False


class REMBO(OptimizationAlgorithm):
    """Random EMbedding Bayesian Optimization.

    Maintains a fixed Gaussian random projection A ∈ R^{D×d_e} and performs
    GP-BO in the low-dimensional embedding space. Points are projected back
    to the ambient bounded space via clipping.

    Operates in **bounded** parameter space.

    Attributes:
        algorithm_str: ``"rembo"``
        algorithm_type: ``SURROGATE_BASED``
    """

    algorithm_str: str = "rembo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        if not _BOTORCH_AVAILABLE:
            raise ImportError(
                "BoTorch is required for REMBO. Install with: uv pip install botorch"
            )
        self.device = DEVICE
        self.dtype = DTYPE

    def _project_up(
        self,
        Z: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Project from embedding space to normalised ambient [0,1]^D.

        Z: (n, d_e), A: (D, d_e) → X: (n, D)
        Uses sigmoid to map unbounded projection outputs into [0, 1].
        """
        raw = Z @ A.T  # (n, D)
        return torch.sigmoid(raw)

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        n_initial: int = 10,
        max_iterations: int | None = None,
        d_embedding: int | None = None,
        **bo_kwargs,
    ) -> None:
        """Run REMBO.

        Args:
            problem_objective: Objective wrapper (mutated in place).
            init_params: Optional starting point (bounded).
            random_seed: Seed for reproducibility.
            n_initial: Sobol initialisation in embedding space.
            max_iterations: BO iterations after initialisation. Required.
            d_embedding: Embedding dimensionality. Defaults to ``min(10, dim)``.
            **bo_kwargs: Extra kwargs for acquisition optimisation.
        """
        if max_iterations is None:
            raise ValueError("max_iterations is required")

        obj = problem_objective
        problem = obj.problem
        D = problem.n_params

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)

        bounds = get_problem_bounds_torch(problem, self.device, self.dtype)

        if d_embedding is None:
            d_embedding = min(10, D)
        d_e = d_embedding

        # Fixed Gaussian random projection
        A = torch.randn(D, d_e, device=self.device, dtype=self.dtype) / np.sqrt(d_e)

        # Optimise in embedding space; define bounds as [-sqrt(d_e), sqrt(d_e)]
        emb_radius = float(np.sqrt(d_e))
        emb_bounds = torch.tensor(
            [[-emb_radius] * d_e, [emb_radius] * d_e],
            device=self.device,
            dtype=self.dtype,
        )

        acqf_opts = {
            "raw_samples": bo_kwargs.get("raw_samples", 256),
            "num_restarts": bo_kwargs.get("num_restarts", 4),
            "retry_on_optimization_warning": False,
            "options": {"maxiter": 200, "batch_limit": 32},
        }

        # JIT warmup
        _ = obj.vmap_value(jnp.zeros((1, D)))
        obj.start_logging()

        # Initial Sobol in embedding space
        sobol = torch.quasirandom.SobolEngine(d_e, scramble=True, seed=random_seed)
        Z_train = sobol.draw(n_initial).to(self.device, self.dtype)
        Z_train = Z_train * 2 * emb_radius - emb_radius  # map [0,1]→[-r,r]

        X_train = self._project_up(Z_train, A)

        Y_init, valid = evaluate_objective(X_train, bounds, obj)
        Z_train = Z_train[valid]
        Y_train = Y_init[valid].unsqueeze(-1)

        if len(Y_train) == 0:
            raise ValueError("All initial evaluations returned NaN/Inf.")

        iteration = 0
        while not obj.budget_exceeded and iteration < max_iterations:
            # Fit GP in embedding space
            model = fit_gp(Z_train, Y_train)
            model.eval()

            acqf = qLogEI(model, Y_train.max())
            Z_cand, _ = optimize_acqf(
                acqf,
                bounds=emb_bounds,
                q=1,
                gen_candidates=gen_candidates_scipy,
                **acqf_opts,
            )

            X_cand = self._project_up(Z_cand, A)
            Y_new, vm = evaluate_objective(X_cand, bounds, obj)

            if vm.any():
                Z_train = torch.cat([Z_train, Z_cand[vm]])
                Y_train = torch.cat([Y_train, Y_new[vm].unsqueeze(-1)])

            iteration += 1
