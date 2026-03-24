"""LineBO — Line Bayesian Optimization via BoTorch.

Restricts each BO iteration to a 1-D line through the ambient space.  The
line direction rotates between random directions and the gradient at the
incumbent, trading off exploration and exploitation naturally.

Reference:
    Kirschner et al., "Adaptive and Safe Bayesian Optimization in High
    Dimensions via One-Dimensional Subspaces", ICML 2019.

This is a thin wrapper that changes the candidate-generation geometry.
It reuses the BoTorch GP and qLogEI infrastructure.

Operates in **bounded** parameter space.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective
from dfbench.core.utils import t2j
from dfbench.algorithms.surrogate_based._botorch_common import (
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


class LineBO(OptimizationAlgorithm):
    """Line Bayesian Optimization.

    At each iteration, selects a 1-D line through the current best point and
    runs 1-D GP-BO along that line.  The direction alternates between a random
    direction and a coordinate direction (or the gradient when available).

    Operates in **bounded** parameter space.

    Attributes:
        algorithm_str: ``"linebo"``
        algorithm_type: ``SURROGATE_BASED``
    """

    algorithm_str: str = "linebo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        if not _BOTORCH_AVAILABLE:
            raise ImportError(
                "BoTorch is required for LineBO. Install with: uv pip install botorch"
            )
        self.device = DEVICE
        self.dtype = DTYPE

    @staticmethod
    def _line_bounds(
        center: torch.Tensor,
        direction: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute the range of scalar *t* such that center + t*direction ∈ [0,1]^D."""
        t_lo = torch.full_like(center, -float("inf"))
        t_hi = torch.full_like(center, float("inf"))

        pos = direction > 1e-12
        neg = direction < -1e-12

        t_lo[pos] = -center[pos] / direction[pos]
        t_hi[pos] = (1.0 - center[pos]) / direction[pos]

        t_lo[neg] = (1.0 - center[neg]) / direction[neg]
        t_hi[neg] = -center[neg] / direction[neg]

        t_min = float(t_lo.max())
        t_max = float(t_hi.min())
        return max(t_min, -5.0), min(t_max, 5.0)

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        n_initial: int = 10,
        max_iterations: int | None = None,
        line_samples: int = 20,
        **bo_kwargs,
    ) -> None:
        """Run LineBO.

        Args:
            problem_objective: Objective wrapper (mutated in place).
            init_params: Optional starting point (bounded).
            random_seed: Seed for reproducibility.
            n_initial: Initial Sobol samples in full ambient space.
            max_iterations: BO iterations (one line per iteration). Required.
            line_samples: Number of 1-D Sobol points sampled along each line.
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

        # JIT warmup
        _ = obj.vmap_value(jnp.zeros((1, D)))
        obj.start_logging()

        # Initial Sobol in full ambient space
        X_train = sobol_initial_samples(D, n_initial, random_seed, device=self.device, dtype=self.dtype)
        if init_params is not None:
            x0 = torch.tensor(
                np.asarray(init_params).reshape(1, -1), device=self.device, dtype=self.dtype
            )
            X_train = torch.cat([normalize(x0, bounds), X_train])

        Y_init, valid = evaluate_objective(X_train, bounds, obj)
        X_train = X_train[valid]
        Y_train = Y_init[valid].unsqueeze(-1)

        if len(Y_train) == 0:
            raise ValueError("All initial evaluations returned NaN/Inf.")

        iteration = 0
        while not obj.budget_exceeded and iteration < max_iterations:
            # Current best (in normalised space)
            center = X_train[Y_train.argmax().item()].clone()

            # Choose direction: alternate random / coordinate
            if iteration % 2 == 0:
                d = torch.randn(D, device=self.device, dtype=self.dtype)
            else:
                coord_idx = iteration % D
                d = torch.zeros(D, device=self.device, dtype=self.dtype)
                d[coord_idx] = 1.0
            d = d / d.norm()

            # Compute feasible line segment
            t_lo, t_hi = self._line_bounds(center, d)
            if t_hi - t_lo < 1e-8:
                iteration += 1
                continue

            # Sample along the line
            sobol_1d = torch.quasirandom.SobolEngine(1, scramble=True, seed=random_seed + iteration)
            T = sobol_1d.draw(line_samples).to(self.device, self.dtype).squeeze(-1)
            T = T * (t_hi - t_lo) + t_lo  # map [0,1]→[t_lo, t_hi]

            X_line = center.unsqueeze(0) + T.unsqueeze(-1) * d.unsqueeze(0)
            X_line = torch.clamp(X_line, 0.0, 1.0)

            Y_line, vm_line = evaluate_objective(X_line, bounds, obj)

            if vm_line.any():
                X_train = torch.cat([X_train, X_line[vm_line]])
                Y_train = torch.cat([Y_train, Y_line[vm_line].unsqueeze(-1)])

            # Now fit 1-D GP on the line and pick the best unseen point
            T_valid = T[vm_line].unsqueeze(-1)
            Y_line_valid = Y_line[vm_line].unsqueeze(-1)

            if len(T_valid) >= 3:
                model_1d = fit_gp(T_valid, Y_line_valid)
                model_1d.eval()

                acqf = qLogEI(model_1d, Y_line_valid.max())
                t_bounds_1d = torch.tensor(
                    [[t_lo], [t_hi]], device=self.device, dtype=self.dtype
                )
                t_cand, _ = optimize_acqf(
                    acqf,
                    bounds=t_bounds_1d,
                    q=1,
                    raw_samples=64,
                    num_restarts=2,
                    retry_on_optimization_warning=False,
                )

                x_cand = center + t_cand.squeeze() * d
                x_cand = torch.clamp(x_cand, 0.0, 1.0).unsqueeze(0)

                Y_new, vm_new = evaluate_objective(x_cand, bounds, obj)
                if vm_new.any():
                    X_train = torch.cat([X_train, x_cand[vm_new]])
                    Y_train = torch.cat([Y_train, Y_new[vm_new].unsqueeze(-1)])

            iteration += 1
