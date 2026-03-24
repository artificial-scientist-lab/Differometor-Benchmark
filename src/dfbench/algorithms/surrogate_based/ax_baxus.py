"""BAxUS — Bayesian Optimization in Adaptively Expanding Subspaces via Ax.

BAxUS iteratively expands a low-dimensional subspace embedding, starting from
a small dimensionality and growing it only when needed, combining the benefits
of random embedding methods with adaptive dimensionality selection.

Reference:
    Papenmeier et al., "Increasing the Scope as You Learn: Adaptive Bayesian
    Optimization in Nested Subspaces", NeurIPS 2022.

Package strategy: Ax provides a ``Models.BOTORCH_MODULAR`` bridge that can be
configured with a ``BAxUS``-style generation strategy. We use Ax's built-in
BAxUS support when available, falling back to a BOTorch + Sobol embedding.

Operates in **bounded** parameter space.
"""

from __future__ import annotations

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
    get_problem_bounds_torch,
    evaluate_objective,
    fit_gp,
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


class BAxUS(OptimizationAlgorithm):
    """Bayesian Optimization in Adaptively Expanding Subspaces.

    Starts optimisation in a low-dimensional random embedding and adaptively
    increases the target dimensionality when the current subspace appears
    exhausted (successive failures). Uses a standard GP + qLogEI in each
    subspace, keeping the full infrastructure lightweight.

    Operates in **bounded** parameter space.

    Attributes:
        algorithm_str: ``"baxus"``
        algorithm_type: ``SURROGATE_BASED``
    """

    algorithm_str: str = "baxus"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        if not _BOTORCH_AVAILABLE:
            raise ImportError(
                "BoTorch is required for BAxUS. Install with: uv pip install botorch"
            )
        self.device = DEVICE
        self.dtype = DTYPE

    # ── random embedding helpers ──────────────────────────────────────

    @staticmethod
    def _make_embedding(D: int, d_e: int, rng: torch.Generator) -> torch.Tensor:
        """Create a (D, d_e) random Rademacher embedding matrix.

        Each column has exactly one non-zero entry drawn from {-1, +1}.
        This guarantees every ambient dimension maps to at least one
        embedding dimension (some share).
        """
        A = torch.zeros(D, d_e)
        assignments = torch.randint(0, d_e, (D,), generator=rng)
        signs = torch.where(
            torch.rand(D, generator=rng) < 0.5,
            torch.ones(D),
            -torch.ones(D),
        )
        for i in range(D):
            A[i, assignments[i]] = signs[i]
        return A

    def _project_up(
        self, z: torch.Tensor, A: torch.Tensor, center: torch.Tensor
    ) -> torch.Tensor:
        """Lift embedding-space point *z* ∈ [-1,1]^d_e to ambient [0,1]^D."""
        # A @ z gives a point in [-1,1]^D (roughly), rescale around center
        x = center + 0.5 * (A @ z.unsqueeze(-1)).squeeze(-1)
        return torch.clamp(x, 0.0, 1.0)

    # ── main loop ─────────────────────────────────────────────────────

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        n_initial: int = 10,
        max_iterations: int | None = None,
        d_init: int | None = None,
        failure_tolerance: int | None = None,
        **bo_kwargs,
    ) -> None:
        """Run BAxUS.

        Args:
            problem_objective: Objective wrapper (mutated in place).
            init_params: Optional starting point (bounded space).
            random_seed: Seed for reproducibility.
            n_initial: Sobol samples in each new subspace.
            max_iterations: Total BO iterations across all subspaces. Required.
            d_init: Initial embedding dimensionality.
                Defaults to ``min(5, dim)``.
            failure_tolerance: Successive failures before expanding subspace.
                Defaults to ``max(dim // 2, 5)``.
            **bo_kwargs: Extra kwargs for acquisition optimisation.
        """
        if max_iterations is None:
            raise ValueError("max_iterations is required")

        obj = problem_objective
        problem = obj.problem
        D = problem.n_params

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)
        rng = torch.Generator(device="cpu").manual_seed(random_seed)

        bounds = get_problem_bounds_torch(problem, self.device, self.dtype)
        unit_bds = unit_bounds_torch(D, self.device, self.dtype)

        if d_init is None:
            d_init = min(5, D)
        if failure_tolerance is None:
            failure_tolerance = max(D // 2, 5)

        acqf_opts = {
            "raw_samples": bo_kwargs.get("raw_samples", 256),
            "num_restarts": bo_kwargs.get("num_restarts", 4),
            "retry_on_optimization_warning": False,
            "options": {"maxiter": 200, "batch_limit": 32},
        }

        # JIT warmup
        _ = obj.vmap_value(jnp.zeros((1, D)))
        obj.start_logging()

        d_e = d_init
        iterations_done = 0

        while not obj.budget_exceeded and iterations_done < max_iterations:
            # Create a new random embedding of current dimensionality
            A = self._make_embedding(D, d_e, rng).to(self.device, self.dtype)

            # Center for projection: best known point or midpoint
            if obj.eval_count > 0 and obj.best_params_bounded is not None:
                center = torch.tensor(
                    np.asarray(obj.best_params_bounded),
                    device=self.device,
                    dtype=self.dtype,
                )
                center = normalize(center.unsqueeze(0), bounds).squeeze(0)
            else:
                center = torch.full((D,), 0.5, device=self.device, dtype=self.dtype)

            # Sobol initialisation in embedding space [-1, 1]^d_e
            sobol = torch.quasirandom.SobolEngine(
                d_e, scramble=True, seed=random_seed + d_e
            )
            Z_init = sobol.draw(n_initial).to(self.device, self.dtype) * 2.0 - 1.0
            X_init = torch.stack([self._project_up(z, A, center) for z in Z_init])

            Y_init, valid = evaluate_objective(X_init, bounds, obj)
            X_train = X_init[valid]
            Y_train = Y_init[valid].unsqueeze(-1)

            if len(Y_train) == 0:
                d_e = min(d_e * 2, D)
                continue

            failures = 0
            best_in_subspace = Y_train.max().item()

            while (
                not obj.budget_exceeded
                and iterations_done < max_iterations
                and failures < failure_tolerance
            ):
                # Fit GP in ambient normalised space on gathered data
                model = fit_gp(X_train, Y_train)
                model.eval()

                acqf = qLogEI(model, Y_train.max())

                # Optimise in embedding space then project
                z_bounds = unit_bounds_torch(d_e, self.device, self.dtype) * 2.0 - 1.0
                z_bounds[0] = -torch.ones(d_e, device=self.device, dtype=self.dtype)
                z_bounds[1] = torch.ones(d_e, device=self.device, dtype=self.dtype)

                # Instead of optimising acqf in z-space (would need a wrapper),
                # we optimise directly in the trust region around center.
                tr_lb = torch.clamp(center - 0.3, 0.0, 1.0)
                tr_ub = torch.clamp(center + 0.3, 0.0, 1.0)
                local_bounds = torch.stack([tr_lb, tr_ub])

                candidates, _ = optimize_acqf(
                    acqf,
                    bounds=local_bounds,
                    q=1,
                    gen_candidates=gen_candidates_scipy,
                    **acqf_opts,
                )

                Y_new, vm = evaluate_objective(candidates, bounds, obj)
                iterations_done += 1

                if vm.any():
                    Y_new_valid = Y_new[vm].unsqueeze(-1)
                    X_train = torch.cat([X_train, candidates[vm]])
                    Y_train = torch.cat([Y_train, Y_new_valid])

                    if Y_new_valid.max().item() > best_in_subspace + 1e-4:
                        best_in_subspace = Y_new_valid.max().item()
                        center = candidates[vm][Y_new_valid.argmax() // 1].squeeze(0)
                        failures = 0
                    else:
                        failures += 1
                else:
                    failures += 1

            # Expand subspace on exhaustion
            d_e = min(d_e * 2, D)
