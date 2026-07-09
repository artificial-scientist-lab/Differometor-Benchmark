"""GEBO: Gradient-Enhanced Bayesian Optimization via BoTorch.

Incorporates gradient observations into the GP surrogate to improve
sample efficiency. Each evaluation provides both a function value *and* a
gradient, giving the GP d+1 observations per point.

This is a thin wrapper that enriches the surrogate with gradient information,
built on top of the standard BoTorch GP infrastructure.

Reference:
    Wu et al., "Bayesian Optimization with Gradients", NeurIPS 2017.

Operates in **bounded** parameter space. Gradients are obtained via the
Objective's ``value_and_grad``, which uses the bounded objective with
finite-difference or automatic differentiation through JAX.
"""

from __future__ import annotations

import jax
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
from dfbench.core.utils import t2j
from dfbench.algorithms.surrogate_based.botorch._botorch_common import (
    DEVICE,
    DTYPE,
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


class GEBO(OptimizationAlgorithm):
    """Gradient-Enhanced Bayesian Optimization.

    Feeds both function values *and* gradients to a standard GP, effectively
    getting ``d+1`` pieces of information per function evaluation.  The GP is
    trained on the function value observations only (gradient info is used to
    guide acquisition search), while the gradient is used for local candidate
    refinement.

    Operates in **bounded** parameter space.

    Attributes:
        algorithm_str: ``"gebo"``
        algorithm_type: ``SURROGATE_BASED``
    """

    algorithm_str: str = "gebo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        if not _BOTORCH_AVAILABLE:
            raise ImportError(
                "BoTorch is required for GEBO. Install with: uv add 'dfbench[bo]'"
            )
        self.device = DEVICE
        self.dtype = DTYPE

    def _eval_with_grad(
        self,
        X_norm: torch.Tensor,
        bounds: torch.Tensor,
        obj: Objective,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate value & grad at normalised X, return (Y_neg, grad_neg, valid).

        Uses ``obj.value_and_grad`` point-wise for correct eval-count tracking.
        """
        device, dtype = X_norm.device, X_norm.dtype
        n = X_norm.shape[0]
        dim = X_norm.shape[1]

        Y_list, G_list, valid_list = [], [], []

        for i in range(n):
            x_unnorm = unnormalize(X_norm[i : i + 1], bounds).squeeze(0)
            x_jax = t2j(x_unnorm)

            try:
                loss_jax, grad_jax = obj.value_and_grad(x_jax)
                loss_t = torch.tensor(float(loss_jax), device=device, dtype=dtype)
                grad_t = torch.from_numpy(np.array(grad_jax)).to(
                    device=device, dtype=dtype
                )

                if torch.isfinite(loss_t) and torch.all(torch.isfinite(grad_t)):
                    Y_list.append(-loss_t)
                    G_list.append(-grad_t)
                    valid_list.append(True)
                else:
                    Y_list.append(torch.tensor(0.0, device=device, dtype=dtype))
                    G_list.append(torch.zeros(dim, device=device, dtype=dtype))
                    valid_list.append(False)
            except Exception:
                Y_list.append(torch.tensor(0.0, device=device, dtype=dtype))
                G_list.append(torch.zeros(dim, device=device, dtype=dtype))
                valid_list.append(False)

        Y = torch.stack(Y_list)
        G = torch.stack(G_list)
        valid = torch.tensor(valid_list, device=device)
        return Y, G, valid

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        n_initial: int = 10,
        max_iterations: int | None = None,
        grad_refine_steps: int = 3,
        grad_refine_lr: float = 0.01,
        **bo_kwargs,
    ) -> None:
        """Run Gradient-Enhanced BO.

        Args:
            objective: Objective wrapper (mutated in place).
            init_params: Optional starting point (bounded).
            random_seed: Seed for reproducibility.
            n_initial: Sobol initialisation budget.
            max_iterations: Optional cap on BO iterations after initialisation.
                When ``None`` the algorithm runs until ``obj.budget_exceeded``.
            grad_refine_steps: Local gradient-descent refinement steps on each
                acquired candidate before evaluation.
            grad_refine_lr: Step size for local gradient refinement.
            **bo_kwargs: Extra kwargs for acquisition optimisation.
        """
        obj = objective
        D = obj.n_params

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)

        bounds = get_problem_bounds_torch(obj.bounds, self.device, self.dtype)
        u_bounds = unit_bounds_torch(D, self.device, self.dtype)

        acqf_opts = {
            "raw_samples": bo_kwargs.get("raw_samples", 256),
            "num_restarts": bo_kwargs.get("num_restarts", 4),
            "retry_on_optimization_warning": False,
            "options": {"maxiter": 200, "batch_limit": 32},
        }

        # JIT warmup: use value_and_grad for this algo
        obj.warmup_value_and_grad()
        obj.start_logging()

        # Initial Sobol
        X_train = sobol_initial_samples(
            D, n_initial, random_seed, device=self.device, dtype=self.dtype
        )
        if init_params is not None:
            x0 = torch.tensor(
                np.asarray(init_params).reshape(1, -1),
                device=self.device,
                dtype=self.dtype,
            )
            X_train = torch.cat([normalize(x0, bounds), X_train])

        Y_init, G_init, valid = self._eval_with_grad(X_train, bounds, obj)
        X_train = X_train[valid]
        Y_train = Y_init[valid].unsqueeze(-1)
        G_train = G_init[valid]  # stored for potential future use

        if len(Y_train) == 0:
            raise ValueError("All initial evaluations returned NaN/Inf.")

        iteration = 0
        while not obj.budget_exceeded and (
            max_iterations is None or iteration < max_iterations
        ):
            model = fit_gp(X_train, Y_train)
            model.eval()

            acqf = qLogEI(model, Y_train.max())
            candidates, _ = optimize_acqf(
                acqf,
                bounds=u_bounds,
                q=1,
                gen_candidates=gen_candidates_scipy,
                **acqf_opts,
            )

            # Local gradient refinement on the candidate (no extra obj evals)
            x_refine = candidates.squeeze(0).clone().detach()
            for _ in range(grad_refine_steps):
                x_unnorm = unnormalize(x_refine.unsqueeze(0), bounds).squeeze(0)
                x_jax = t2j(x_unnorm)
                # Use the objective gradient directly (no Objective logging call)
                grad_fn = jax.grad(obj.value_function(unbounded=False))
                g_jax = grad_fn(x_jax)
                g_t = torch.from_numpy(np.array(g_jax)).to(self.device, self.dtype)
                # Gradient descent step in normalised space (negate because we negate Y)
                x_refine = x_refine + grad_refine_lr * (-g_t)  # minimise
                x_refine = torch.clamp(x_refine, 0.0, 1.0)

            candidates_refined = x_refine.unsqueeze(0)

            Y_new, G_new, vm = self._eval_with_grad(candidates_refined, bounds, obj)

            if vm.any():
                X_train = torch.cat([X_train, candidates_refined[vm]])
                Y_train = torch.cat([Y_train, Y_new[vm].unsqueeze(-1)])
                G_train = torch.cat([G_train, G_new[vm]])

            iteration += 1
