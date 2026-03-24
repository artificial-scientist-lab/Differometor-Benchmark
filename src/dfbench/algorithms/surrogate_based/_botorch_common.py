"""Shared BoTorch utilities for surrogate-based algorithms.

Centralises GP fitting, objective evaluation through the ``Objective`` wrapper,
and torch/jax conversion so individual algorithm files stay thin.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import torch
from botorch.exceptions.errors import ModelFittingError
from botorch.exceptions.warnings import OptimizationWarning
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import gpytorch

from dfbench.core.objective import Objective
from dfbench.core.utils import t2j


# ── Device / dtype helpers ────────────────────────────────────────────

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def get_problem_bounds_torch(problem, device=DEVICE, dtype=DTYPE):
    """Return problem bounds as a (2, d) torch tensor."""
    lb = np.asarray(problem.bounds[0])
    ub = np.asarray(problem.bounds[1])
    return torch.tensor(np.stack([lb, ub]), device=device, dtype=dtype)


def unit_bounds_torch(dim: int, device=DEVICE, dtype=DTYPE):
    """Return a (2, dim) tensor of [0, 1] bounds."""
    return torch.stack(
        [
            torch.zeros(dim, device=device, dtype=dtype),
            torch.ones(dim, device=device, dtype=dtype),
        ]
    )


# ── Evaluation ────────────────────────────────────────────────────────


def evaluate_objective(
    X: torch.Tensor,
    bounds: torch.Tensor,
    obj: Objective,
    *,
    negate: bool = True,
    max_retries: int = 3,
    perturbation_scale: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate ``obj`` at normalised inputs *X* ∈ [0, 1]^d.

    Returns ``(Y, valid_mask)`` where *Y* is shape ``(n,)`` and
    *valid_mask* is a boolean tensor of the same shape.  When *negate*
    is ``True`` (default) the values are negated so BoTorch can
    maximise them.

    Invalid (NaN / Inf) points are retried up to *max_retries* times
    with small perturbations.
    """
    unnormalised = unnormalize(X, bounds)
    X_jax = t2j(unnormalised)
    device, dtype = X.device, X.dtype

    if X_jax.ndim == 1:
        y = obj.value(X_jax)
        Y = torch.tensor([y.item()], device=device, dtype=dtype)
    else:
        y = obj.vmap_value(X_jax)
        Y = torch.from_numpy(np.array(y)).to(device=device, dtype=dtype)

    invalid = torch.isnan(Y) | torch.isinf(Y)

    if torch.any(invalid) and max_retries > 0:
        for idx in torch.where(invalid)[0]:
            for retry in range(max_retries):
                xp = X[idx].clone()
                xp += torch.randn_like(xp) * perturbation_scale * (retry + 1)
                xp = torch.clamp(xp, 0.0, 1.0)
                xp_jax = t2j(unnormalize(xp.unsqueeze(0), bounds).squeeze(0))
                yr = obj.value(xp_jax)
                yr_t = torch.tensor(yr.item(), device=device, dtype=dtype)
                if torch.isfinite(yr_t):
                    Y[idx] = yr_t
                    invalid[idx] = False
                    break

    if negate:
        Y = -Y
    return Y, ~invalid


# ── GP fitting ────────────────────────────────────────────────────────


def fit_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    *,
    turbo_constraints: bool = False,
) -> SingleTaskGP:
    """Fit a ``SingleTaskGP`` to the given data.

    When *turbo_constraints* is ``True`` the GP uses a constrained Matérn
    kernel and noise, matching the TuRBO paper.  Falls back to a vanilla GP
    if the constrained fit fails.
    """
    dim = train_X.shape[-1]

    def _turbo_model():
        lik = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        cov = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=dim,
                lengthscale_constraint=Interval(0.005, 4.0),
            ),
        )
        m = SingleTaskGP(train_X, train_Y, covar_module=cov, likelihood=lik)
        mll = ExactMarginalLogLikelihood(m.likelihood, m)
        return m, mll

    def _simple_model():
        m = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(m.likelihood, m)
        return m, mll

    with gpytorch.settings.max_cholesky_size(float("inf")):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            if turbo_constraints:
                try:
                    model, mll = _turbo_model()
                    fit_gpytorch_mll(mll)
                    return model
                except ModelFittingError:
                    pass
            try:
                model, mll = _simple_model()
                fit_gpytorch_mll(mll)
                return model
            except ModelFittingError:
                model, _ = _simple_model()
                return model


# ── Initial Sobol samples ─────────────────────────────────────────────


def sobol_initial_samples(
    dim: int,
    n: int,
    seed: int,
    *,
    device=DEVICE,
    dtype=DTYPE,
) -> torch.Tensor:
    """Draw *n* scrambled Sobol points in [0,1]^dim."""
    sobol = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    return sobol.draw(n=n).to(device=device, dtype=dtype)
