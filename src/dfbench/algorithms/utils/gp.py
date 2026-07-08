from __future__ import annotations
from typing import Any, Mapping

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "torch is required for this algorithm. Install with:  uv add 'dfbench[bo]'"
    ) from exc
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf
import gpytorch
import warnings
from botorch.exceptions.errors import ModelFittingError
from botorch.exceptions.warnings import OptimizationWarning

from dfbench.algorithms.utils.weighted_acq import WeightedAcquisitionFunction


def fit_gp(
    train_X: torch.Tensor,
    train_Y: torch.Tensor,
    *,
    tr_modeling: bool = False,
    use_turbo_constraints: bool = False,
    max_cholesky_size: float = np.inf,
) -> SingleTaskGP:
    """Fit GP model to current data.

    Args:
        train_X: Training inputs of shape (n, d).
        train_Y: Training targets of shape (n, 1).

    Returns:
        Trained Gaussian Process model.
    """

    gp = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        outcome_transform=Standardize(m=train_Y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(
        likelihood=gp.likelihood,
        model=gp,
    )

    if tr_modeling:

        def create_turbo_model():
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=train_X.shape[-1],
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

        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=OptimizationWarning)

                if use_turbo_constraints:
                    try:
                        gp, mll = create_turbo_model()
                        fit_gpytorch_mll(mll)
                        return gp
                    except ModelFittingError:
                        pass

                try:
                    fit_gpytorch_mll(mll)
                    return gp
                except ModelFittingError:
                    return gp  # Return unfitted model if fitting fails

    fit_gpytorch_mll(mll)
    return gp


def optimize_acqfn(
    acquisition_function: AcquisitionFunction,
    bounds: torch.Tensor,
    q: int = 1,
    num_restarts: int = 20,
    raw_samples: int | None = None,
    gamma_exp_term: float | None = None,
    prior: Any | None = None,
    seed: int | None = None,
    acqf_options: Mapping[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimize the acquisition function.

    Args:
        acquisition_function: Acquisition function to optimize.
        bounds: Bounds for optimization of shape (2, d).
        q: Number of candidates to optimize for.
        num_restarts: Number of random restarts for multistart optimization.
        raw_samples: Number of raw samples for initialization.
        gamma_exp_term: gamma exponent for weighted acquisition function
        prior: Prior information for optimization
        seed: Random seed for reproducibility. If None, no seed is set.
        acqf_options: Additional options for acquisition function optimization.

    Returns:
        Tuple of (optimal points, acquisition values at optimal points).
    """
    if prior:
        assert gamma_exp_term is not None, (
            "gamma_exp_term must be provided when using a prior."
        )
        acquisition_function = WeightedAcquisitionFunction(
            acq_fn=acquisition_function,
            gamma_exp_term=gamma_exp_term,
            prior=prior,
        )

    if raw_samples is None:
        raw_samples = min(64 * len(bounds) ** 2, 4096)

    return optimize_acqf(
        acq_function=acquisition_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        **acqf_options,
    )
