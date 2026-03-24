"""SAASBO — Sparse Axis-Aligned Subspace BO via Ax/BoTorch.

Uses fully Bayesian GP inference with a sparsity-inducing half-Cauchy prior on
lengthscales, making it effective in high-dimensional spaces where only a few
dimensions matter.

Reference:
    Eriksson & Jankowiak, "High-Dimensional Bayesian Optimization with Sparse
    Axis-Aligned Subspaces", UAI 2021.

Package strategy: Ax ``Models.FULLYBAYESIAN`` schedules SAAS-GP fitting and
acquisition internally. We wrap the Ax ``Client`` (or ``GenerationStrategy``)
so every evaluation still routes through the ``Objective`` wrapper.

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
    sobol_initial_samples,
    unit_bounds_torch,
)

try:
    from ax.service.ax_client import AxClient
    from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
    from ax.modelbridge.registry import Models
    from ax.service.utils.instantiation import ObjectiveProperties

    _AX_AVAILABLE = True
except ImportError:
    _AX_AVAILABLE = False


class AxSAASBO(OptimizationAlgorithm):
    """Sparse Axis-Aligned Subspace BO via Ax fully-Bayesian GP.

    Uses Ax's ``FULLYBAYESIAN`` model which places a half-Cauchy prior on GP
    lengthscales, encouraging sparsity. Particularly effective for
    high-dimensional problems where only a subset of parameters matter.

    The algorithm operates in **bounded** parameter space.

    Attributes:
        algorithm_str: ``"ax_saasbo"``
        algorithm_type: ``SURROGATE_BASED``
    """

    algorithm_str: str = "ax_saasbo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        if not _AX_AVAILABLE:
            raise ImportError(
                "Ax is required for AxSAASBO. Install with: "
                "uv pip install 'ax-platform[botorch]'"
            )

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        n_initial: int = 10,
        max_iterations: int | None = None,
        num_warmup: int = 256,
        num_samples: int = 128,
        **ax_kwargs,
    ) -> None:
        """Run SAASBO.

        Args:
            problem_objective: Objective wrapper (mutated in place).
            init_params: Optional starting point (bounded space).
            random_seed: Seed for reproducibility.
            n_initial: Sobol initialisation budget.
            max_iterations: BO iterations after initialisation. Required.
            num_warmup: NUTS warm-up samples for the fully-Bayesian GP.
            num_samples: NUTS posterior samples.
            **ax_kwargs: Forwarded to the Ax model bridge.
        """
        if max_iterations is None:
            raise ValueError("max_iterations is required")

        obj = problem_objective
        problem = obj.problem
        dim = problem.n_params
        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        torch.manual_seed(random_seed)

        bounds_torch = get_problem_bounds_torch(problem)
        lb = np.asarray(problem.bounds[0])
        ub = np.asarray(problem.bounds[1])

        # ── Ax client setup ───────────────────────────────────────────
        gs = GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=n_initial),
                GenerationStep(
                    model=Models.FULLYBAYESIAN,
                    num_trials=-1,
                    model_kwargs={
                        "num_samples": num_samples,
                        "warmup_steps": num_warmup,
                        **ax_kwargs,
                    },
                ),
            ]
        )

        ax_client = AxClient(generation_strategy=gs, random_seed=random_seed, verbose_logging=False)

        parameters = [
            {
                "name": f"x{i}",
                "type": "range",
                "bounds": [float(lb[i]), float(ub[i])],
                "value_type": "float",
            }
            for i in range(dim)
        ]
        ax_client.create_experiment(
            name="saasbo",
            parameters=parameters,
            objectives={"loss": ObjectiveProperties(minimize=True)},
        )

        # ── JIT warmup ───────────────────────────────────────────────
        _ = obj.vmap_value(jnp.zeros((1, dim)))

        obj.start_logging()

        for _ in range(n_initial + max_iterations):
            if obj.budget_exceeded:
                break

            params_dict, trial_index = ax_client.get_next_trial()
            x_np = np.array([params_dict[f"x{i}"] for i in range(dim)])
            x_jax = jnp.asarray(x_np)

            loss_val = float(obj.value(x_jax))

            ax_client.complete_trial(
                trial_index=trial_index,
                raw_data={"loss": (loss_val, 0.0)},
            )
