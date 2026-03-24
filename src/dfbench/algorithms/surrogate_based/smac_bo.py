"""SMAC — Sequential Model-based Algorithm Configuration via SMAC3.

SMAC3 combines random forests as surrogate models with aggressive racing for
early termination and is the de-facto standard for algorithm configuration and
hyperparameter tuning.

Reference:
    Lindauer et al., "SMAC3: A versatile Bayesian optimization package for
    hyperparameter optimization", JMLR 2022.

Operates in **bounded** parameter space.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective

try:
    from ConfigSpace import ConfigurationSpace, Float as CS_Float
    from smac import HyperparameterOptimizationFacade, Scenario

    _SMAC_AVAILABLE = True
except ImportError:
    _SMAC_AVAILABLE = False


class SMAC(OptimizationAlgorithm):
    """SMAC — Sequential Model-based Algorithm Configuration.

    Wraps SMAC3's ``HyperparameterOptimizationFacade`` with a random-forest
    surrogate to work with the ``Objective`` protocol. SMAC handles bounded
    continuous spaces natively.

    Operates in **bounded** parameter space.

    Attributes:
        algorithm_str: ``"smac"``
        algorithm_type: ``SURROGATE_BASED``
    """

    algorithm_str: str = "smac"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        if not _SMAC_AVAILABLE:
            raise ImportError(
                "SMAC3 is required. Install with: uv pip install smac"
            )

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        max_iterations: int | None = None,
        n_initial: int = 10,
        **smac_kwargs,
    ) -> None:
        """Run SMAC.

        Args:
            problem_objective: Objective wrapper (mutated in place).
            init_params: Optional starting point (bounded).
            random_seed: Seed for reproducibility.
            max_iterations: Total evaluations (Sobol + BO). Required.
            n_initial: Initial random configurations.
            **smac_kwargs: Forwarded to SMAC Scenario.
        """
        if max_iterations is None:
            raise ValueError("max_iterations is required")

        obj = problem_objective
        problem = obj.problem
        dim = problem.n_params
        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb = np.asarray(problem.bounds[0])
        ub = np.asarray(problem.bounds[1])

        # Build ConfigSpace
        cs = ConfigurationSpace(seed=random_seed)
        for i in range(dim):
            cs.add(CS_Float(f"x{i}", bounds=(float(lb[i]), float(ub[i]))))

        # JIT warmup
        _ = obj.vmap_value(jnp.zeros((1, dim)))
        obj.start_logging()

        # Target function called by SMAC
        def _target(config, seed: int = 0):
            if obj.budget_exceeded:
                # Return a large value to signal termination
                return float("inf")
            x_np = np.array([config[f"x{i}"] for i in range(dim)])
            x_jax = jnp.asarray(x_np)
            return float(obj.value(x_jax))

        scenario = Scenario(
            cs,
            deterministic=True,
            n_trials=n_initial + max_iterations,
            seed=random_seed,
            **{k: v for k, v in smac_kwargs.items() if k not in ("cs", "deterministic", "n_trials", "seed")},
        )

        smac = HyperparameterOptimizationFacade(
            scenario,
            _target,
            overwrite=True,
        )

        # Run SMAC optimisation loop — it controls its own budget internally
        smac.optimize()
