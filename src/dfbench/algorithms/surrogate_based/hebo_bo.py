"""HEBO: Heteroscedastic Evolutionary Bayesian Optimization.

Uses the ``hebo`` package which won the NeurIPS 2020 BBO challenge. HEBO
employs a heteroscedastic GP, input warping, and a multi-objective
acquisition function that balances exploration and exploitation.

Reference:
    Cowen-Rivers et al., "An Empirical Study of Assumptions in Bayesian
    Optimisation", 2020.

Operates in **bounded** parameter space.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective

try:
    import pandas as pd
    from hebo.design_space.design_space import DesignSpace
    from hebo.optimizers.hebo import HEBO as HEBOOptimizer

    _HEBO_AVAILABLE = True
except ImportError:
    _HEBO_AVAILABLE = False


class HEBO(OptimizationAlgorithm):
    """HEBO: Heteroscedastic Evolutionary Bayesian Optimization.

    Wraps the ``hebo`` package to work with the ``Objective`` protocol.
    HEBO internally uses a heteroscedastic GP with input warping and a
    multi-objective acquisition to select candidates.

    Operates in **bounded** parameter space.

    Attributes:
        algorithm_str: ``"hebo"``
        algorithm_type: ``SURROGATE_BASED``
    """

    algorithm_str: str = "hebo"
    algorithm_type: AlgorithmType = AlgorithmType.SURROGATE_BASED

    def __init__(self) -> None:
        if not _HEBO_AVAILABLE:
            raise ImportError("HEBO is required. Install with: uv add 'dfbench[bo]'")

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        batch_size: int = 1,
        **hebo_kwargs,
    ) -> None:
        """Run HEBO.

        Termination is driven entirely by the ``Objective``'s budget
        (``max_evals`` / ``max_time``).

        Args:
            objective: Objective wrapper (mutated in place).
            init_params: Optional starting point (bounded). Currently unused.
            random_seed: Seed for reproducibility.
            batch_size: Candidates per suggestion.
            **hebo_kwargs: Forwarded to HEBO optimizer.
        """
        obj = objective
        dim = obj.n_params
        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb = np.asarray(obj.bounds[0])
        ub = np.asarray(obj.bounds[1])

        # Build HEBO design space
        params_spec = [
            {
                "name": f"x{i}",
                "type": "num",
                "lb": float(lb[i]),
                "ub": float(ub[i]),
            }
            for i in range(dim)
        ]
        space = DesignSpace().parse(params_spec)

        hebo_opt = HEBOOptimizer(
            space,
            model_name=hebo_kwargs.get("model_name", "gpy"),
            rand_sample=batch_size,
            scramble_seed=random_seed,
            **{k: v for k, v in hebo_kwargs.items() if k != "model_name"},
        )

        # JIT warmup
        obj.warmup_vmap_value(batch_size=1)
        obj.start_logging()

        while not obj.budget_exceeded:
            suggestion: pd.DataFrame = hebo_opt.suggest(n_suggestions=batch_size)

            # Evaluate each candidate via the Objective
            losses = []
            for row_idx in range(len(suggestion)):
                x_np = np.array([suggestion.iloc[row_idx][f"x{i}"] for i in range(dim)])
                x_jax = jnp.asarray(x_np)
                loss = float(obj.value(x_jax))
                losses.append(loss)

            suggestion["loss"] = losses
            hebo_opt.observe(suggestion, suggestion[["loss"]])
