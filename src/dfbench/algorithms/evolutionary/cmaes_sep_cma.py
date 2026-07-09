"""sep-CMA-ES backed by the ``cmaes`` package.

The ``cmaes`` library (https://github.com/CyberAgentAILab/cmaes) provides a
clean, dependency-free Python implementation of several CMA variants.  This
module wraps its ``SepCMA`` class, which uses a diagonal covariance matrix to
scale cheaply to high-dimensional problems.

sep-CMA-ES differs from vanilla CMA-ES in that the covariance matrix is
restricted to a diagonal, reducing the per-generation complexity from
O(n³) to O(n²).  It is recommended for problems with more than ~100
parameters or when the variables are structurally separable.

Bounded behaviour
-----------------
The ``cmaes`` package accepts bounds directly and applies an internal repair
strategy.  Candidates are additionally hard-clipped before being passed to
the Objective for extra safety.

Requires
--------
    cmaes >= 0.10  (``uv add 'dfbench[evolution]'``)
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

try:
    from cmaes import SepCMA
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'cmaes' package is required for CMAESSepCMA. "
        "Install it with:  uv add 'dfbench[evolution]'"
    ) from exc

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


class CMAESSepCMA(OptimizationAlgorithm):
    """sep-CMA-ES using the ``cmaes`` package (diagonal covariance).

    sep-CMA-ES maintains only the diagonal of the covariance matrix.  This
    makes each CMA update O(n²) instead of O(n³), so it is well-suited to
    moderate-to-high-dimensional problems where the full CMA update would be
    expensive.  Convergence on non-separable functions may be slower than
    vanilla CMA-ES.

    Bounded behaviour
    -----------------
    Operates in bounded parameter space.  The ``cmaes`` library handles
    boundary repair internally; candidates are additionally clipped to bounds
    before evaluation for strict feasibility.

    Attributes:
        algorithm_str: ``"cmaes_sep_cma"``
        algorithm_type: ``EVOLUTIONARY``

    Example::

        obj = Objective(problem, max_evals=20_000)
        CMAESSepCMA(batch_size=50).optimize(obj, pop_size=50, random_seed=42)
    """

    algorithm_str: str = "cmaes_sep_cma"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, batch_size: int = 1) -> None:
        """Initialise sep-CMA-ES.

        Args:
            batch_size: Number of candidates to evaluate per
                ``vmap_value`` call.  Controls peak device memory.
                Defaults to 1.
        """
        self._batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        init_params: np.ndarray | None = None,
        random_seed: int | None = None,
        sigma0: float | None = None,
        pop_size: int | None = None,
        max_iterations: int | None = None,
        max_no_improvement: int | None = None,
    ) -> None:
        """Run sep-CMA-ES.

        Args:
            objective: Pre-configured Objective instance.
            init_params: Initial mean vector.  Sampled uniformly in bounds
                when ``None``.
            random_seed: Seed for reproducibility.
            sigma0: Initial step size, as a dimensionless fraction of the
                unit cube (the search is performed in ``[0, 1]^n`` and
                mapped to physical bounds at evaluation time).  Defaults to
                ``0.3``.
            pop_size: Population size λ.  If ``None``, the ``cmaes`` library
                uses its default (``4 + floor(3·ln n)``).
            max_iterations: Maximum number of CMA generations.  ``None``
                means unlimited (budget and ``should_stop()`` govern stopping).
            max_no_improvement: Stop after this many generations without
                improving the best loss seen by the Objective.  ``None``
                disables stagnation stopping.
        """
        obj = objective

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb_np = np.asarray(obj.bounds[0])
        ub_np = np.asarray(obj.bounds[1])
        width = ub_np - lb_np
        n = obj.n_params

        # The cmaes package has no per-coordinate sigma.  On problems where
        # bound widths span orders of magnitude (e.g. Voyager), running the
        # CMA search directly in physical coordinates means a single scalar
        # ``sigma`` cannot be reasonable for every axis simultaneously.  We
        # therefore run the search in the unit cube [0, 1]^n and map to
        # physical space only when evaluating the Objective.
        if init_params is None:
            x0_unit = np.random.uniform(0.0, 1.0, size=n)
        else:
            x0_phys = np.clip(np.asarray(init_params, dtype=float), lb_np, ub_np)
            x0_unit = (x0_phys - lb_np) / width

        # sigma is now a dimensionless fraction of the unit cube.
        sigma = sigma0 if sigma0 is not None else 0.3
        batch_size = self._batch_size

        # Bounds in unit space for cmaes' internal repair.
        bounds_array = np.stack([np.zeros(n), np.ones(n)], axis=1)  # (n, 2)

        optimizer = SepCMA(
            mean=x0_unit,
            sigma=sigma,
            bounds=bounds_array,
            population_size=pop_size,
            seed=random_seed,
        )
        actual_pop = optimizer.population_size

        # JIT warmup before timing starts
        obj.warmup_vmap_value(batch_size=min(batch_size, actual_pop))
        obj.start_logging()

        iteration = 0
        gens_without_improvement = 0
        prev_best = float("inf")

        while not obj.budget_exceeded:
            if max_iterations is not None and iteration >= max_iterations:
                break
            if optimizer.should_stop():
                break

            # Collect the full population in unit space, then map to physical
            # space for evaluation.  Covariance updates stay in unit space.
            candidates_unit = [
                np.clip(optimizer.ask(), 0.0, 1.0) for _ in range(actual_pop)
            ]
            candidates_phys = [lb_np + c * width for c in candidates_unit]

            batch = jnp.asarray(np.stack(candidates_phys))
            # Evaluate in chunks of batch_size
            all_losses = []
            for i in range(0, actual_pop, batch_size):
                chunk = batch[i : i + batch_size]
                all_losses.append(obj.vmap_value(chunk))
            losses_jnp = jnp.concatenate(all_losses)
            losses = np.asarray(losses_jnp).tolist()

            solutions = list(zip(candidates_unit, losses))
            optimizer.tell(solutions)

            iteration += 1

            # Optional stagnation stopping
            if max_no_improvement is not None:
                current_best = (
                    float(obj.best_loss) if obj.best_loss is not None else float("inf")
                )
                if current_best < prev_best:
                    prev_best = current_best
                    gens_without_improvement = 0
                else:
                    gens_without_improvement += 1
                if gens_without_improvement >= max_no_improvement:
                    break
