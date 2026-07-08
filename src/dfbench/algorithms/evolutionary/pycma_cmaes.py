"""pycma-backed CMA-family algorithms.

Provides vanilla CMA-ES, IPOP-CMA-ES, BIPOP-CMA-ES, and active-CMA-ES,
all backed by the canonical `pycma` library.

Bounded behaviour
-----------------
All variants operate exclusively in bounded parameter space by default.  To
handle problems whose bound widths span several orders of magnitude (e.g.
Voyager), the CMA search runs in the unit cube ``[0, 1]^n`` and candidates
are mapped to physical bounds (``lb + u * (ub - lb)``) only at evaluation
time, so a single scalar ``sigma`` is a sensible step for every coordinate.

Unbounded mode is not supported by this backend; the algorithms expect a
finite domain and will raise ``ValueError`` during ``optimize()`` if the
Objective has been configured in unbounded mode prior to this call.

Requires
--------
    pycma >= 3.3  (``uv add cma``)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import jax.numpy as jnp

try:
    import cma
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pycma is required for PyCMA* algorithms. Install it with:  uv add cma"
    ) from exc

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _default_pop_size(n_params: int) -> int:
    """Return pycma's default population size for *n_params* dimensions."""
    return int(4 + 3 * np.floor(np.log(max(n_params, 1))))


def _build_opts(
    n: int,
    pop_size: int,
    active: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a pycma CMAOptions dict for unit-cube search.

    The CMA search runs in ``[0, 1]^n`` and candidates are mapped to physical
    bounds only at evaluation time (see :func:`_ask_eval_tell`).  This is
    essential on problems whose bound widths span multiple orders of
    magnitude (e.g. Voyager): a single scalar ``sigma`` cannot be a sensible
    step for every coordinate in physical space, and using ``CMA_stds`` to
    rescale interacts poorly with pycma's boundary handler (initial samples
    are pushed far outside the box and then clipped to the boundary,
    destroying the rank-μ update).  Search in the unit cube avoids both
    issues.

    Args:
        n: Problem dimensionality.
        pop_size: Population size (lambda).
        active: If True, enable active-CMA negative weight updates.
        extra: Additional key-value pairs that override the defaults.

    Returns:
        dict: Ready-to-pass CMAOptions dictionary.
    """
    opts: dict[str, Any] = {
        "bounds": [[0.0] * n, [1.0] * n],
        "popsize": pop_size,
        "CMA_active": active,
        "verbose": -9,  # silent
        "maxfevals": np.inf,  # budget is managed by the Objective
        "tolx": 1e-12,
        "tolfun": 1e-12,
    }
    if extra:
        opts.update(extra)
    return opts


def _eval_batched(
    candidates: jnp.ndarray,
    obj: Objective,
    batch_size: int,
) -> jnp.ndarray:
    """Evaluate *candidates* through *obj* in ``batch_size``-sized chunks.

    Args:
        candidates: (pop, n) array of clipped candidate solutions.
        obj: Objective wrapper; logs all evaluations and manages the budget.
        batch_size: Maximum number of candidates evaluated per ``vmap_value``
            call.  Controls peak device memory.

    Returns:
        (pop,) loss array.
    """
    n_pop = candidates.shape[0]
    if batch_size >= n_pop:
        return obj.vmap_value(candidates)
    all_losses = []
    for i in range(0, n_pop, batch_size):
        batch = candidates[i : i + batch_size]
        all_losses.append(obj.vmap_value(batch))
    return jnp.concatenate(all_losses, axis=0)


def _ask_eval_tell(
    es: "cma.CMAEvolutionStrategy",
    obj: Objective,
    lb_np: np.ndarray,
    width: np.ndarray,
    batch_size: int,
) -> None:
    """Run one ask-evaluate-tell generation in unit-cube coordinates.

    The CMA strategy operates in ``[0, 1]^n``.  Candidates are clipped to
    the unit cube (in case pycma's repair leaves a tiny overshoot), mapped
    to physical space ``lb + u * width`` for evaluation, and then the
    unit-cube candidates are passed back to ``es.tell`` so the covariance
    update stays in unit space.

    Args:
        es: Active ``CMAEvolutionStrategy`` instance managed by the caller.
        obj: Objective wrapper; logs all evaluations and manages the budget.
        lb_np: Lower bounds in physical space.
        width: ``ub - lb`` in physical space.
        batch_size: Max candidates per ``vmap_value`` call.
    """
    solutions_unit = [np.clip(np.asarray(x), 0.0, 1.0) for x in es.ask()]
    physical = [lb_np + u * width for u in solutions_unit]
    batch = jnp.asarray(np.stack(physical))
    losses_jnp = _eval_batched(batch, obj, batch_size)
    es.tell(solutions_unit, np.asarray(losses_jnp).tolist())


# ---------------------------------------------------------------------------
# PyCMACMAES
# ---------------------------------------------------------------------------


class PyCMACMAES(OptimizationAlgorithm):
    """Vanilla CMA-ES backed by pycma.

    Runs a single CMA-ES instance (no restarts).  Suitable as a clean
    single-run baseline.  For restarted variants use :class:`PyCMAIPOP` or
    :class:`PyCMABIPOP`.

    Bounded behaviour
    -----------------
    Operates exclusively in bounded parameter space.  The CMA search runs in
    the unit cube ``[0, 1]^n`` and candidates are mapped to physical bounds
    only at evaluation time.

    Attributes:
        algorithm_str: ``"pycma_cmaes"``
        algorithm_type: ``EVOLUTIONARY``

    Example::

        obj = Objective(problem, max_evals=10_000)
        PyCMACMAES(batch_size=20).optimize(obj, pop_size=20, random_seed=0)
    """

    algorithm_str: str = "pycma_cmaes"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, batch_size: int = 1) -> None:
        """Initialise PyCMA CMA-ES.

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
    ) -> None:
        """Run CMA-ES.

        Args:
            objective: Pre-configured Objective instance.
            init_params: Initial mean vector.  Sampled uniformly in bounds
                when ``None``.
            random_seed: Seed for reproducibility.
            sigma0: Initial step size.  Defaults to ``0.3`` (dimensionless fraction of the unit cube; search runs in ``[0, 1]^n`` and is mapped to physical bounds at evaluation).
            pop_size: Population size λ.  If ``None``, pycma's default
                (``4 + floor(3·ln n)``) is used.
            max_iterations: Maximum number of CMA generations.  ``None``
                means unlimited (budget and convergence govern stopping).
        """
        obj = objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb_np = np.asarray(problem.bounds[0])
        ub_np = np.asarray(problem.bounds[1])
        width = ub_np - lb_np
        n = problem.n_params

        # Build the initial mean in unit-cube coordinates.
        if init_params is None:
            x0_unit = np.random.uniform(0.0, 1.0, size=n)
        else:
            x0_phys = np.clip(np.asarray(init_params, dtype=float), lb_np, ub_np)
            x0_unit = (x0_phys - lb_np) / width

        # sigma is a dimensionless fraction of the unit cube.
        sigma = sigma0 if sigma0 is not None else 0.3
        pop_size = pop_size or _default_pop_size(n)
        opts = _build_opts(n, pop_size)

        # JIT warmup before timing starts
        warmup_bs = min(self._batch_size, pop_size)
        obj.warmup_vmap_value(batch_size=warmup_bs)

        obj.start_logging()

        es = cma.CMAEvolutionStrategy(x0_unit.tolist(), sigma, opts)
        iteration = 0
        while not es.stop() and not obj.budget_exceeded:
            if max_iterations is not None and iteration >= max_iterations:
                break
            _ask_eval_tell(es, obj, lb_np, width, self._batch_size)
            iteration += 1


# ---------------------------------------------------------------------------
# PyCMAActiveCMAES
# ---------------------------------------------------------------------------


class PyCMAActiveCMAES(OptimizationAlgorithm):
    """Active CMA-ES (aCMA-ES) backed by pycma.

    Extends vanilla CMA-ES with *negative* covariance matrix updates for
    unsuccessful candidate directions, which actively steers the distribution
    away from poor regions and often accelerates convergence.

    The only code difference from :class:`PyCMACMAES` is the ``'CMA_active'``
    option.  The backend, evaluation loop, and bounded-space handling are
    identical.

    Bounded behaviour
    -----------------
    Identical to :class:`PyCMACMAES` (bounded by default; CMA runs in the
    unit cube).

    Attributes:
        algorithm_str: ``"pycma_acmaes"``
        algorithm_type: ``EVOLUTIONARY``
    """

    algorithm_str: str = "pycma_acmaes"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, batch_size: int = 1) -> None:
        """Initialise active CMA-ES.

        Args:
            batch_size: Number of candidates to evaluate per
                ``vmap_value`` call.  Defaults to 1.
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
    ) -> None:
        """Run active CMA-ES.

        Args:
            objective: Pre-configured Objective instance.
            init_params: Initial mean vector.  Sampled uniformly in bounds
                when ``None``.
            random_seed: Seed for reproducibility.
            sigma0: Initial step size.  Defaults to ``0.3`` (dimensionless fraction of the unit cube; search runs in ``[0, 1]^n`` and is mapped to physical bounds at evaluation).
            pop_size: Population size λ.  Defaults to pycma's
                ``4 + floor(3·ln n)``.
            max_iterations: Maximum CMA generations.  ``None`` = unlimited.
        """
        obj = objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb_np = np.asarray(problem.bounds[0])
        ub_np = np.asarray(problem.bounds[1])
        width = ub_np - lb_np
        n = problem.n_params

        # Build the initial mean in unit-cube coordinates.
        if init_params is None:
            x0_unit = np.random.uniform(0.0, 1.0, size=n)
        else:
            x0_phys = np.clip(np.asarray(init_params, dtype=float), lb_np, ub_np)
            x0_unit = (x0_phys - lb_np) / width

        # sigma is a dimensionless fraction of the unit cube.
        sigma = sigma0 if sigma0 is not None else 0.3
        pop_size = pop_size or _default_pop_size(n)
        opts = _build_opts(n, pop_size, active=True)

        warmup_bs = min(self._batch_size, pop_size)
        obj.warmup_vmap_value(batch_size=warmup_bs)
        obj.start_logging()

        es = cma.CMAEvolutionStrategy(x0_unit.tolist(), sigma, opts)
        iteration = 0
        while not es.stop() and not obj.budget_exceeded:
            if max_iterations is not None and iteration >= max_iterations:
                break
            _ask_eval_tell(es, obj, lb_np, width, self._batch_size)
            iteration += 1


# ---------------------------------------------------------------------------
# PyCMAIPOP
# ---------------------------------------------------------------------------


class PyCMAIPOP(OptimizationAlgorithm):
    """IPOP-CMA-ES: CMA-ES with increasing population restarts.

    After CMA-ES converges (or stalls), the algorithm restarts from a fresh
    random point with a population size doubled from the previous run.  This
    is repeated up to ``max_restarts`` times or until the budget is exhausted.

    Each restart uses its own fresh CMA instance; the overall best solution is
    tracked by the Objective.

    Reference: Auger & Hansen, "A restart CMA evolution strategy with
    increasing population size" (CEC 2005).

    Bounded behaviour
    -----------------
    Identical to :class:`PyCMACMAES`.

    Attributes:
        algorithm_str: ``"pycma_ipop"``
        algorithm_type: ``EVOLUTIONARY``
    """

    algorithm_str: str = "pycma_ipop"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, batch_size: int = 1) -> None:
        """Initialise IPOP-CMA-ES.

        Args:
            batch_size: Number of candidates to evaluate per
                ``vmap_value`` call.  Defaults to 1.
        """
        self._batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        init_params: np.ndarray | None = None,
        random_seed: int | None = None,
        sigma0: float | None = None,
        pop_size: int | None = None,
        max_restarts: int = 9,
        max_iterations_per_restart: int | None = None,
    ) -> None:
        """Run IPOP-CMA-ES.

        Args:
            objective: Pre-configured Objective instance.
            init_params: Initial mean for the *first* run.  Subsequent
                restarts always use fresh random points.
            random_seed: Seed for reproducibility.
            sigma0: Initial step size for each restart (dimensionless
                fraction of the unit cube; search runs in ``[0, 1]^n`` and
                is mapped to physical bounds at evaluation).  Defaults to
                ``0.3``.
            pop_size: Base population size at restart 0.  Doubles each
                restart.  Defaults to pycma's ``4 + floor(3·ln n)``.
            max_restarts: Maximum number of CMA restarts (excluding the
                first run).  Defaults to 9.
            max_iterations_per_restart: Cap on CMA generations per restart
                (independent of the overall budget).  ``None`` = unlimited.
        """
        obj = objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb_np = np.asarray(problem.bounds[0])
        ub_np = np.asarray(problem.bounds[1])
        width = ub_np - lb_np
        n = problem.n_params

        base_pop = pop_size or _default_pop_size(n)
        # sigma is a dimensionless fraction of the unit cube.
        sigma_base = sigma0 if sigma0 is not None else 0.3

        # JIT warmup
        warmup_bs = min(self._batch_size, base_pop)
        obj.warmup_vmap_value(batch_size=warmup_bs)
        obj.start_logging()

        cur_pop = base_pop
        for restart in range(max_restarts + 1):
            if obj.budget_exceeded:
                break

            if restart == 0 and init_params is not None:
                x0_phys = np.clip(np.asarray(init_params, dtype=float), lb_np, ub_np)
                x0_unit = (x0_phys - lb_np) / width
            else:
                x0_unit = np.random.uniform(0.0, 1.0, size=n)

            opts = _build_opts(n, cur_pop)
            es = cma.CMAEvolutionStrategy(x0_unit.tolist(), sigma_base, opts)

            iteration = 0
            while not es.stop() and not obj.budget_exceeded:
                if (
                    max_iterations_per_restart is not None
                    and iteration >= max_iterations_per_restart
                ):
                    break
                _ask_eval_tell(es, obj, lb_np, width, self._batch_size)
                iteration += 1

            # Double population for next restart (IPOP rule)
            cur_pop = cur_pop * 2


# ---------------------------------------------------------------------------
# PyCMABIPOP
# ---------------------------------------------------------------------------


class PyCMABIPOP(OptimizationAlgorithm):
    """BIPOP-CMA-ES: bi-population restart strategy.

    After the first run, BIPOP alternates between two regimes:

    * **Large population**: population size is doubled each time this regime
      is entered, and the initial step size is reset to ``sigma0``.
    * **Small population**: population size is drawn uniformly at random
      from the range ``[base_pop, large_pop // 2]``; initial step size is
      drawn from ``sigma0 × 10^U(−2, 0)``.

    The regime alternation rule follows Hansen (2009): choose whichever regime
    has accumulated fewer total function evaluations so far.

    Reference: Hansen, N., "Benchmarking a BI-population CMA-ES on the
    BBOB-2009 function testbed" (GECCO Workshop 2009).

    Bounded behaviour
    -----------------
    Identical to :class:`PyCMACMAES`.

    Attributes:
        algorithm_str: ``"pycma_bipop"``
        algorithm_type: ``EVOLUTIONARY``
    """

    algorithm_str: str = "pycma_bipop"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, batch_size: int = 1) -> None:
        """Initialise BIPOP-CMA-ES.

        Args:
            batch_size: Number of candidates to evaluate per
                ``vmap_value`` call.  Defaults to 1.
        """
        self._batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        init_params: np.ndarray | None = None,
        random_seed: int | None = None,
        sigma0: float | None = None,
        pop_size: int | None = None,
        max_restarts: int = 18,
        max_iterations_per_restart: int | None = None,
    ) -> None:
        """Run BIPOP-CMA-ES.

        Args:
            objective: Pre-configured Objective instance.
            init_params: Initial mean for the first run only.
            random_seed: Seed for reproducibility.
            sigma0: Base step size.  Defaults to ``0.3`` (dimensionless fraction of the unit cube; search runs in ``[0, 1]^n`` and is mapped to physical bounds at evaluation).
            pop_size: Base population size (first run and small regime
                lower bound).  Defaults to pycma's
                ``4 + floor(3·ln n)``.
            max_restarts: Maximum total restarts (large + small).
                Defaults to 18 (matching the BBOB benchmark convention).
            max_iterations_per_restart: Per-restart generation cap.
                ``None`` = unlimited.
        """
        obj = objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb_np = np.asarray(problem.bounds[0])
        ub_np = np.asarray(problem.bounds[1])
        width = ub_np - lb_np
        n = problem.n_params

        base_pop = pop_size or _default_pop_size(n)
        # sigma is a dimensionless fraction of the unit cube.
        sigma_base = sigma0 if sigma0 is not None else 0.3

        warmup_bs = min(self._batch_size, base_pop)
        obj.warmup_vmap_value(batch_size=warmup_bs)
        obj.start_logging()

        large_pop = base_pop  # grows each large-regime restart
        large_restarts = 0
        evals_large = 0  # evaluations consumed by large-regime runs
        evals_small = 0  # evaluations consumed by small-regime runs

        for restart in range(max_restarts + 1):
            if obj.budget_exceeded:
                break

            evals_before = obj.eval_count

            if restart == 0:
                # First run: default settings
                pop_size = base_pop
                sigma = sigma_base
                if init_params is not None:
                    x0_phys = np.clip(
                        np.asarray(init_params, dtype=float), lb_np, ub_np
                    )
                    x0_unit = (x0_phys - lb_np) / width
                else:
                    x0_unit = np.random.uniform(0.0, 1.0, size=n)
            elif evals_small <= evals_large:
                # Small-population regime (Hansen 2009)
                # λ_S = floor(λ_def * (λ_L / λ_def)^{u^2}), biased towards small values
                u = np.random.uniform()
                ratio = large_pop / max(base_pop, 1)
                pop_size = max(2, int(base_pop * ratio ** (u * u)))
                pop_size = min(pop_size, max(2, large_pop // 2))
                sigma = sigma_base * 10 ** (-2.0 * np.random.random())
                x0_unit = np.random.uniform(0.0, 1.0, size=n)
            else:
                # Large-population regime
                large_restarts += 1
                large_pop = base_pop * (2**large_restarts)
                pop_size = large_pop
                sigma = sigma_base
                x0_unit = np.random.uniform(0.0, 1.0, size=n)

            opts = _build_opts(n, pop_size)
            es = cma.CMAEvolutionStrategy(x0_unit.tolist(), sigma, opts)

            iteration = 0
            while not es.stop() and not obj.budget_exceeded:
                if (
                    max_iterations_per_restart is not None
                    and iteration >= max_iterations_per_restart
                ):
                    break
                _ask_eval_tell(es, obj, lb_np, width, self._batch_size)
                iteration += 1

            # Track evaluations per regime for alternation heuristic
            evals_this_run = obj.eval_count - evals_before
            if restart > 0 and evals_small <= evals_large:
                evals_small += evals_this_run
            else:
                evals_large += evals_this_run
