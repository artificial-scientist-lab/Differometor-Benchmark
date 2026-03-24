"""evosax-backed MA-ES and LM-MA-ES algorithms.

evosax (https://github.com/RobertTLange/evosax) is a JAX-native evolution
strategy library providing several CMA-family variants.  This module exposes:

* :class:`EvosaxMAES`   -- Matrix Adaptation Evolution Strategy (MA-ES).
* :class:`EvosaxLMMAES` -- Limited-Memory MA-ES (LM-MA-ES), which scales
  to very high dimensions by maintaining only a low-rank approximation of
  the adaptation matrix.

Both classes operate in bounded parameter space by default: evosax candidates
(which live in an unconstrained Gaussian regime) are clipped to ``[lb, ub]``
before being passed to the Objective.

API compatibility
-----------------
This module targets **evosax >= 0.2** (released late 2024), whose interface
differs substantially from v0.1:

* Constructor: ``MA_ES(population_size=pop, solution=jnp.zeros(n))``
* Init:  ``state = strategy.init(key, mean, params)``
* Ask:   ``x, state = strategy.ask(key, state, params)``
* Tell:  ``state, metrics = strategy.tell(key, x, fitness, state, params)``
* Step-size parameter field: ``std_init`` (not ``sigma_init``).

Note on backend distinction
---------------------------
The EvoX library (``evox_es.py``) also exposes a CMA-ES variant under
``EvoxES(variant="CMAES")``.  The classes here are *complementary*, not
replacements: they use a different backend (evosax vs. EvoX), expose
different MA-ES/LM-MA-ES algorithm families, and tag their results with
distinct ``algorithm_str`` values so runs can be compared in benchmarks.

Requires
--------
    evosax >= 0.2  (``uv add evosax``)
"""

from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

try:
    from evosax.algorithms import MA_ES, LM_MA_ES
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "evosax is required for Evosax* algorithms. "
        "Install it with:  uv add evosax"
    ) from exc

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _run_evosax_loop(
    strategy,
    es_params,
    state,
    obj: Objective,
    lb_jnp: jnp.ndarray,
    ub_jnp: jnp.ndarray,
    rng: jax.Array,
    max_iterations: int | None,
    batch_size: int = 1,
) -> None:
    """Inner ask-clip-evaluate-tell loop for evosax v2 strategies.

    Runs until ``obj.budget_exceeded`` is True or an optional generation cap
    is reached.  Evaluations are split into ``batch_size``-sized chunks.

    Args:
        strategy: Initialised evosax strategy instance.
        es_params: Strategy hyper-parameters (from ``strategy.default_params``).
        state: Current strategy state.
        obj: Objective wrapper managing budget and logging.
        lb_jnp: Lower bounds as JAX array.
        ub_jnp: Upper bounds as JAX array.
        rng: JAX PRNG key (consumed incrementally).
        max_iterations: Generation cap.  ``None`` = unlimited.
        batch_size: Number of candidates to evaluate per ``vmap_value`` call.
    """
    iteration = 0
    while not obj.budget_exceeded:
        if max_iterations is not None and iteration >= max_iterations:
            break

        rng, ask_rng, tell_rng = jax.random.split(rng, 3)
        x, state = strategy.ask(ask_rng, state, es_params)  # (pop, n)

        # Clip to bounded space before evaluation
        x_clipped = jnp.clip(x, lb_jnp, ub_jnp)

        # Evaluate in chunks of batch_size
        n_pop = x_clipped.shape[0]
        chunks = [obj.vmap_value(x_clipped[i : i + batch_size])
                  for i in range(0, n_pop, batch_size)]
        fitness = jnp.concatenate(chunks)

        state, _ = strategy.tell(tell_rng, x_clipped, fitness, state, es_params)
        iteration += 1


# ---------------------------------------------------------------------------
# EvosaxMAES
# ---------------------------------------------------------------------------


class EvosaxMAES(OptimizationAlgorithm):
    """Matrix Adaptation Evolution Strategy (MA-ES) via evosax.

    MA-ES is a CMA-ES variant that replaces the full covariance matrix update
    with a simpler cumulative matrix adaptation step.  It achieves similar
    convergence to CMA-ES on many benchmarks while being conceptually simpler
    to implement.  MA-ES is distinct from the EvoX-backed CMA-ES
    (``EvoxES(variant="CMAES")``); results will differ because the backends,
    adaptation rules, and default hyper-parameters differ.

    Bounded behaviour
    -----------------
    Operates in bounded parameter space by default.  evosax generates
    candidates in an unconstrained Gaussian regime; they are clipped to
    ``[lb, ub]`` before evaluation.

    Attributes:
        algorithm_str: ``"evosax_maes"``
        algorithm_type: ``EVOLUTIONARY``

    Example::

        obj = Objective(problem, max_evals=10_000)
        EvosaxMAES(batch_size=40).optimize(obj, pop_size=40, random_seed=0)
    """

    algorithm_str: str = "evosax_maes"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, batch_size: int = 1) -> None:
        """Initialise MA-ES.

        Args:
            batch_size: Number of candidates to evaluate per
                ``vmap_value`` call.  Controls peak device memory.
                Defaults to 1.
        """
        self._batch_size = batch_size

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        sigma0: float | None = None,
        pop_size: int = 20,
        max_iterations: int | None = None,
    ) -> None:
        """Run MA-ES.

        Args:
            problem_objective: Pre-configured Objective instance.
            init_params: Initial mean.  Sampled uniformly in bounds when
                ``None``.
            random_seed: Seed for reproducibility.
            sigma0: Initial step size.  Defaults to 0.3 * mean(ub - lb).
            pop_size: Population size lambda.  Defaults to 20.
            max_iterations: Maximum CMA generations.  ``None`` = unlimited.
        """
        obj = problem_objective
        problem = obj.problem

        random_seed, rng = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb_np = np.asarray(problem.bounds[0])
        ub_np = np.asarray(problem.bounds[1])
        lb_jnp = jnp.asarray(lb_np)
        ub_jnp = jnp.asarray(ub_np)
        n = problem.n_params

        sigma = float(sigma0 if sigma0 is not None else np.mean(0.3 * (ub_np - lb_np)))

        solution_template = jnp.zeros(n)
        strategy = MA_ES(population_size=pop_size, solution=solution_template)
        es_params = strategy.default_params.replace(std_init=sigma)

        if init_params is not None:
            mean0 = jnp.clip(jnp.asarray(init_params), lb_jnp, ub_jnp)
        else:
            rng, mean_rng = jax.random.split(rng)
            mean0 = jax.random.uniform(mean_rng, shape=(n,), minval=lb_jnp, maxval=ub_jnp)

        rng, init_rng = jax.random.split(rng)
        state = strategy.init(init_rng, mean0, es_params)

        # JIT warmup before timing starts
        batch_size = self._batch_size
        _ = obj.vmap_value(jnp.zeros((min(batch_size, pop_size), n)))
        obj.start_logging()

        rng, loop_rng = jax.random.split(rng)
        _run_evosax_loop(
            strategy, es_params, state, obj, lb_jnp, ub_jnp, loop_rng,
            max_iterations, batch_size,
        )


# ---------------------------------------------------------------------------
# EvosaxLMMAES
# ---------------------------------------------------------------------------


class EvosaxLMMAES(OptimizationAlgorithm):
    """Limited-Memory MA-ES (LM-MA-ES) via evosax.

    LM-MA-ES stores only a low-rank (``memory_size`` vectors) approximation
    of the adaptation matrix.  This reduces memory from O(n^2) to O(n*m)
    where m is the memory size (default: ``4 + floor(3*ln(n))``), making it
    practical for high-dimensional problems (n > 1000).

    LM-MA-ES is distinct from the EvoX-backed CMA-ES (``evox_es.py``) both
    algorithmically and in its backend.

    Bounded behaviour
    -----------------
    Identical to :class:`EvosaxMAES` (bounded, clipped candidates).

    Attributes:
        algorithm_str: ``"evosax_lm_maes"``
        algorithm_type: ``EVOLUTIONARY``
    """

    algorithm_str: str = "evosax_lm_maes"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, batch_size: int = 1) -> None:
        """Initialise LM-MA-ES.

        Args:
            batch_size: Number of candidates to evaluate per
                ``vmap_value`` call.  Controls peak device memory.
                Defaults to 1.
        """
        self._batch_size = batch_size

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        sigma0: float | None = None,
        pop_size: int = 20,
        memory_size: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """Run LM-MA-ES.

        Args:
            problem_objective: Pre-configured Objective instance.
            init_params: Initial mean.  Sampled uniformly in bounds when
                ``None``.
            random_seed: Seed for reproducibility.
            sigma0: Initial step size.  Defaults to 0.3 * mean(ub - lb).
            pop_size: Population size lambda.  Defaults to 20.
            memory_size: Number of stored direction vectors (low-rank order).
                If ``None``, evosax uses its default
                (approximately ``4 + floor(3*ln(n))``).
            max_iterations: Maximum generations.  ``None`` = unlimited.
        """
        obj = problem_objective
        problem = obj.problem

        random_seed, rng = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb_np = np.asarray(problem.bounds[0])
        ub_np = np.asarray(problem.bounds[1])
        lb_jnp = jnp.asarray(lb_np)
        ub_jnp = jnp.asarray(ub_np)
        n = problem.n_params

        sigma = float(sigma0 if sigma0 is not None else np.mean(0.3 * (ub_np - lb_np)))

        solution_template = jnp.zeros(n)
        lm_kwargs: dict = dict(population_size=pop_size, solution=solution_template)
        if memory_size is not None:
            lm_kwargs["memory_size"] = memory_size
        strategy = LM_MA_ES(**lm_kwargs)
        es_params = strategy.default_params.replace(std_init=sigma)

        if init_params is not None:
            mean0 = jnp.clip(jnp.asarray(init_params), lb_jnp, ub_jnp)
        else:
            rng, mean_rng = jax.random.split(rng)
            mean0 = jax.random.uniform(mean_rng, shape=(n,), minval=lb_jnp, maxval=ub_jnp)

        rng, init_rng = jax.random.split(rng)
        state = strategy.init(init_rng, mean0, es_params)

        batch_size = self._batch_size
        _ = obj.vmap_value(jnp.zeros((min(batch_size, pop_size), n)))
        obj.start_logging()

        rng, loop_rng = jax.random.split(rng)
        _run_evosax_loop(
            strategy, es_params, state, obj, lb_jnp, ub_jnp, loop_rng,
            max_iterations, batch_size,
        )
