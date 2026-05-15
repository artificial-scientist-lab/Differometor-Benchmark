"""Native JAX implementations of classical evolution strategies.

Provides:

* :class:`JAXOnePlusOneES`  – (1+1)-ES with the one-fifth success rule.
* :class:`JAXMuLambdaES`   – (μ,λ)-ES with truncation selection and
  cumulative step-size accumulation.

Both algorithms are implemented from scratch in pure JAX/NumPy and have **no
third-party ES library dependency**.  They are benchmark-oriented, readable
implementations of the foundational ES algorithms that predate CMA-ES.

Algorithm notes
---------------
(1+1)-ES
    A single parent generates a single offspring each step.  If the offspring
    is *not worse* than the parent, the offspring replaces the parent.  The
    step size sigma is adapted via the one-fifth success rule over a sliding window
    of ``success_window`` evaluations.

(μ,λ)-ES
    Each generation, λ offspring are sampled from the current mean with
    isotropic Gaussian noise scaled by sigma.  The best μ offspring (truncation
    selection, **comma** semantics: parents are *not* retained) are averaged
    to form the new mean.  Step size is adapted via a cumulative success-rate
    estimate.

Both algorithms use **no covariance adaptation** — they are classical ES, not
CMA.  For covariance-adapting variants see ``pycma_cmaes.py`` or
``evosax_es.py``.

Bounded behaviour
-----------------
Both algorithms operate in bounded parameter space by default.  Candidates
that fall outside ``[lb, ub]`` are clipped before evaluation.  The
distribution mean is also clipped after each update so the search does not
drift outside bounds permanently.  Unbounded mode is not supported.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from collections import deque
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


# ---------------------------------------------------------------------------
# JAXOnePlusOneES
# ---------------------------------------------------------------------------


class JAXOnePlusOneES(OptimizationAlgorithm):
    """(1+1)-Evolution Strategy with the one-fifth success rule.

    A classic single-parent, single-offspring ES.  At each step:

    1. Sample offspring ``y = clip(x + σ·z,  lb, ub)``  where ``z ~ N(0, I)``.
    2. Evaluate ``f(y)``; if ``f(y) ≤ f(x)`` accept: ``x ← y``.
    3. Every ``success_window`` steps, adapt σ::

           if p_success > 0.20:  σ ← σ · exp(1/n)
           if p_success < 0.20:  σ ← σ · exp(-1/(5n))

    The initial σ defaults to ``0.3 × mean(ub − lb)``.

    Bounded behaviour
    -----------------
    Operates exclusively in bounded parameter space.  Candidates are clipped
    to ``[lb, ub]`` before evaluation; the parent is never allowed to leave
    the feasible region.  Unbounded mode is not supported.

    Attributes:
        algorithm_str: ``"jax_1p1es"``
        algorithm_type: ``EVOLUTIONARY``

    Example::

        obj = Objective(problem, max_evals=5_000)
        JAXOnePlusOneES().optimize(obj, random_seed=1)
    """

    algorithm_str: str = "jax_1p1es"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, batch_size: int = 1) -> None:
        """Initialise (1+1)-ES.

        Args:
            batch_size: Number of candidates to evaluate per
                ``vmap_value`` call.  Not used by (1+1)-ES (which is
                single-point) but accepted for interface consistency.
                Defaults to 1.
        """
        self._batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        sigma0: float | None = None,
        sigma_min: float = 1e-10,
        success_window: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        """Run (1+1)-ES.

        Args:
            objective: Pre-configured Objective instance.
            init_params: Starting point.  Sampled uniformly in bounds when
                ``None``.
            random_seed: Seed for reproducibility.
            sigma0: Initial step size as a fraction of each parameter's
                own range ``(ub - lb)``.  Defaults to ``0.3`` (30% of
                each dimension's range).
            sigma_min: Minimum allowed step size; stops on σ < σ_min.
            success_window: Number of steps between σ adaptation checks.
                If ``None``, defaults to ``10 × n_params`` (at least 10).
                Larger windows give smoother adaptation at the cost of
                slower response to non-stationarities.
            max_iterations: Maximum number of offspring evaluations.  ``None``
                means unlimited (budget governs stopping).
        """
        obj = objective
        problem = obj.problem

        random_seed, rng = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb = jnp.asarray(problem.bounds[0])
        ub = jnp.asarray(problem.bounds[1])
        n = problem.n_params

        if init_params is None:
            rng, init_rng = jax.random.split(rng)
            x = jax.random.uniform(init_rng, shape=(n,), minval=lb, maxval=ub)
        else:
            x = jnp.clip(jnp.asarray(init_params), lb, ub)

        # sigma is a fraction of each parameter's own range so that
        # heterogeneous bounds (e.g. length ∈ [1,4000] vs reflectivity ∈ [0,1])
        # are handled correctly. A single global sigma would cause narrow-range
        # parameters to always be clipped to their boundary values.
        scale = ub - lb  # per-parameter range, shape (n,)
        sigma = float(sigma0 if sigma0 is not None else 0.3)
        window = success_window if success_window is not None else max(10, 10 * n)

        # JIT warmup — single-point evaluation
        obj.warmup_value()
        obj.start_logging()

        # Evaluate starting point
        fx = obj.value(x)

        successes: deque[int] = deque(maxlen=window)
        step = 0

        while not obj.budget_exceeded:
            if max_iterations is not None and step >= max_iterations:
                break
            if sigma < sigma_min:
                break

            rng, noise_rng = jax.random.split(rng)
            z = jax.random.normal(noise_rng, shape=(n,))
            y = jnp.clip(x + sigma * scale * z, lb, ub)

            fy = obj.value(y)
            success = int(float(fy) <= float(fx))
            successes.append(success)

            if success:
                x = y
                fx = fy

            # Adapt σ every `window` steps via the 1/5 success rule
            if len(successes) == window and (step + 1) % window == 0:
                p_succ = sum(successes) / window
                if p_succ > 0.2:
                    sigma = sigma * float(jnp.exp(jnp.array(1.0 / n)))
                elif p_succ < 0.2:
                    sigma = sigma * float(jnp.exp(jnp.array(-1.0 / (5.0 * n))))

            step += 1


# ---------------------------------------------------------------------------
# JAXMuLambdaES
# ---------------------------------------------------------------------------


class JAXMuLambdaES(OptimizationAlgorithm):
    """(μ,λ)-Evolution Strategy with truncation selection.

    Each generation:

    1.  λ offspring: ``y_i = clip(mean + σ·z_i,  lb, ub)``  for ``z_i ~ N(0, I)``.
    2.  Evaluate all offspring via ``obj.vmap_value``.
    3.  Select the best μ by fitness (truncation, **comma** semantics — parents
        are *not* retained).
    4.  New mean ``← weighted average of the μ best offspring``
        (uniform weights by default).
    5.  Step size σ adapted via a cumulative success-rate estimate based on
        generational improvement::

            improved ← 1 if mean_loss(selected_μ) < prev_mean_loss else 0
            p_succ   ← (1 − cₛ) · p_succ + cₛ · improved
            σ        ← σ · exp((p_succ − target_succ) / d_σ)

        where ``target_succ = μ / λ`` and ``cₛ = 1/n``, ``d_σ = 1``.

    This is a classical ES **without** covariance matrix adaptation.  For
    CMA variants see :mod:`pycma_cmaes` or :mod:`evosax_es`.

    Bounded behaviour
    -----------------
    Identical to :class:`JAXOnePlusOneES` (bounded, clipped candidates,
    mean clipped after recombination).

    Attributes:
        algorithm_str: ``"jax_mu_lambda_es"``
        algorithm_type: ``EVOLUTIONARY``

    Example::

        obj = Objective(problem, max_evals=20_000)
        JAXMuLambdaES(batch_size=50).optimize(obj, mu=10, lam=50, random_seed=0)
    """

    algorithm_str: str = "jax_mu_lambda_es"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(self, batch_size: int = 1) -> None:
        """Initialise (μ,λ)-ES.

        Args:
            batch_size: Number of candidates to evaluate per
                ``vmap_value`` call.  Controls peak device memory.
                Defaults to 1.
        """
        self._batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        sigma0: float | None = None,
        sigma_min: float = 1e-10,
        mu: int = 10,
        lam: int = 50,
        max_iterations: int | None = None,
    ) -> None:
        """Run (μ,λ)-ES.

        Args:
            objective: Pre-configured Objective instance.
            init_params: Initial mean.  Sampled uniformly in bounds when
                ``None``.
            random_seed: Seed for reproducibility.
            sigma0: Initial step size as a fraction of each parameter's
                own range ``(ub - lb)``.  Defaults to ``0.3`` (30% of
                each dimension's range).
            sigma_min: Minimum sigma; stops when sigma drops below this.
            mu: Number of survivors (parents for next generation).
                Defaults to 10.
            lam: Number of offspring per generation.  Must be > ``mu``.
                Defaults to 50.
            max_iterations: Maximum number of generations.  Each generation
                costs λ evaluations.  ``None`` = unlimited.

        Raises:
            ValueError: If ``mu >= lam``.
        """
        if mu >= lam:
            raise ValueError(
                f"(μ,λ)-ES requires mu < lam, got mu={mu}, lam={lam}."
            )

        obj = objective
        problem = obj.problem

        random_seed, rng = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb = jnp.asarray(problem.bounds[0])
        ub = jnp.asarray(problem.bounds[1])
        n = problem.n_params

        if init_params is None:
            rng, init_rng = jax.random.split(rng)
            mean = jax.random.uniform(init_rng, shape=(n,), minval=lb, maxval=ub)
        else:
            mean = jnp.clip(jnp.asarray(init_params), lb, ub)

        # sigma is a fraction of each parameter's own range so that
        # heterogeneous bounds (e.g. length ∈ [1,4000] vs reflectivity ∈ [0,1])
        # are handled correctly. A single global sigma would cause narrow-range
        # parameters to always be clipped to their boundary values.
        scale = ub - lb  # per-parameter range, shape (n,)
        sigma = float(sigma0 if sigma0 is not None else 0.3)

        # Step-size adaptation parameters
        target_succ = mu / lam       # expected success fraction
        c_s = 1.0 / max(n, 1)      # smoothing coefficient
        d_sigma = 1.0               # damping
        p_succ = target_succ        # initialise cumulative success rate
        prev_mean_loss = float("inf")  # track improvement for adaptation

        # JIT warmup with the same effective batch size used in-loop.
        batch_size = self._batch_size
        obj.warmup_vmap_value(batch_size=min(batch_size, lam))
        obj.start_logging()

        iteration = 0
        while not obj.budget_exceeded:
            if max_iterations is not None and iteration >= max_iterations:
                break
            if sigma < sigma_min:
                break

            # Sample λ offspring with per-parameter scaling so sigma is a
            # fraction of each dimension's own range.
            rng, noise_rng = jax.random.split(rng)
            noise = jax.random.normal(noise_rng, shape=(lam, n))
            offspring = jnp.clip(mean[None, :] + sigma * scale[None, :] * noise, lb, ub)

            # Evaluate via vmap in chunks of batch_size
            chunks = [obj.vmap_value(offspring[i : i + batch_size])
                      for i in range(0, lam, batch_size)]
            losses = jnp.concatenate(chunks)  # (lam,)

            if obj.budget_exceeded:
                break

            # Replace any non-finite loss with a large finite penalty so
            # truncation selection and step-size adaptation stay well-posed
            # even when candidates land in constraint-violating regions.
            losses = jnp.where(jnp.isfinite(losses), losses, jnp.float32(1e30))

            # Truncation selection: keep best μ (comma semantics)
            sorted_idx = jnp.argsort(losses)[:mu]
            selected = offspring[sorted_idx]         # (mu, n)

            # Uniform recombination (intermediate recombination)
            mean = jnp.clip(jnp.mean(selected, axis=0), lb, ub)

            # Cumulative step-size adaptation based on actual improvement
            mean_selected_loss = float(jnp.mean(losses[sorted_idx]))
            improved = 1.0 if mean_selected_loss < prev_mean_loss else 0.0
            prev_mean_loss = mean_selected_loss
            p_succ = (1.0 - c_s) * p_succ + c_s * improved
            sigma = sigma * float(jnp.exp(jnp.array((p_succ - target_succ) / d_sigma)))
            sigma = max(sigma, sigma_min)

            iteration += 1
