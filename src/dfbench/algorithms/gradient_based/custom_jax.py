"""Native-JAX custom and hybrid gradient-based optimizers.

These implementations are intentionally benchmark-oriented and lightweight.
They follow the dfbench algorithm contract:

- Never create an Objective inside the algorithm.
- Warm up JIT paths before calling ``obj.start_logging()``.
- Use Objective methods directly whenever possible.
- If internals must bypass Objective wrappers (e.g. Optax L-BFGS internals),
  use ``obj.log_evaluation(...)`` to keep evaluation counting and timing fair.

Bounded vs unbounded
--------------------
All algorithms in this file default to ``unbounded=True`` and therefore optimize
in unconstrained coordinates while Objective maps to bounded problem space.
This is explicit and intentional for gradient / quasi-gradient methods.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective


def _split_once(rng_key: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Split RNG key and return (new_key, subkey)."""
    new_key, subkey = jax.random.split(rng_key)
    return new_key, subkey


def _maybe_restart_params(
    *,
    obj: Objective,
    params: Float[Array, " n_params"],
    rng_key: jax.Array,
    step: int,
    restart_every: int | None,
    restart_noise_std: float,
    restart_from_best: bool,
) -> tuple[Float[Array, " n_params"], jax.Array, bool]:
    """Apply periodic restarts in unbounded space.

    Returns:
        (possibly updated params, new rng_key, did_restart)
    """
    if restart_every is None or restart_every <= 0 or step <= 0:
        return params, rng_key, False
    if step % restart_every != 0:
        return params, rng_key, False

    if restart_from_best and obj.best_params is not None:
        base = jnp.asarray(obj.best_params)
    else:
        base = obj.random_params_unbounded()

    if restart_noise_std <= 0.0:
        return base, rng_key, True

    rng_key, noise_key = _split_once(rng_key)
    noise = restart_noise_std * jax.random.normal(noise_key, shape=base.shape)
    return base + noise, rng_key, True


def _lbfgs_linesearch_eval_count(opt_state) -> int:
    """Extract Optax LBFGS internal line-search steps from optimizer state."""
    states = opt_state if isinstance(opt_state, tuple) else (opt_state,)
    for state in reversed(states):
        info = getattr(state, "info", None)
        if info is None:
            continue
        num_steps = getattr(info, "num_linesearch_steps", None)
        if num_steps is not None:
            return int(jax.device_get(num_steps))
    return 0


class SGLDJAX(OptimizationAlgorithm):
    """Stochastic Gradient Langevin Dynamics (native JAX, optimizer style).

    This is not a full Bayesian posterior sampler. It is an optimizer-style
    SGLD variant for benchmarking with controlled injected noise.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).
    """

    algorithm_str = "sgld_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.05,
        temperature: float = 1.0,
        patience: int | None = None,
        restart_every: int | None = None,
        restart_noise_std: float = 0.05,
        restart_from_best: bool = True,
    ) -> None:
        obj = problem_objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params

        _ = obj.value_and_grad(params)
        obj.start_logging()

        step = 0
        while not obj.budget_exceeded:
            loss, grad = obj.value_and_grad(params)
            if patience is not None and obj.evals_since_improvement > patience:
                break

            rng_key, noise_key = _split_once(rng_key)
            noise = jax.random.normal(noise_key, shape=params.shape)
            drift = -0.5 * learning_rate * grad
            diffusion = jnp.sqrt(learning_rate * max(temperature, 0.0)) * noise
            params = params + drift + diffusion

            step += 1
            params, rng_key, _ = _maybe_restart_params(
                obj=obj,
                params=params,
                rng_key=rng_key,
                step=step,
                restart_every=restart_every,
                restart_noise_std=restart_noise_std,
                restart_from_best=restart_from_best,
            )


class ASAMJAX(OptimizationAlgorithm):
    """Adaptive SAM (ASAM) in native JAX.

    Thin adaptive-SAM variant:
    - Build an adversarial perturbation from gradient and parameter magnitude.
    - Evaluate gradient at perturbed point.
    - Apply Adam update using perturbed gradient.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).
    """

    algorithm_str = "asam_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.05,
        rho: float = 0.2,
        eta: float = 1e-12,
        patience: int | None = None,
        restart_every: int | None = None,
        restart_noise_std: float = 0.02,
        restart_from_best: bool = True,
    ) -> None:
        obj = problem_objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params

        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate))
        opt_state = optimizer.init(params)

        # Warmup both value_and_grad paths used in-loop.
        _ = obj.value_and_grad(params)
        scale = jnp.abs(params) + eta
        grad_norm = jnp.linalg.norm(scale * jnp.ones_like(params)) + 1e-12
        _ = obj.value_and_grad(params + rho * scale / grad_norm)
        obj.start_logging()

        step = 0
        while not obj.budget_exceeded:
            _, grad = obj.value_and_grad(params)
            scale = jnp.abs(params) + eta
            perturb = rho * (scale * grad) / (jnp.linalg.norm(scale * grad) + 1e-12)
            _, adv_grad = obj.value_and_grad(params + perturb)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            updates, opt_state = optimizer.update(adv_grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            step += 1
            params, rng_key, did_restart = _maybe_restart_params(
                obj=obj,
                params=params,
                rng_key=rng_key,
                step=step,
                restart_every=restart_every,
                restart_noise_std=restart_noise_std,
                restart_from_best=restart_from_best,
            )
            if did_restart:
                opt_state = optimizer.init(params)


class AdamToLBFGSJAX(OptimizationAlgorithm):
    """Two-stage hybrid optimizer: Adam exploration then Optax L-BFGS refinement.

    Stage 1 uses Adam for coarse exploration.
    Stage 2 starts from the current incumbent and refines with Optax L-BFGS.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).

    Logging fairness:
        L-BFGS internal line-search probes are replay-logged via
        ``obj.log_evaluation(...)`` to keep eval/time accounting benchmark-fair.
    """

    algorithm_str = "adam_to_lbfgs_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        adam_learning_rate: float = 0.05,
        adam_fraction: float = 0.6,
        min_adam_steps: int = 20,
        patience: int | None = None,
        lbfgs_patience: int | None = None,
        **lbfgs_kwargs,
    ) -> None:
        obj = problem_objective
        problem = obj.problem
        _, _ = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params

        adam_opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(adam_learning_rate),
        )
        adam_state = adam_opt.init(params)

        value_fn = problem.sigmoid_objective_function
        value_and_grad_fn = jax.value_and_grad(value_fn)
        lbfgs_opt = optax.lbfgs(**lbfgs_kwargs)
        lbfgs_state = lbfgs_opt.init(params)

        @jax.jit
        def _lbfgs_step(p, state):
            loss, grad = value_and_grad_fn(p)
            updates, new_state = lbfgs_opt.update(
                grad,
                state,
                p,
                value=loss,
                grad=grad,
                value_fn=value_fn,
            )
            new_p = optax.apply_updates(p, updates)
            return new_p, new_state, loss, grad

        _ = obj.value_and_grad(params)
        warm_state = lbfgs_opt.init(params)
        _, warm_state, _, _ = _lbfgs_step(params, warm_state)
        _ = _lbfgs_step(params, warm_state)
        obj.start_logging()

        initial_budget = obj.evals_left
        if initial_budget is None:
            adam_budget = min_adam_steps
        else:
            target = int(max(1, round(float(initial_budget) * adam_fraction)))
            adam_budget = min(max(1, target), max(1, initial_budget - 1))
            adam_budget = max(adam_budget, min_adam_steps)

        # Stage 1: Adam exploration
        adam_steps = 0
        while not obj.budget_exceeded and adam_steps < adam_budget:
            loss, grad = obj.value_and_grad(params)
            if patience is not None and obj.evals_since_improvement > patience:
                break
            updates, adam_state = adam_opt.update(grad, adam_state, params)
            params = optax.apply_updates(params, updates)
            adam_steps += 1

        # Stage 2: L-BFGS refinement
        lbfgs_state = lbfgs_opt.init(params)
        while not obj.budget_exceeded:
            prior_params = params
            params, lbfgs_state, loss, grad = _lbfgs_step(params, lbfgs_state)
            obj.log_evaluation(prior_params, loss, grad)

            extra_evals = _lbfgs_linesearch_eval_count(lbfgs_state)
            for _ in range(extra_evals):
                if obj.budget_exceeded:
                    break
                obj.log_evaluation(prior_params, loss, grad)

            if lbfgs_patience is not None and obj.evals_since_improvement > lbfgs_patience:
                break


class EntropySGDJAX(OptimizationAlgorithm):
    """Entropy-SGD with a minimal local-entropy inner loop.

    This implementation keeps a small inner loop around the current iterate,
    averaging noisy local gradients and then taking one outer update step.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).
    """

    algorithm_str = "entropy_sgd_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.03,
        inner_steps: int = 3,
        inner_step_size: float = 0.05,
        local_noise_std: float = 0.02,
        patience: int | None = None,
        restart_every: int | None = None,
        restart_noise_std: float = 0.03,
        restart_from_best: bool = True,
    ) -> None:
        obj = problem_objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params
        _ = obj.value_and_grad(params)
        obj.start_logging()

        step = 0
        while not obj.budget_exceeded:
            local = params
            avg_grad = jnp.zeros_like(params)

            for _ in range(max(1, inner_steps)):
                if obj.budget_exceeded:
                    break
                loss, grad = obj.value_and_grad(local)
                avg_grad = avg_grad + grad
                rng_key, nkey = _split_once(rng_key)
                noise = local_noise_std * jax.random.normal(nkey, shape=local.shape)
                local = local - inner_step_size * grad + noise

            if patience is not None and obj.evals_since_improvement > patience:
                break

            avg_grad = avg_grad / max(1, inner_steps)
            params = params - learning_rate * avg_grad

            step += 1
            params, rng_key, _ = _maybe_restart_params(
                obj=obj,
                params=params,
                rng_key=rng_key,
                step=step,
                restart_every=restart_every,
                restart_noise_std=restart_noise_std,
                restart_from_best=restart_from_best,
            )


class SGHMCJAX(OptimizationAlgorithm):
    """Stochastic Gradient Hamiltonian Monte Carlo style optimizer.

    Implemented as momentum dynamics with friction and Gaussian noise.
    Intended as an optimization-oriented stochastic dynamics method.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).
    """

    algorithm_str = "sghmc_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.01,
        momentum_decay: float = 0.05,
        noise_scale: float = 0.01,
        patience: int | None = None,
        restart_every: int | None = None,
        restart_noise_std: float = 0.05,
        restart_from_best: bool = True,
    ) -> None:
        obj = problem_objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params
        momentum = jnp.zeros_like(params)

        _ = obj.value_and_grad(params)
        obj.start_logging()

        step = 0
        while not obj.budget_exceeded:
            _, grad = obj.value_and_grad(params)
            if patience is not None and obj.evals_since_improvement > patience:
                break

            rng_key, nkey = _split_once(rng_key)
            noise = noise_scale * jax.random.normal(nkey, shape=params.shape)
            momentum = (1.0 - momentum_decay) * momentum - learning_rate * grad + noise
            params = params + momentum

            step += 1
            params, rng_key, did_restart = _maybe_restart_params(
                obj=obj,
                params=params,
                rng_key=rng_key,
                step=step,
                restart_every=restart_every,
                restart_noise_std=restart_noise_std,
                restart_from_best=restart_from_best,
            )
            if did_restart:
                momentum = jnp.zeros_like(params)


class ARCJAX(OptimizationAlgorithm):
    """Adaptive Regularization with Cubics placeholder (native JAX).

    ARC is intentionally not implemented in this batch. A stable ARC
    implementation requires robust Hessian linear algebra and careful trust-region
    bookkeeping that is currently too fragile for benchmark-fair default use here.

    Space behavior:
        Would use unbounded coordinates by default if implemented.
    """

    algorithm_str = "arc_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        **kwargs,
    ) -> None:
        raise NotImplementedError(
            "ARCJAX is intentionally disabled in this batch: stable and readable "
            "ARC Hessian/trust-region internals are not benchmark-ready yet."
        )


class OGDJAX(OptimizationAlgorithm):
    """Optimistic Gradient Descent (native JAX).

    Uses the optimistic update direction ``2*g_t - g_{t-1}``.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).
    """

    algorithm_str = "ogd_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.05,
        patience: int | None = None,
        restart_every: int | None = None,
        restart_noise_std: float = 0.02,
        restart_from_best: bool = True,
    ) -> None:
        obj = problem_objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params
        prev_grad = None

        _ = obj.value_and_grad(params)
        obj.start_logging()

        step = 0
        while not obj.budget_exceeded:
            _, grad = obj.value_and_grad(params)
            if patience is not None and obj.evals_since_improvement > patience:
                break

            if prev_grad is None:
                optimistic_grad = grad
            else:
                optimistic_grad = 2.0 * grad - prev_grad
            params = params - learning_rate * optimistic_grad
            prev_grad = grad

            step += 1
            params, rng_key, did_restart = _maybe_restart_params(
                obj=obj,
                params=params,
                rng_key=rng_key,
                step=step,
                restart_every=restart_every,
                restart_noise_std=restart_noise_std,
                restart_from_best=restart_from_best,
            )
            if did_restart:
                prev_grad = None


class OAdamJAX(OptimizationAlgorithm):
    """Optimistic Adam (native JAX).

    Combines optimistic gradients ``2*g_t - g_{t-1}`` with Adam preconditioning.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).
    """

    algorithm_str = "oadam_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.05,
        patience: int | None = None,
        restart_every: int | None = None,
        restart_noise_std: float = 0.02,
        restart_from_best: bool = True,
        **adam_kwargs,
    ) -> None:
        obj = problem_objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params
        prev_grad = None

        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs))
        opt_state = optimizer.init(params)

        _ = obj.value_and_grad(params)
        obj.start_logging()

        step = 0
        while not obj.budget_exceeded:
            _, grad = obj.value_and_grad(params)
            if patience is not None and obj.evals_since_improvement > patience:
                break

            if prev_grad is None:
                optimistic_grad = grad
            else:
                optimistic_grad = 2.0 * grad - prev_grad
            updates, opt_state = optimizer.update(optimistic_grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            prev_grad = grad

            step += 1
            params, rng_key, did_restart = _maybe_restart_params(
                obj=obj,
                params=params,
                rng_key=rng_key,
                step=step,
                restart_every=restart_every,
                restart_noise_std=restart_noise_std,
                restart_from_best=restart_from_best,
            )
            if did_restart:
                prev_grad = None
                opt_state = optimizer.init(params)


class PerturbedGDJAX(OptimizationAlgorithm):
    """Gradient descent with occasional random perturbations.

    This is a simple ruggedness-control baseline: plain gradient descent with
    periodic additive Gaussian perturbations.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).
    """

    algorithm_str = "perturbed_gd_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.05,
        perturb_every: int = 25,
        perturb_std: float = 0.05,
        patience: int | None = None,
        restart_every: int | None = None,
        restart_noise_std: float = 0.02,
        restart_from_best: bool = True,
    ) -> None:
        obj = problem_objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params

        _ = obj.value_and_grad(params)
        obj.start_logging()

        step = 0
        while not obj.budget_exceeded:
            _, grad = obj.value_and_grad(params)
            if patience is not None and obj.evals_since_improvement > patience:
                break

            params = params - learning_rate * grad

            if perturb_every > 0 and step > 0 and step % perturb_every == 0:
                rng_key, pkey = _split_once(rng_key)
                params = params + perturb_std * jax.random.normal(pkey, shape=params.shape)

            step += 1
            params, rng_key, _ = _maybe_restart_params(
                obj=obj,
                params=params,
                rng_key=rng_key,
                step=step,
                restart_every=restart_every,
                restart_noise_std=restart_noise_std,
                restart_from_best=restart_from_best,
            )


class NoisyAdamJAX(OptimizationAlgorithm):
    """Adam with fixed-scale additive Gaussian noise.

    Kept intentionally simple as a ruggedness-control baseline.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).
    """

    algorithm_str = "noisy_adam_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.05,
        noise_std: float = 0.01,
        patience: int | None = None,
        restart_every: int | None = None,
        restart_noise_std: float = 0.02,
        restart_from_best: bool = True,
        **adam_kwargs,
    ) -> None:
        obj = problem_objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params

        optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs))
        opt_state = optimizer.init(params)

        _ = obj.value_and_grad(params)
        obj.start_logging()

        step = 0
        while not obj.budget_exceeded:
            _, grad = obj.value_and_grad(params)
            if patience is not None and obj.evals_since_improvement > patience:
                break

            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)

            if noise_std > 0:
                rng_key, nkey = _split_once(rng_key)
                params = params + noise_std * jax.random.normal(nkey, shape=params.shape)

            step += 1
            params, rng_key, did_restart = _maybe_restart_params(
                obj=obj,
                params=params,
                rng_key=rng_key,
                step=step,
                restart_every=restart_every,
                restart_noise_std=restart_noise_std,
                restart_from_best=restart_from_best,
            )
            if did_restart:
                opt_state = optimizer.init(params)


class GDRestartsJAX(OptimizationAlgorithm):
    """Gradient descent with first-class periodic restarts.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).
    """

    algorithm_str = "gd_restarts_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.05,
        patience: int | None = None,
        restart_every: int | None = 50,
        restart_noise_std: float = 0.03,
        restart_from_best: bool = True,
    ) -> None:
        obj = problem_objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params

        _ = obj.value_and_grad(params)
        obj.start_logging()

        step = 0
        while not obj.budget_exceeded:
            _, grad = obj.value_and_grad(params)
            if patience is not None and obj.evals_since_improvement > patience:
                break

            params = params - learning_rate * grad
            step += 1

            params, rng_key, _ = _maybe_restart_params(
                obj=obj,
                params=params,
                rng_key=rng_key,
                step=step,
                restart_every=restart_every,
                restart_noise_std=restart_noise_std,
                restart_from_best=restart_from_best,
            )


class GaussianSmoothingGDJAX(OptimizationAlgorithm):
    """Gaussian-smoothing gradient descent.

    Uses finite-difference-style Gaussian smoothing with antithetic perturbations
    to estimate gradients in rugged landscapes.

    Space behavior:
        Uses unbounded optimization coordinates by default (``unbounded=True``).
    """

    algorithm_str = "gaussian_smoothing_gd_jax"
    algorithm_type = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.03,
        sigma: float = 0.05,
        n_directions: int = 4,
        patience: int | None = None,
        restart_every: int | None = None,
        restart_noise_std: float = 0.02,
        restart_from_best: bool = True,
    ) -> None:
        obj = problem_objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = obj.random_params_unbounded() if init_params is None else init_params

        # Warmup the exact calls used in-loop.
        _ = obj.value(params)
        _ = obj.value(params + 1e-3)
        obj.start_logging()

        step = 0
        dirs = max(1, int(n_directions))
        while not obj.budget_exceeded:
            base_loss = obj.value(params)
            if patience is not None and obj.evals_since_improvement > patience:
                break

            est_grad = jnp.zeros_like(params)
            for _ in range(dirs):
                if obj.budget_exceeded:
                    break
                rng_key, dkey = _split_once(rng_key)
                direction = jax.random.normal(dkey, shape=params.shape)
                norm = jnp.linalg.norm(direction) + 1e-12
                direction = direction / norm

                plus = params + sigma * direction
                minus = params - sigma * direction

                f_plus = obj.value(plus)
                if obj.budget_exceeded:
                    break
                f_minus = obj.value(minus)

                est_grad = est_grad + ((f_plus - f_minus) / (2.0 * sigma)) * direction

            est_grad = est_grad / float(dirs)
            params = params - learning_rate * est_grad

            step += 1
            params, rng_key, _ = _maybe_restart_params(
                obj=obj,
                params=params,
                rng_key=rng_key,
                step=step,
                restart_every=restart_every,
                restart_noise_std=restart_noise_std,
                restart_from_best=restart_from_best,
            )
