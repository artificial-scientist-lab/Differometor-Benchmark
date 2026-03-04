import jax
import jax.numpy as jnp
import numpy as np
import optax
from typing import Literal
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


NoiseSchedule = Literal["linear", "exponential"]
NoiseInjection = Literal["update", "params"]


def _anneal_sigma(
    progress: float,
    sigma_start: float,
    sigma_end: float,
    schedule: NoiseSchedule,
) -> float:
    """Compute noise standard deviation at a given point in the annealing schedule.

    Args:
        progress: Fraction of annealing complete, in [0, 1]. Values outside
            this range are clipped.
        sigma_start: Initial noise standard deviation (at progress=0).
        sigma_end: Final noise standard deviation (at progress=1).
        schedule: Annealing curve type.
            - "linear": sigma = sigma_start + (sigma_end - sigma_start) * progress
            - "exponential": sigma = sigma_start * (sigma_end / sigma_start)^progress
              (geometric interpolation, smoother decay)

    Returns:
        The noise standard deviation for the current progress.
    """
    progress = float(np.clip(progress, 0.0, 1.0))

    if schedule == "linear":
        return sigma_start + (sigma_end - sigma_start) * progress

    # Exponential interpolation (geometric) between start and end.
    # Handles sigma_end == 0 by using a tiny floor.
    eps = 1e-12
    s0 = max(float(sigma_start), 0.0)
    s1 = max(float(sigma_end), 0.0)
    if s0 <= 0.0:
        return 0.0
    ratio = max(s1, eps) / max(s0, eps)
    return s0 * (ratio**progress)


def _clip_step_by_global_norm(step: jnp.ndarray, max_norm: float) -> jnp.ndarray:
    """Clip a vector so its L2 norm does not exceed max_norm.

    Args:
        step: The vector to clip.
        max_norm: Maximum allowed L2 norm.

    Returns:
        The original step if its norm <= max_norm, otherwise a scaled-down
        version with norm exactly equal to max_norm.
    """
    norm = jnp.linalg.norm(step)
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-12))
    return step * scale


def _cap_step_relative(
    step: jnp.ndarray, reference: jnp.ndarray, ratio: float
) -> jnp.ndarray:
    """Cap a vector's norm relative to a reference vector's norm.

    Ensures: ||output|| <= ratio * ||reference||.

    This is used to prevent the noise step from overwhelming the optimizer
    update. For example, with ratio=0.25, the noise cannot exceed 25% of
    the update magnitude.

    Args:
        step: The vector to potentially shrink (typically the noise step).
        reference: The reference vector (typically the Adam update).
        ratio: Maximum allowed ratio of step norm to reference norm.

    Returns:
        The original step if already within bounds, otherwise scaled down.
    """
    step_norm = jnp.linalg.norm(step)
    ref_norm = jnp.linalg.norm(reference)
    max_norm = ratio * ref_norm
    scale = jnp.minimum(1.0, max_norm / (step_norm + 1e-12))
    return step * scale


class NAAdamGD(OptimizationAlgorithm):
    """Noisy-Annealing Adam (NA-Adam) optimizer.

    Combines Adam optimization with decaying Gaussian noise injection for
    exploration. High noise early on helps escape local minima; noise decays
    over iterations to allow fine-tuning convergence.

    All history tracking, printing, and checkpointing is handled by the
    `Objective` wrapper. The algorithm loop is minimal.

    Algorithm per iteration:
        1. Compute loss and gradient via autodiff
        2. Get Adam update (with gradient clipping, norm <= 1.0)
        3. Generate noise: xi ~ N(0, sigma_t^2 * I)
        4. Apply safety caps to noise (optional, see below)
        5. Combine: params = params + adam_update + noise

    Noise schedule:
        sigma_t decays from noise_std_start to noise_std_end over
        noise_anneal_iters iterations. Two schedules available:
        - "linear": constant decay rate
        - "exponential": geometric decay (fast early drop, then tail off)

    Noise injection modes:
        - "update": noise added to the Adam update before applying
        - "params": noise added to parameters after the Adam update

    Stability controls:
        - noise_clip_norm: hard cap on noise vector's L2 norm
        - noise_cap_relative_to_update: caps noise norm to this fraction of
          the Adam update norm. Activates after noise_cap_start_iter to
          prevent noise from dominating when gradients are small.

    Bounded optimization:
        Internally optimizes in unbounded space. The objective function uses
        sigmoid bounding to map unbounded params to the constrained domain.
        Final results are returned in bounded space.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("na_adam_gd").
        algorithm_type (AlgorithmType): Type classification (GRADIENT_BASED).
    """

    algorithm_str: str = "na_adam_gd"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        """Initialize the NA-Adam optimizer."""
        pass

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.1,
        patience: int | None = None,
        noise_std_start: float = 0.3,
        noise_std_end: float = 0.0,
        noise_schedule: NoiseSchedule = "exponential",
        noise_injection: NoiseInjection = "update",
        noise_clip_norm: float | None = None,
        noise_anneal_iters: int = 5000,
        noise_cap_relative_to_update: float | None = 0.25,
        noise_cap_start_iter: int = 500,
        **adam_kwargs,
    ) -> None:
        """Run the NA-Adam optimization loop using Objective for logging.

        Each iteration performs exactly one ``value_and_grad`` call, so the
        evaluation budget on the Objective (``max_evals``) directly controls
        the number of gradient steps.

        Args:
            problem_objective: The Objective instance wrapping the problem.
            init_params: Starting point in unbounded space. If None, initialized
                via obj.random_params_unbounded(). Defaults to None.
            random_seed: Seed for NumPy and JAX. If None,
                uses system entropy. Defaults to None.
            learning_rate: Adam learning rate. Defaults to 0.1.
            patience: Early stopping trigger. Stops if best loss doesn't improve
                for this many consecutive iterations. Defaults to None (no early stopping).
            noise_std_start: Initial noise standard deviation. Defaults to 0.3.
            noise_std_end: Final noise standard deviation. Defaults to 0.0.
            noise_schedule: How noise decays over time.
                - "exponential" (default): geometric decay
                - "linear": linear decay
            noise_injection: Where to add noise.
                - "update" (default): add to Adam update before applying
                - "params": add to params after applying Adam update
            noise_clip_norm: If set, clips noise vector to this max L2 norm.
                Defaults to None (no clipping).
            noise_anneal_iters: Number of iterations over which noise decays from
                start to end value. Defaults to 5,000.
            noise_cap_relative_to_update: If set, caps noise norm to this fraction
                of the Adam update norm. Defaults to 0.25.
            noise_cap_start_iter: Iteration at which relative noise capping
                activates. Defaults to 500.
            **adam_kwargs: Passed to optax.adam().
        """
        obj = problem_objective
        problem = obj.problem

        random_seed, rng_key = self.prepare(
            obj, unbounded=True, random_seed=random_seed
        )

        if init_params is None:
            params = obj.random_params_unbounded()
        else:
            params = init_params

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(params)

        # Warm-up JIT
        _ = obj.value_and_grad(params)

        obj.start_logging()

        iteration = 0
        while not obj.budget_exceeded:
            progress = iteration / max(1, noise_anneal_iters)
            sigma_t = _anneal_sigma(
                progress=progress,
                sigma_start=noise_std_start,
                sigma_end=noise_std_end,
                schedule=noise_schedule,
            )

            loss, grads = obj.value_and_grad(params)

            # Early stopping: patience check
            if patience is not None and obj.evals_since_improvement > patience:
                break

            updates, optimizer_state = optimizer.update(grads, optimizer_state, params)

            if sigma_t > 0:
                rng_key, subkey = jax.random.split(rng_key)
                noise_step = jax.random.normal(subkey, shape=params.shape) * sigma_t
                if noise_clip_norm is not None:
                    noise_step = _clip_step_by_global_norm(noise_step, noise_clip_norm)

                if (
                    noise_cap_relative_to_update is not None
                    and iteration >= noise_cap_start_iter
                ):
                    noise_step = _cap_step_relative(
                        noise_step, updates, float(noise_cap_relative_to_update)
                    )

                if noise_injection == "update":
                    updates = updates + noise_step

            params = optax.apply_updates(params, updates)

            if sigma_t > 0 and noise_injection == "params":
                params = params + noise_step

            iteration += 1
