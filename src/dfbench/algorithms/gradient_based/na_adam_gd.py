import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from collections import deque
from typing import Literal
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from differometor.utils import sigmoid_bounding
from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)


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

    Stopping criteria:
        - With wall_times: runs until max wall time, records iteration indices
          at each checkpoint
        - Without wall_times: runs up to max_iterations, stops early if no
          improvement > 1e-4 for patience iterations
    """

    algorithm_str: str = "na_adam_gd"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(self, problem: ContinuousProblem) -> None:
        """Initialize the NA-Adam optimizer.

        Args:
            problem: A continuous optimization problem that provides:
                - sigmoid_objective_function: loss function in unbounded space
                - n_params: number of parameters to optimize
                - bounds: parameter bounds for final sigmoid transformation
                - output_to_files: method to save results
        """
        self._problem = problem

        self._grad_fn = jax.jit(
            jax.value_and_grad(self._problem.sigmoid_objective_function)
        )

    @jaxtyped(typechecker=typechecker)
    def optimize(
        self,
        save_to_file: bool = True,
        init_params: Float[Array, "{self._problem.n_params}"] | None = None,
        return_best_params_history: bool = False,
        random_seed: int | None = None,
        wall_times: list[int | float] | None = None,
        learning_rate: float = 0.1,
        max_iterations: int = 50000,
        patience: int = 1000,
        noise_std_start: float = 0.3,
        noise_std_end: float = 0.0,
        noise_schedule: NoiseSchedule = "exponential",
        noise_injection: NoiseInjection = "update",
        noise_clip_norm: float | None = None,
        noise_anneal_iters: int = 5000,
        noise_cap_relative_to_update: float | None = 0.25,
        noise_cap_start_iter: int = 500,
        **adam_kwargs,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_iters {self._problem.n_params}"] | None,
        Float[Array, "n_iters"],
        list[int] | None,
    ]:
        """Run the NA-Adam optimization loop.

        Args:
            save_to_file: If True, saves best params and losses via
                problem.output_to_files(). Defaults to True.
            init_params: Starting point in unbounded space. If None, samples
                uniformly from [-10, 10] for each parameter. Defaults to None.
            return_best_params_history: If True, records the best params found
                so far at each iteration. Memory-intensive. Defaults to False.
            random_seed: Seed for NumPy (init params) and JAX (noise). If None,
                uses current time. Defaults to None.
            wall_times: List of wall-clock checkpoints in seconds, e.g. [10, 60, 300].
                If provided, runs until max(wall_times) and records which iteration
                was reached at each checkpoint. Ignores max_iterations and patience.
                Defaults to None.
            learning_rate: Adam learning rate. Defaults to 0.1.
            max_iterations: Hard cap on iterations (only used if wall_times is None).
                Defaults to 50,000.
            patience: Early stopping trigger. Stops if best loss doesn't improve
                by > 1e-4 for this many consecutive iterations. Only used if
                wall_times is None. Defaults to 1,000.
            noise_std_start: Initial noise standard deviation. Defaults to 0.3.
            noise_std_end: Final noise standard deviation. Defaults to 0.0.
            noise_schedule: How noise decays over time.
                - "exponential" (default): geometric decay, sigma_t = start * (end/start)^progress
                - "linear": sigma_t = start + (end - start) * progress
            noise_injection: Where to add noise.
                - "update" (default): add to Adam update before applying
                - "params": add to params after applying Adam update
            noise_clip_norm: If set, clips noise vector to this max L2 norm.
                Defaults to None (no clipping).
            noise_anneal_iters: Number of iterations over which noise decays from
                start to end value. After this, noise stays at noise_std_end.
                Defaults to 5,000.
            noise_cap_relative_to_update: If set, caps noise norm to this fraction
                of the Adam update norm. Prevents noise from dominating when
                gradients are small. Defaults to 0.25 (noise <= 25% of update).
            noise_cap_start_iter: Iteration at which relative noise capping
                activates. Allows unrestricted exploration early on.
                Defaults to 500.
            **adam_kwargs: Passed to optax.adam(). Useful options:
                - b1: first moment decay (default 0.9)
                - b2: second moment decay (default 0.999)
                - eps: numerical stability term (default 1e-8)

        Returns:
            A 4-tuple:
                - best_params: Best parameters found (in bounded space).
                - best_params_history: Array of shape (n_iters, n_params) with
                    best params at each iteration (bounded), or None if
                    return_best_params_history=False.
                - losses: Array of shape (n_iters,) with loss at each iteration.
                - wall_time_indices: List of iteration indices reached at each
                    wall_times checkpoint (sorted), or None if wall_times=None.
        """
        # Seed both NumPy (init params) and JAX (noise)
        if random_seed is not None:
            np.random.seed(random_seed)
            rng_key = jax.random.PRNGKey(random_seed)
        else:
            rng_key = jax.random.PRNGKey(int(time.time() * 1000) % (2**31))

        # Initialize parameters
        best_params: Float[Array, "{self._problem.n_params}"] = (
            jnp.array(np.random.uniform(-10, 10, self._problem.n_params))
            if init_params is None
            else init_params
        )

        # warmup the function to compile it
        _ = self._grad_fn(best_params)

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(best_params)

        params, losses = best_params, []
        best_params_history = []  # later shape: (n_iterations, n_params)
        best_loss = 1e10

        # Initialize wall_time_indices tracking
        wall_time_indices: list[int] | None = None
        wall_times_remaining: deque[int | float] | None = None
        if wall_times is not None:
            wall_time_indices = []
            wall_times_remaining = deque(sorted(wall_times))
            max_wall_time = wall_times_remaining[-1]

        # Separate loops for wall-time constrained vs iteration/patience constrained
        if wall_times is not None:
            # Wall-time constrained: ignore max_iterations and patience
            start_time = time.time()
            i = 0
            while (time.time() - start_time) < max_wall_time:
                elapsed = time.time() - start_time
                progress = i / max(1, noise_anneal_iters)
                sigma_t = _anneal_sigma(
                    progress=progress,
                    sigma_start=noise_std_start,
                    sigma_end=noise_std_end,
                    schedule=noise_schedule,
                )

                # Record iteration index at wall_times checkpoints
                while wall_times_remaining and elapsed >= wall_times_remaining[0]:
                    wall_time_indices.append(i)
                    wall_times_remaining.popleft()

                loss, grads = self._grad_fn(params)

                if i % 100 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                if loss < best_loss - 1e-4:
                    best_loss, best_params = loss, params
                    print(f"Iteration {i}: New best loss = {loss}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                updates, optimizer_state = optimizer.update(
                    grads, optimizer_state, params
                )

                if sigma_t > 0:
                    rng_key, subkey = jax.random.split(rng_key)
                    noise_step = jax.random.normal(subkey, shape=params.shape) * sigma_t
                    if noise_clip_norm is not None:
                        noise_step = _clip_step_by_global_norm(
                            noise_step, noise_clip_norm
                        )

                    if (
                        noise_cap_relative_to_update is not None
                        and i >= noise_cap_start_iter
                    ):
                        noise_step = _cap_step_relative(
                            noise_step, updates, float(noise_cap_relative_to_update)
                        )

                    if noise_injection == "update":
                        updates = updates + noise_step
                    elif noise_injection == "params":
                        # Apply optimizer update first, then perturb.
                        pass

                params = optax.apply_updates(params, updates)

                if sigma_t > 0 and noise_injection == "params":
                    params = params + noise_step
                losses.append(float(loss))
                i += 1

            # Fill remaining wall_times that weren't reached with final iteration
            while wall_times_remaining:
                wall_time_indices.append(i - 1 if i > 0 else 0)
                wall_times_remaining.popleft()
        else:
            # Iteration/patience constrained
            no_improve_count = 0
            for i in range(max_iterations):
                progress = i / max(1, noise_anneal_iters)
                sigma_t = _anneal_sigma(
                    progress=progress,
                    sigma_start=noise_std_start,
                    sigma_end=noise_std_end,
                    schedule=noise_schedule,
                )
                loss, grads = self._grad_fn(params)

                if i % 500 == 0:
                    print(f"Iteration {i}: Loss = {loss}")

                if loss < best_loss - 1e-4:
                    best_loss, best_params, no_improve_count = loss, params, 0
                    print(f"Iteration {i}: New best loss = {loss}")
                else:
                    no_improve_count += 1

                if return_best_params_history:
                    best_params_history.append(best_params)

                updates, optimizer_state = optimizer.update(
                    grads, optimizer_state, params
                )

                if sigma_t > 0:
                    rng_key, subkey = jax.random.split(rng_key)
                    noise_step = jax.random.normal(subkey, shape=params.shape) * sigma_t
                    if noise_clip_norm is not None:
                        noise_step = _clip_step_by_global_norm(
                            noise_step, noise_clip_norm
                        )

                    if (
                        noise_cap_relative_to_update is not None
                        and i >= noise_cap_start_iter
                    ):
                        noise_step = _cap_step_relative(
                            noise_step, updates, float(noise_cap_relative_to_update)
                        )

                    if noise_injection == "update":
                        updates = updates + noise_step
                    elif noise_injection == "params":
                        pass

                params = optax.apply_updates(params, updates)

                if sigma_t > 0 and noise_injection == "params":
                    params = params + noise_step
                losses.append(float(loss))

                if no_improve_count > patience:
                    break

        losses = jnp.array(losses)
        best_params_history = (
            jnp.array(best_params_history) if return_best_params_history else None
        )

        # Transform unbounded parameters back to bounded space
        best_params_bounded = sigmoid_bounding(best_params, self._problem.bounds)
        if return_best_params_history:
            sigmoid_bounding_v = jax.vmap(
                lambda p: sigmoid_bounding(p, self._problem.bounds)
            )
            best_params_history_bounded = sigmoid_bounding_v(best_params_history)
        else:
            best_params_history_bounded = None

        if save_to_file:
            self._problem.output_to_files(
                best_params=best_params_bounded,
                losses=losses,
                algorithm_str=self.algorithm_str,
                hyper_param_str=(
                    f"lr{learning_rate}"
                    f"_ns{noise_std_start}"
                    f"_ne{noise_std_end}"
                    f"_{noise_schedule}"
                    f"_{noise_injection}"
                ),
            )  # TODO maybe conditionally add more hyperparameters to string

        return (
            best_params_bounded,
            best_params_history_bounded,
            losses,
            wall_time_indices,
        )
