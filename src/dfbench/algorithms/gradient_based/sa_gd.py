import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
import math
from collections import deque
from jaxtyping import Array, Float, jaxtyped
from beartype import beartype as typechecker

from differometor.utils import sigmoid_bounding
from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)


class SAGD(OptimizationAlgorithm):
    """Simulated Annealing Gradient Descent (SA-GD) optimization algorithm.

    Implements the SA-GD algorithm from the paper:
    "SA-GD: Improved Gradient Descent Learning Strategy with Simulated Annealing"
    (arXiv:2107.07558, Cai 2021)

    The algorithm introduces simulated annealing to gradient descent, giving the
    optimizer a probabilistic "hill-mounting" ability to escape local minima and
    saddle points. With a certain probability (based on temperature and loss
    difference), the algorithm performs gradient ASCENT instead of descent.

    Key equations:
    - Transition probability: P_i = exp(-|ΔE|^k / (T_0 * ε * ln(n+1)))
    - With probability P_i: perform gradient descent (normal)
    - With probability 1-P_i: perform gradient ascent (uphill)

    The probability of going uphill starts low and increases over iterations,
    but stays below a ceiling (default 33%) to ensure convergence.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("sagd").
        algorithm_type (AlgorithmType): Type classification (GRADIENT_BASED).
        _problem (ContinuousProblem): The optimization problem instance.
        _grad_fn (Callable): JIT-compiled gradient function for the objective.

    Note:
        Like AdamGD, this algorithm uses `problem.sigmoid_objective_function`
        for unbounded optimization with internal sigmoid bounding.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = SAGD(problem)
        >>> best_params, history, losses, wall_indices = optimizer.optimize(
        ...     learning_rate=0.1,
        ...     T0=1.0,
        ...     max_iterations=10000,
        ...     wall_times=[30, 60, 120],
        ... )
    """

    algorithm_str: str = "sa_gd"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(self, problem: ContinuousProblem) -> None:
        """Initialize SA-GD optimizer.

        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
        """
        self._problem = problem

        self._grad_fn = jax.jit(
            jax.value_and_grad(self._problem.sigmoid_objective_function)
        )

    def _compute_transition_probability(
        self,
        delta_e: float,
        epoch: int,
        T0: float,
        learning_rate: float,
        use_double_annealing: bool = False,
        lr_decay: float = 1.0,
        initial_lr: float = 0.1,
    ) -> float:
        """Compute the transition probability for SA-GD.

        This determines the probability of performing gradient DESCENT (not ascent).
        When random_value < P_i, we do gradient descent; otherwise gradient ascent.

        Args:
            delta_e: Absolute difference between current and previous loss |ΔE|
            epoch: Current epoch/iteration number (n)
            T0: Initial temperature hyperparameter
            learning_rate: Current learning rate (ε)
            use_double_annealing: Whether to use the double SA formula for decaying LR
            lr_decay: Learning rate decay factor (γ) for double annealing
            initial_lr: Initial learning rate (ε_0) for double annealing

        Returns:
            float: Transition probability P_i in [0, 1]
        """
        # Ensure epoch >= 0 to avoid log(0)
        n = max(epoch, 0)

        # Small epsilon to avoid numerical issues
        eps = 1e-10
        delta_e = max(abs(delta_e), eps)

        if use_double_annealing:
            # Double simulated annealing formula (Eq. 14 in paper)
            # For decaying learning rate strategies
            # P_i = exp(-( |ΔE|^(ln(n+2)^(-1/α)) / (T_0 * ε_0 * γ^n * ln(n+2)) )^(β * ln(n+2)))

            alpha = math.e
            beta = 0.5772  # Euler-Mascheroni constant

            # Fractional power exponent: ln(n+2)^(-1/α)
            frac_power = math.log(n + 2) ** (-1.0 / alpha)

            # Temperature: T_0 * ε_0 * γ^n * ln(n+2)
            current_lr = initial_lr * (lr_decay**n)
            temperature = T0 * current_lr * math.log(n + 2)

            # Numerator: |ΔE|^(fractional power)
            numerator = delta_e**frac_power

            # Inner ratio
            ratio = numerator / max(temperature, eps)

            # Outer exponent: β * ln(n+2)
            outer_exp = beta * math.log(n + 2)

            # Final probability
            exponent = -(ratio**outer_exp)

        else:
            # Simple formula (Eq. 11 in paper)
            # P_i = exp(-|ΔE| / (T_0 * ε * ln(n+1)))
            temperature = (
                T0 * learning_rate * math.log(n + 2)
            )  # Use n+2 to avoid log(1)=0
            exponent = -delta_e / max(temperature, eps)

        # Clamp exponent to avoid overflow
        exponent = max(exponent, -100)

        probability = math.exp(exponent)

        # Clamp probability to [0, 1]
        return min(max(probability, 0.0), 1.0)

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
        T0: float = 15.0,
        sigma: float = 1.0,
        max_ascent_prob: float = 0.33,
        use_double_annealing: bool = False,
        lr_decay: float = 1.0,
        **adam_kwargs,
    ) -> tuple[
        Float[Array, "{self._problem.n_params}"],
        Float[Array, "n_iters {self._problem.n_params}"] | None,
        Float[Array, "n_iters"],
        list[int] | None,
    ]:
        """Run SA-GD (Simulated Annealing Gradient Descent) optimization.

        This algorithm combines gradient descent with simulated annealing concepts.
        It probabilistically performs gradient ascent to escape local minima.

        Args:
            save_to_file (bool): Whether to save optimization results to file. Defaults to True.
            init_params (Float[Array, "n_params"] | None): Initial parameters. If None,
                randomly initialized in range [-10, 10]. Defaults to None.
            return_best_params_history (bool): Whether to track best parameters at each
                iteration. Defaults to False.
            random_seed (int | None): Random seed for reproducibility. Controls initial
                parameter generation and stochastic ascent decisions. Defaults to None.
            wall_times (list[int | float] | None): List of wall-time checkpoints (in seconds).
                The algorithm runs until the maximum checkpoint. At each checkpoint,
                the current iteration index is recorded. If None, runs for max_iterations or
                until patience is exceeded. Defaults to None.
            learning_rate (float): Learning rate for Adam optimizer. Defaults to 0.1.
            max_iterations (int): Maximum number of optimization iterations.
                Only applies when wall_times is None. Defaults to 50,000.
            patience (int): Stop if no improvement (>1e-4) after this many iterations.
                Only applies when wall_times is None. Defaults to 1,000.
            T0 (float): Initial temperature for simulated annealing. Higher values
                lead to higher probability of gradient ascent. Recommended: 15-19 for
                fixed LR, 0.0001 for simple SA without LR consideration. Defaults to 15.0.
            sigma (float): Expansion factor for gradient ascent step size. The gradient
                ascent step is multiplied by this factor. Higher values mean larger
                uphill jumps. Paper uses 1-4. Defaults to 1.0.
            max_ascent_prob (float): Maximum probability of performing gradient ascent.
                Paper recommends keeping this below 0.33 for convergence. Defaults to 0.33.
            use_double_annealing (bool): Whether to use the double simulated annealing
                formula designed for exponentially decaying learning rates. Should be
                True if using lr_decay < 1. Defaults to False.
            lr_decay (float): Learning rate decay factor per iteration. Applied as
                lr_current = learning_rate * (lr_decay ** iteration). Use 1.0 for
                fixed learning rate. Defaults to 1.0.
            **adam_kwargs: Additional keyword arguments passed to optax.adam().
                Common options: b1 (float, default 0.9), b2 (float, default 0.999),
                eps (float, default 1e-8), eps_root (float, default 0.0),
                nesterov (bool, default False).

        Returns:
            tuple: A 4-tuple containing:
                - best_params (Float[Array, "n_params"]): Best parameters found.
                - best_params_history (Float[Array, "n_iters n_params"] | None): History of
                  best parameters per iteration. None if return_best_params_history=False.
                - losses (Float[Array, "n_iters"]): Loss at each iteration.
                - wall_time_indices (list[int] | None): Iteration indices corresponding to
                  each wall_times checkpoint (in sorted ascending order). None if wall_times is None.
        """
        # Set random seed if provided
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

        # Warmup the function to compile it
        _ = self._grad_fn(best_params)

        # Create optimizer with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(best_params)

        params, losses = best_params, []
        best_params_history = []
        best_loss = float("inf")
        prev_loss = 0.0  # Initial previous loss (E_0 = 0 as per paper)

        # Statistics tracking
        ascent_count = 0
        descent_count = 0

        # Initialize wall_time_indices tracking
        wall_time_indices: list[int] | None = None
        wall_times_remaining: deque[int | float] | None = None
        if wall_times is not None:
            wall_time_indices = []
            wall_times_remaining = deque(sorted(wall_times))
            max_wall_time = wall_times_remaining[-1]

        def do_optimization_step(
            params,
            optimizer_state,
            prev_loss,
            iteration,
            rng_key,
            ascent_count,
            descent_count,
        ):
            """Single optimization step with SA-GD logic."""
            loss, grads = self._grad_fn(params)

            # Compute loss difference (ΔE)
            delta_e = abs(float(loss) - prev_loss)

            # Compute current learning rate (with decay if applicable)
            current_lr = learning_rate * (lr_decay**iteration)

            # Compute transition probability
            trans_prob = self._compute_transition_probability(
                delta_e=delta_e,
                epoch=iteration,
                T0=T0,
                learning_rate=current_lr,
                use_double_annealing=use_double_annealing,
                lr_decay=lr_decay,
                initial_lr=learning_rate,
            )

            # Probability of gradient ascent = 1 - trans_prob
            # But we cap it at max_ascent_prob
            ascent_prob = min(1.0 - trans_prob, max_ascent_prob)

            # Sample random value to decide descent vs ascent
            rng_key, subkey = jax.random.split(rng_key)
            random_val = float(jax.random.uniform(subkey))

            # Compute updates from optimizer
            updates, new_optimizer_state = optimizer.update(
                grads, optimizer_state, params
            )

            if random_val < ascent_prob:
                # Gradient ASCENT: go uphill
                # Negate the updates and scale by sigma
                updates = jax.tree.map(lambda x: -sigma * x, updates)
                ascent_count += 1
            else:
                # Normal gradient DESCENT
                descent_count += 1

            new_params = optax.apply_updates(params, updates)

            return (
                new_params,
                new_optimizer_state,
                float(loss),
                rng_key,
                ascent_count,
                descent_count,
            )

        # Main optimization loop
        if wall_times is not None:
            # Wall-time constrained: ignore max_iterations and patience
            start_time = time.time()
            i = 0
            while (time.time() - start_time) < max_wall_time:
                elapsed = time.time() - start_time

                # Record iteration index at wall_times checkpoints
                while wall_times_remaining and elapsed >= wall_times_remaining[0]:
                    wall_time_indices.append(i)
                    wall_times_remaining.popleft()

                params, optimizer_state, loss, rng_key, ascent_count, descent_count = (
                    do_optimization_step(
                        params,
                        optimizer_state,
                        prev_loss,
                        i,
                        rng_key,
                        ascent_count,
                        descent_count,
                    )
                )

                if i % 500 == 0:
                    ascent_pct = (
                        100 * ascent_count / max(ascent_count + descent_count, 1)
                    )
                    print(
                        f"Iteration {i}: Loss = {loss:.6f}, Ascent% = {ascent_pct:.1f}%"
                    )

                if loss < best_loss - 1e-4:
                    best_loss, best_params = loss, params
                    # print(f"Iteration {i}: New best loss = {loss}")

                if return_best_params_history:
                    best_params_history.append(best_params)

                losses.append(loss)
                prev_loss = loss
                i += 1

            # Fill remaining wall_times that weren't reached with final iteration
            while wall_times_remaining:
                wall_time_indices.append(i - 1 if i > 0 else 0)
                wall_times_remaining.popleft()
        else:
            # Iteration/patience constrained
            no_improve_count = 0
            for i in range(max_iterations):
                params, optimizer_state, loss, rng_key, ascent_count, descent_count = (
                    do_optimization_step(
                        params,
                        optimizer_state,
                        prev_loss,
                        i,
                        rng_key,
                        ascent_count,
                        descent_count,
                    )
                )

                if i % 100 == 0:
                    ascent_pct = (
                        100 * ascent_count / max(ascent_count + descent_count, 1)
                    )
                    print(
                        f"Iteration {i}: Loss = {loss:.6f}, Ascent% = {ascent_pct:.1f}%"
                    )

                if loss < best_loss - 1e-4:
                    best_loss, best_params, no_improve_count = loss, params, 0
                    print(f"Iteration {i}: New best loss = {loss}")
                else:
                    no_improve_count += 1

                if return_best_params_history:
                    best_params_history.append(best_params)

                losses.append(loss)
                prev_loss = loss

                if no_improve_count > patience:
                    break

        # Final statistics
        total_steps = ascent_count + descent_count
        if total_steps > 0:
            print("\nSA-GD Statistics:")
            print(f"  Total steps: {total_steps}")
            print(
                f"  Ascent steps: {ascent_count} ({100 * ascent_count / total_steps:.1f}%)"
            )
            print(
                f"  Descent steps: {descent_count} ({100 * descent_count / total_steps:.1f}%)"
            )

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
                hyper_param_str=f"lr{learning_rate}_T{T0}_sigma{sigma}",
            )

        return best_params_bounded, best_params_history_bounded, losses, wall_time_indices
