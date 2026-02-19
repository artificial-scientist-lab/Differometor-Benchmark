"""
Simulated annealing gradient descent based on https://arxiv.org/abs/2107.07558
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.objective import Objective


class SAGD(OptimizationAlgorithm):
    """Simulated Annealing Gradient Descent (SA-GD) optimization algorithm.

    Implements the SA-GD algorithm from the paper:
    "SA-GD: Improved Gradient Descent Learning Strategy with Simulated Annealing"
    (arXiv:2107.07558, Cai 2021)

    The algorithm introduces simulated annealing to gradient descent, giving the
    optimizer a probabilistic "hill-mounting" ability to escape local minima and
    saddle points. With a certain probability (based on temperature and loss
    difference), the algorithm performs gradient ASCENT instead of descent.

    All history tracking, printing, and checkpointing is handled by the
    `Objective` wrapper. The algorithm loop is minimal.

    Key equations:
    - Transition probability: P_i = exp(-|ΔE|^k / (T_0 * ε * ln(n+1)))
    - With probability P_i: perform gradient descent (normal)
    - With probability 1-P_i: perform gradient ascent (uphill)

    The probability of going uphill starts low and increases over iterations,
    but stays below a ceiling (default 33%) to ensure convergence.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("sa_gd").
        algorithm_type (AlgorithmType): Type classification (GRADIENT_BASED).
    """

    algorithm_str: str = "sa_gd"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        """Initialize SA-GD optimizer."""
        pass

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

    def optimize(
        self,
        problem_objective: Objective,
        max_iterations: int | None = None,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.1,
        patience: int = 1000,
        T0: float = 15.0,
        sigma: float = 1.0,
        max_ascent_prob: float = 0.33,
        use_double_annealing: bool = False,
        lr_decay: float = 1.0,
        **adam_kwargs,
    ) -> Objective:
        """Run SA-GD (Simulated Annealing Gradient Descent) optimization.

        This algorithm combines gradient descent with simulated annealing concepts.
        It probabilistically performs gradient ascent to escape local minima.

        Args:
            problem_objective: The Objective instance wrapping the problem.
            max_iterations: Maximum number of gradient steps. If None, runs until budget exceeded.
            init_params: Initial parameters. If None, initialized via
                obj.random_params_unbounded(). Defaults to None.
            random_seed: Random seed for reproducibility. If None,
                uses system entropy. Defaults to None.
            learning_rate: Learning rate for Adam optimizer. Defaults to 0.1.
            patience: Stop if no improvement for this many iterations. Defaults to 1,000.
            T0: Initial temperature for simulated annealing. Higher values
                lead to higher probability of gradient ascent. Defaults to 15.0.
            sigma: Expansion factor for gradient ascent step size. Defaults to 1.0.
            max_ascent_prob: Maximum probability of performing gradient ascent.
                Paper recommends keeping this below 0.33 for convergence. Defaults to 0.33.
            use_double_annealing: Whether to use the double simulated annealing
                formula designed for exponentially decaying learning rates. Defaults to False.
            lr_decay: Learning rate decay factor per iteration. Defaults to 1.0.
            **adam_kwargs: Additional keyword arguments passed to optax.adam().

        Returns:
            The Objective instance with all logged data.
        """
        obj = problem_objective
        problem = obj.problem

        random_seed, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_unbounded()
        else:
            params = init_params

        # Create optimizer with gradient clipping
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        optimizer_state = optimizer.init(params)

        # Warm-up JIT
        _ = obj.value_and_grad(params)

        obj.start_logging()

        prev_loss = 0.0  # Initial previous loss (E_0 = 0 as per paper)
        ascent_count = 0
        descent_count = 0
        iteration = 0

        while not obj.budget_exceeded:
            if max_iterations is not None and iteration >= max_iterations:
                break
                
            loss, grads = obj.value_and_grad(params)

            # Early stopping: patience check
            if obj.evals_since_improvement > patience:
                break

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
            updates, optimizer_state = optimizer.update(
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

            params = optax.apply_updates(params, updates)
            prev_loss = float(loss)
            iteration += 1

        return obj
