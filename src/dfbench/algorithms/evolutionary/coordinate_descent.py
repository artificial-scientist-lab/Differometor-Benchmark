"""Cyclic coordinate descent with multi-scale grid search."""

import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float
from wandb import Run

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective


class CoordinateDescent(OptimizationAlgorithm):
    """Cyclic coordinate descent with discretised line search.

    Cycles through each parameter in turn, performing a 1D grid search
    over a window centred on the current value. The window starts large
    and shrinks geometrically each round, giving a coarse-to-fine search.

    This is a standard derivative-free optimisation method (see Nocedal &
    Wright, *Numerical Optimization*; Wright, *Coordinate Descent
    Algorithms*). The grid-based line search is a practical discretisation
    of exact coordinate-wise minimisation.

    Attributes:
        algorithm_str: Identifier string ("coordinate_descent").
        algorithm_type: Algorithm classification (EVOLUTIONARY).
        n_sweep: Number of grid points per 1D line search.
        initial_window: Initial sweep half-width as a fraction of each
            parameter's range.
        window_shrink: Factor by which the window shrinks each round.
        min_window: Minimum window fraction (stops shrinking below this).
    """

    algorithm_str: str = "coordinate_descent"
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY

    def __init__(
        self,
        n_sweep: int = 51,
        initial_window: float = 0.01,
        window_shrink: float = 0.9,
        min_window: float = 1e-6,
    ) -> None:
        """Initialise coordinate descent.

        Args:
            n_sweep: Number of grid points per 1D line search.
            initial_window: Initial sweep half-width as a fraction of each
                parameter's range (e.g. 0.01 = 1% of range each side).
            window_shrink: Multiplicative factor applied to the window each
                round (e.g. 0.5 = halve each round).
            min_window: Minimum window fraction; the window will not shrink
                below this value.
        """
        self.n_sweep = n_sweep
        self.initial_window = initial_window
        self.window_shrink = window_shrink
        self.min_window = min_window

    def optimize(
        self,
        problem_objective: Objective,
        rounds: int,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        save_path: Path | None = None,
        wandb_run: Run | None = None,
    ) -> None:
        """Run cyclic coordinate descent.

        Args:
            problem_objective: Pre-configured Objective instance.
            rounds: Maximum number of coordinate sweep rounds.
            init_params: Initial bounded parameters. If None, sampled
                uniformly at random within bounds.
            random_seed: Random seed for reproducibility.
            save_path: If provided, save best params to this path on
                completion.
            wandb_run: Optional Weights & Biases run for logging.
        """
        obj = problem_objective
        problem = obj.problem
        bounds = problem.bounds
        lower, upper = bounds[0], bounds[1]
        n_params = problem.n_params

        _, key = self.prepare(obj, unbounded=False, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_bounded(rng_key=key)
        else:
            params = init_params

        # Identify free parameters (not stuck at bounds)
        free_indices = [
            i
            for i in range(n_params)
            if float(params[i]) > float(lower[i]) and float(params[i]) < float(upper[i])
        ]
        logging.info(
            f"Coordinate descent: {len(free_indices)}/{n_params} free parameters, "
            f"n_sweep={self.n_sweep}, initial_window={self.initial_window}"
        )

        # JIT warmup
        _ = obj.value(params)

        obj.start_logging()

        current_loss = float(obj.value(params))
        logging.info(f"Starting loss: {current_loss:.8f}\n")

        if wandb_run is not None:
            wandb_run.log({"loss": current_loss, "round": 0})

        round_idx = 0
        while round_idx < rounds:
            window_frac = max(
                self.min_window,
                self.initial_window * (self.window_shrink**round_idx),
            )
            improved_this_round = False

            for param_idx in free_indices:
                comp, prop = problem.optimization_pairs[param_idx]
                center = float(params[param_idx])
                lo = float(lower[param_idx])
                hi = float(upper[param_idx])
                window = window_frac * (hi - lo)

                sweep_lo = max(lo, center - window)
                sweep_hi = min(hi, center + window)
                sweep_vals = jnp.linspace(sweep_lo, sweep_hi, self.n_sweep)

                best_val = center
                best_l = current_loss
                for v in sweep_vals:
                    candidate = params.at[param_idx].set(v)
                    l = float(obj.value(candidate))
                    if np.isfinite(l) and l < best_l:
                        best_l = l
                        best_val = float(v)

                if best_l < current_loss - 1e-10:
                    logging.info(
                        f"  Round {round_idx + 1}: {comp}.{prop}: "
                        f"{center:.6f} -> {best_val:.6f}, "
                        f"loss: {current_loss:.8f} -> {best_l:.8f}"
                    )
                    params = params.at[param_idx].set(best_val)
                    current_loss = best_l
                    improved_this_round = True

            if wandb_run is not None:
                wandb_run.log({"loss": current_loss, "round": round_idx + 1})

            logging.info(
                f"  Round {round_idx + 1} complete. Best loss: {current_loss:.8f} "
                f"(window_frac={window_frac:.6f})\n"
            )

            if not improved_this_round:
                if window_frac <= self.min_window:
                    logging.info("  No improvement at minimum window — converged.")
                    break

            round_idx += 1

        logging.info(f"\nFinal best loss: {current_loss:.8f}")

        if save_path is not None:
            np.savez_compressed(
                save_path,
                best_params=np.array(params),
                best_loss=current_loss,
            )
