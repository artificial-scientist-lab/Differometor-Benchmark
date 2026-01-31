import logging
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from differometor.utils import sigmoid_bounding
from jaxtyping import Array, Float
from wandb import Run

from dfbench.core.objective import Objective
from dfbench.core.protocols import (
    AlgorithmType,
    ContinuousProblem,
    OptimizationAlgorithm,
)
from dfbench.core.utils import inverse_sigmoid_bounding


class BatchedAdamGD(OptimizationAlgorithm):
    """Batched Adam optimizer running multiple trajectories in parallel via jax.vmap.

    Each trajectory has independent parameters and optimizer state, but the
    forward/backward passes are batched on GPU for efficiency. Per-trajectory
    best losses and parameters are tracked on-device.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("batched_adam_gd").
        algorithm_type (AlgorithmType): Type classification (GRADIENT_BASED).
        _problem (ContinuousProblem): The optimization problem instance.
    """

    algorithm_str: str = "batched_adam_gd"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(
        self,
        problem: ContinuousProblem,
        verbose: int = 0,
        save_params_history: bool = False,
    ) -> None:
        """Initialize Batched Adam optimizer.

        Args:
            problem: The continuous optimization problem to solve.
            verbose: Verbosity level (0=silent, 1+=prints). Defaults to 0.
            save_params_history: Whether to save parameter history. Defaults to
                False since each step stores a representative from the batch.
        """
        self._problem = problem
        self._verbose = verbose
        self._save_params_history = save_params_history

    def optimize(
        self,
        random_seed: int,
        batch_size: int = 64,
        init_params: Float[Array, "batch n_params"] | None = None,
        max_time: float | None = None,
        learning_rate: float = 0.1,
        max_iterations: int | None = None,
        patience: int = 2000,
        print_every: int = 100,
        plot_loss: bool = False,
        save_run_to_file: bool = False,
        save_path: Path | None = None,
        wandb_run: Run | None = None,
        loss_transform: str | None = None,
        **adam_kwargs,
    ) -> Objective:
        """Run batched Adam using `Objective` for logging.

        Args:
            random_seed: Seed for generating batch of starting points.
            batch_size: Number of parallel trajectories. Ignored if init_params
                is provided (batch_size is inferred from its shape).
            init_params: Initial parameters of shape (batch_size, n_params) in
                bounded space. If None, randomly sampled within bounds.
            max_time: Time budget in seconds. None for unlimited.
            learning_rate: Adam learning rate.
            max_iterations: Max iterations (each iteration = batch_size evals).
                None for time-limited only.
            patience: Stop after this many gradient steps without improvement.
            print_every: Print summary every N evaluations.
            plot_loss: If True, call problem.output_to_files for plotting.
            save_run_to_file: If True, call obj.save_run_data for checkpointing.
            save_path: File path to save run info to. If provided, saves history,
                best params, and per-trajectory bests.
            wandb_run: Optional Weights & Biases run for logging.
            loss_transform: Monotonic transform applied to the objective for
                optimization. "arcsinh" compresses the dynamic range. Metrics
                are always reported in the original (untransformed) space.
            **adam_kwargs: Passed to optax.adam().

        Returns:
            The Objective instance with all logged data.
        """
        if max_iterations is None and max_time is None:
            raise ValueError("Either max_iterations or max_time must be specified.")

        if loss_transform is not None and loss_transform != "arcsinh":
            raise ValueError(
                f"Unsupported loss_transform: {loss_transform!r}. Use 'arcsinh' or None."
            )

        if wandb_run is not None:
            wandb_run.define_metric("time_elapsed")
            wandb_run.define_metric("best_loss_vs_time", step_metric="time_elapsed")

        n_params = self._problem.n_params
        bounds = self._problem.bounds

        # Initialize batch of starting points
        if init_params is not None:
            batch_constrained = jnp.asarray(init_params)
            batch_size = batch_constrained.shape[0]
        else:
            keys = jr.split(jr.PRNGKey(random_seed), batch_size)
            batch_constrained = jax.vmap(
                lambda key: jr.uniform(
                    key, shape=(n_params,), minval=bounds[0], maxval=bounds[1]
                )
            )(keys)

        batch_unconstrained = jax.vmap(lambda p: inverse_sigmoid_bounding(p, bounds))(
            batch_constrained
        )

        # Per-trajectory best tracking (on device)
        batch_best_losses = jnp.full((batch_size,), jnp.inf)
        batch_best_params = batch_unconstrained.copy()

        # Max evals accounts for batch_size per step
        max_evals = max_iterations * batch_size if max_iterations is not None else None

        obj = Objective(
            self._problem,
            unbounded=True,
            max_time=max_time,
            max_evals=max_evals,
            save_params_history=self._save_params_history,
            print_every=print_every,
            verbose=self._verbose,
            algorithm_str=self.algorithm_str,
        )

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        batch_opt_states = jax.vmap(optimizer.init)(batch_unconstrained)

        # Set up objective function (with optional transform)
        raw_value_fn = self._problem.sigmoid_objective_function
        if loss_transform == "arcsinh":

            def opt_value_fn(params):
                return jnp.arcsinh(raw_value_fn(params))
        else:
            opt_value_fn = raw_value_fn

        opt_value_and_grad_fn = jax.value_and_grad(opt_value_fn)

        @jax.jit
        def _batch_step(
            batch_params, batch_opt_states, batch_best_losses, batch_best_params
        ):
            opt_losses, grads = jax.vmap(opt_value_and_grad_fn)(batch_params)

            def _single_update(g, s, p):
                updates, new_s = optimizer.update(g, s, p)
                new_p = optax.apply_updates(p, updates)
                return new_p, new_s

            new_params, new_states = jax.vmap(_single_update)(
                grads, batch_opt_states, batch_params
            )

            raw_losses = (
                jnp.sinh(opt_losses) if loss_transform == "arcsinh" else opt_losses
            )

            # Update per-trajectory bests
            improved = raw_losses < batch_best_losses
            new_best_losses = jnp.where(improved, raw_losses, batch_best_losses)
            new_best_params = jnp.where(
                improved[:, None], new_params, batch_best_params
            )

            return new_params, new_states, raw_losses, new_best_losses, new_best_params

        if self._verbose >= 1:
            print(f"Warming up JIT compilation (batch_size={batch_size})...")
        _ = _batch_step(
            batch_unconstrained, batch_opt_states, batch_best_losses, batch_best_params
        )

        obj.start_logging()

        i = 1
        best_loss = jnp.inf
        best_unconstrained_params = batch_unconstrained[0]
        steps_since_improvement = 0

        while not obj.budget_exceeded:
            iter_start_time = time.time()

            (
                batch_unconstrained,
                batch_opt_states,
                batch_losses,
                batch_best_losses,
                batch_best_params,
            ) = _batch_step(
                batch_unconstrained,
                batch_opt_states,
                batch_best_losses,
                batch_best_params,
            )

            obj._log_time()
            obj._log_evals(batch_unconstrained, batch_losses, None)
            obj._log_to_file()

            min_loss = jnp.min(batch_losses)
            if min_loss < best_loss:
                best_loss = min_loss
                best_idx = jnp.argmin(batch_losses)
                best_unconstrained_params = batch_unconstrained[best_idx]
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1

            # Early stopping
            if steps_since_improvement > patience:
                logging.info(
                    f"Early stopping at iteration {i} after {obj.eval_count} evaluations. "
                    f"Best loss: {best_loss:.6f}"
                )
                break

            iter_time = time.time() - iter_start_time
            logging.info(
                f"Completed iteration: {i} ({batch_size} evals) in {iter_time:.4f}s, "
                f"best loss: {best_loss:.6f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "batch_min_loss": float(min_loss),
                        "best_loss": float(best_loss),
                        "best_loss_vs_time": float(best_loss),
                        "iter_time": iter_time,
                        "time_elapsed": obj.time_elapsed,
                        "evals_since_improvement": obj.evals_since_improvement,
                    },
                    step=i,
                )

            i += 1

        best_constrained_params = sigmoid_bounding(
            best_unconstrained_params, self._problem.bounds
        )
        batch_best_constrained_params = jax.vmap(
            lambda p: sigmoid_bounding(p, self._problem.bounds)
        )(batch_best_params)

        logging.info(
            f"Optimization finished. Best loss: {best_loss:.6f}, "
            f"total evaluations: {obj.eval_count}."
        )

        # Outputs
        if plot_loss:
            obj.output_to_files()
        if save_run_to_file:
            obj.save_run_data()

        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "final/best_loss": float(best_loss),
                    "final/total_evals": obj.eval_count,
                    "final/total_time": obj.time_elapsed,
                    "final/improvement_count": obj.improvement_count,
                },
            )

        if save_path is not None:
            save_path = Path(str(save_path) + ".npz")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                save_path,
                loss_history=np.array(obj.loss_history, dtype=object),
                time_steps=np.array(obj.time_steps),
                best_constrained_params=np.array(best_constrained_params),
                best_unconstrained_params=np.array(best_unconstrained_params),
                best_loss=np.array(best_loss),
                eval_count=obj.eval_count,
                batch_size=batch_size,
                batch_best_losses=np.array(batch_best_losses),
                batch_best_constrained_params=np.array(batch_best_constrained_params),
                batch_best_unconstrained_params=np.array(batch_best_params),
            )
            logging.info(f"Run info saved to {save_path}")

        return obj
