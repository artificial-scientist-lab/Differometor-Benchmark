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


class LBFGS(OptimizationAlgorithm):
    """L-BFGS optimization algorithm.

    Implements gradient-based optimization using the L-BFGS optimizer from Optax.
    Includes early stopping based on patience.

    All history tracking, printing, and checkpointing is handled by the
    `Objective` wrapper. The algorithm loop is minimal.

    Attributes:
        algorithm_str (str): Identifier string for this algorithm ("lbfgs").
        algorithm_type (AlgorithmType): Type classification (GRADIENT_BASED).
        _problem (ContinuousProblem): The optimization problem instance.

    Note:
        This algorithm uses `problem.sigmoid_objective_function` which expects
        unbounded parameters. The sigmoid bounding is applied internally by the
        objective function, allowing the optimizer to search in (-∞, +∞) space.

    Example:
        >>> problem = VoyagerProblem()
        >>> optimizer = LBFGS(problem)
        >>> objective = optimizer.optimize(
        ...     learning_rate=0.1,
        ...     max_iterations=10000,
        ...     patience=500,
        ... )
    """

    algorithm_str: str = "lbfgs"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(
        self,
        problem: ContinuousProblem,
        verbose: int = 0,
        save_params_history: bool = True,
    ) -> None:
        """Initialize L-BFGS optimizer.
        Args:
            problem (ContinuousProblem): The continuous optimization problem to solve.
            verbose (int): Verbosity level (0=silent, 1+=prints). Defaults to 0.
            save_params_history: Whether to save parameter history. Defaults to True.
        """
        self._problem = problem
        self._verbose = verbose
        self._save_params_history = save_params_history

    def optimize(
        self,
        random_seed: int,
        init_params: Float[Array, "{self._problem.n_params}"] | None = None,
        max_time: float | None = None,
        max_iterations: int | None = None,
        patience: int = 1000,
        print_every: int = 100,
        plot_loss: bool = False,
        save_run_to_file: bool = False,
        save_path: Path | None = None,
        wandb_run: Run | None = None,
        loss_transform: str | None = None,
        **lbfgs_kwargs,
    ) -> Objective:
        """Run L-BFGS using `Objective` for logging.
        Args:
            random_seed: Seed for init param generation.
            init_params: Initial parameters. If None, random in [-10, 10].
            max_time: Time budget in seconds. None for unlimited.
            max_iterations: Max iterations. None for time-limited only.
            patience: Stop after this many iterations without improvement.
            print_every: Print summary every N evaluations.
            plot_loss: If True, call problem.output_to_files for plotting.
            save_run_to_file: If True, call obj.save_run_data for checkpointing.
            save_path: File path to save run info to. If provided, saves history
                and best params. The caller is responsible for naming.
            wandb_run: Optional Weights & Biases run for logging.
            loss_transform: Monotonic transform applied to the objective for
                optimization. "arcsinh" compresses the dynamic range, improving
                LBFGS curvature estimates on high-range landscapes. Metrics are
                always reported in the original (untransformed) space.
            **lbfgs_kwargs: Passed to optax.lbfgs().

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

        constrained_params = (
            jr.uniform(
                jr.PRNGKey(random_seed),
                shape=(self._problem.n_params,),
                minval=self._problem.bounds[0],
                maxval=self._problem.bounds[1],
            )
            if init_params is None
            else init_params
        )
        unconstrained_params = inverse_sigmoid_bounding(
            constrained_params, self._problem.bounds
        )
        best_unconstrained_params = unconstrained_params

        obj = Objective(
            self._problem,
            unbounded=True,
            max_time=max_time,
            max_evals=max_iterations,
            save_params_history=self._save_params_history,
            print_every=print_every,
            verbose=self._verbose,
            algorithm_str=self.algorithm_str,
        )

        optimizer = optax.lbfgs(**lbfgs_kwargs)
        optimizer_state = optimizer.init(unconstrained_params)

        raw_value_fn = self._problem.sigmoid_objective_function

        if loss_transform == "arcsinh":
            def opt_value_fn(params):
                return jnp.arcsinh(raw_value_fn(params))
        else:
            opt_value_fn = raw_value_fn

        opt_value_and_grad_fn = jax.value_and_grad(opt_value_fn)

        @jax.jit
        def _step(params, opt_state):
            opt_loss, grads = opt_value_and_grad_fn(params)
            updates, new_opt_state = optimizer.update(
                grads,
                opt_state,
                params,
                value=opt_loss,
                grad=grads,
                value_fn=opt_value_fn,
            )
            new_params = optax.apply_updates(params, updates)
            # Recover original loss for logging (sinh inverts arcsinh exactly)
            raw_loss = jnp.sinh(opt_loss) if loss_transform == "arcsinh" else opt_loss
            return jnp.asarray(new_params), new_opt_state, raw_loss, grads

        if self._verbose >= 1:
            print("Warming up JIT compilation...")
        _ = _step(unconstrained_params, optimizer_state)  # Warm-up JIT

        obj.start_logging()

        i = 1
        best_loss = jnp.inf
        while not obj.budget_exceeded:
            iter_start_time = time.time()
            unconstrained_params, optimizer_state, loss, grads = _step(
                unconstrained_params, optimizer_state
            )
            obj._log_time()
            obj._log_evals(unconstrained_params, loss, grads)
            obj._log_to_file()

            if loss < best_loss:
                best_loss = loss
                best_unconstrained_params = unconstrained_params

            # Early stopping: patience check using Objective's improvement tracker
            if obj.evals_since_improvement > patience:
                logging.info(
                    f"Early stopping at iteration {i} after {obj.eval_count} evaluations. Best loss: {best_loss:.6f}"
                )
                break

            iter_time = time.time() - iter_start_time
            logging.info(
                f"Completed iteration: {i} in {iter_time:.4f} seconds, best loss: {best_loss:.6f}"
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "loss": float(loss),
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
        logging.info(
            f"Optimization finished. Best loss: {best_loss:.6f}, total evaluations: {obj.eval_count}."
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
                unconstrained_params_history=np.array(obj.params_history, dtype=object),
                time_steps=np.array(obj.time_steps),
                best_constrained_params=np.array(best_constrained_params),
                best_unconstrained_params=np.array(best_unconstrained_params),
                best_loss=np.array(best_loss),
                eval_count=obj.eval_count,
            )
            logging.info(f"Run info saved to {save_path}")

        return obj
