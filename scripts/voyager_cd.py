"""Single CoordinateDescent run from Voyager parameters."""

import argparse
import logging
from pathlib import Path

import jax.numpy as jnp
import wandb
from differometor.setups import voyager

from dfbench.algorithms import CoordinateDescent
from dfbench.core.objective import Objective
from dfbench.problems import (
    ConstrainedVoyagerProblem,
    RandomUIFOProblem,
    VoyagerProblem,
)


def main():
    parser = argparse.ArgumentParser(description="Run CoordinateDescent optimizer.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--rounds", type=int, default=50, help="Number of CD rounds per run"
    )
    parser.add_argument(
        "--n-sweep", type=int, default=51, help="Grid points per 1D line search"
    )
    parser.add_argument(
        "--initial-window",
        type=float,
        default=0.01,
        help="Initial sweep window as fraction of parameter range",
    )
    parser.add_argument(
        "--window-shrink",
        type=float,
        default=0.9,
        help="Window shrink factor per round",
    )
    parser.add_argument(
        "--min-window",
        type=float,
        default=1e-6,
        help="Minimum window fraction (stops shrinking below this)",
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="constrained_voyager",
        help="Type of problem to solve",
    )
    parser.add_argument(
        "--log-wandb", action="store_true", help="Log results to Weights & Biases"
    )
    args = parser.parse_args()

    results_root = Path(f"results/cd/{args.problem}")
    results_root.mkdir(parents=True, exist_ok=True)

    exp_name = (
        f"cd_{args.problem}_seed_{args.random_seed}_rounds_{args.rounds}"
        f"_window_{args.initial_window}_shrink_{args.window_shrink}"
        f"_min_window_{args.min_window}_sweep_{args.n_sweep}"
    )
    save_path = results_root / exp_name

    logging.basicConfig(
        filename=results_root / f"{exp_name}.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    if save_path.with_suffix(".npz").exists():
        raise FileExistsError(
            f"Results already exist at {save_path.with_suffix('.npz')}"
        )

    if args.problem == "voyager":
        problem = VoyagerProblem()
    elif args.problem == "constrained_voyager":
        problem = ConstrainedVoyagerProblem()
    elif args.problem == "random_uifo":
        problem = RandomUIFOProblem()
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

    logger.info("Starting run (%s)", exp_name)

    wandb_run = None
    if args.log_wandb:
        wandb_run = wandb.init(
            project="differometor",
            name=exp_name,
            config=vars(args),
        )

    # Extract Voyager parameter values as init_params
    S, _ = voyager()
    init_params = []
    for component_name, property_name in problem.optimization_pairs:
        if "_" not in component_name:
            value = S.nodes[component_name]["properties"][property_name]
        else:
            value = S.edges[component_name]["properties"][property_name]
        init_params.append(value)
    init_params = jnp.array(init_params)

    obj = Objective(
        problem,
        unbounded=False,
        verbose=0,
        save_params_history=False,
        save_time_steps=False,
    )
    optimizer = CoordinateDescent(
        n_sweep=args.n_sweep,
        initial_window=args.initial_window,
        window_shrink=args.window_shrink,
        min_window=args.min_window,
    )

    optimizer.optimize(
        obj,
        rounds=args.rounds,
        init_params=init_params,
        random_seed=args.random_seed,
        save_path=save_path,
        wandb_run=wandb_run,
    )

    logger.info(
        "Done — best loss: %.6f, evals: %d",
        obj.best_loss,
        obj.eval_count,
    )

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
