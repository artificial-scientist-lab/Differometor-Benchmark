"""Script for Batched Adam optimizer (multiple parallel trajectories)."""

import argparse
import logging
from pathlib import Path

import wandb

from dfbench.algorithms import BatchedAdamGD
from dfbench.problems import (
    ConstrainedVoyagerProblem,
    RandomUIFOProblem,
    VoyagerProblem,
)


def main():
    parser = argparse.ArgumentParser(description="Run Batched Adam optimizer.")
    parser.add_argument(
        "--max-time",
        type=int,
        default=1200,
        help="Maximum optimization time in seconds",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of parallel trajectories",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate for Adam"
    )
    parser.add_argument(
        "--patience", type=int, default=2000, help="Early stopping patience (in evals)"
    )
    parser.add_argument(
        "--problem", type=str, default="voyager", help="Type of problem to solve"
    )
    parser.add_argument(
        "--loss-transform",
        type=str,
        default=None,
        choices=["arcsinh"],
        help="Monotonic transform applied to loss for optimization (metrics stay in original space)",
    )
    parser.add_argument(
        "--log-wandb", action="store_true", help="Log results to Weights & Biases"
    )
    args = parser.parse_args()

    transform_str = f"_transform_{args.loss_transform}" if args.loss_transform else ""
    exp_name = (
        f"batched_adam_{args.problem}_seed_{args.random_seed}"
        f"_bs_{args.batch_size}_lr_{args.learning_rate}"
        f"_patience_{args.patience}_max_time_{args.max_time}{transform_str}"
    )

    wandb_run = None
    if args.log_wandb:
        wandb_run = wandb.init(project="differometor", name=exp_name, config=vars(args))
    results_root = Path(f"results/batched_adam/{args.problem}")
    results_root.mkdir(parents=True, exist_ok=True)
    save_path = results_root / exp_name
    if save_path.exists():
        raise FileExistsError(
            f"Results path already exists: {save_path}. "
            "Aborting to avoid overwriting existing experiment results."
        )

    logging.basicConfig(
        filename=results_root / f"{exp_name}.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting experiment with args: %s", vars(args))

    if args.problem == "voyager":
        problem = VoyagerProblem()
    elif args.problem == "constrained_voyager":
        problem = ConstrainedVoyagerProblem()
    elif args.problem == "random_uifo":
        problem = RandomUIFOProblem()
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

    optimizer = BatchedAdamGD(problem, verbose=1)

    # Run optimization
    obj = optimizer.optimize(
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        max_time=args.max_time,
        learning_rate=args.learning_rate,
        patience=args.patience,
        plot_loss=True,
        save_run_to_file=True,
        save_path=save_path,
        print_every=100,
        wandb_run=wandb_run,
        loss_transform=args.loss_transform,
    )

    logging.info(f"Best loss: {obj.best_loss:.6f}")
    logging.info(f"Total evaluations: {obj.eval_count}")


if __name__ == "__main__":
    main()
