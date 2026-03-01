"""Repeated AdamGD runs with incrementing seeds, for a fixed time budget."""

import argparse
import logging
from pathlib import Path

import wandb

from dfbench.algorithms import AdamGD
from dfbench.core.objective import Objective
from dfbench.problems import (
    ConstrainedVoyagerProblem,
    RandomUIFOProblem,
    VoyagerProblem,
)


def main():
    parser = argparse.ArgumentParser(description="Run AdamGD optimizer.")
    parser.add_argument(
        "--max-evals",
        type=int,
        default=20000,
        help="Maximum number of evaluations per run",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Starting random seed"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="Learning rate for Adam"
    )
    parser.add_argument(
        "--problem",
        type=str,
        default="constrained_voyager",
        help="Type of problem to solve",
    )
    parser.add_argument(
        "--use-arcsinh-transform",
        action="store_true",
        help="Use arcsinh transform on loss for optimization",
    )
    parser.add_argument(
        "--log-wandb", action="store_true", help="Log results to Weights & Biases"
    )
    args = parser.parse_args()

    results_root = Path(f"results/adam/{args.problem}")
    results_root.mkdir(parents=True, exist_ok=True)
    transform_str = "_arcsinh_transform" if args.use_arcsinh_transform else ""

    logging.basicConfig(
        filename=results_root
        / f"adam_{args.problem}_lr_{args.learning_rate}{transform_str}.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    if args.problem == "voyager":
        make_problem = VoyagerProblem
    elif args.problem == "constrained_voyager":
        make_problem = ConstrainedVoyagerProblem
    elif args.problem == "random_uifo":
        make_problem = RandomUIFOProblem
    else:
        raise ValueError(f"Unknown problem type: {args.problem}")

    seed = args.random_seed
    best_overall_loss = float("inf")
    best_overall_seed = None

    while True:
        exp_name = f"adam_{args.problem}_seed_{seed}_lr_{args.learning_rate}_max_evals_{args.max_evals}{transform_str}"
        save_path = results_root / exp_name

        if save_path.with_suffix(".npz").exists():
            logger.info(
                "Skipping seed %d — results already exist at %s", seed, save_path
            )
            seed += 1
            continue

        logger.info("Starting run with seed %d (%s)", seed, exp_name)

        wandb_run = None
        if args.log_wandb:
            wandb_run = wandb.init(
                project="differometor",
                name=exp_name,
                config={**vars(args), "random_seed": seed},
                reinit=True,
            )

        obj = Objective(
            make_problem(),
            unbounded=True,
            max_evals=args.max_evals,
            verbose=0,
            save_params_history=False,
            save_time_steps=False,
        )
        optimizer = AdamGD()

        optimizer.optimize(
            obj,
            random_seed=seed,
            learning_rate=args.learning_rate,
            patience=args.max_evals,  # No early stopping, run until max_evals
            save_path=save_path,
            wandb_run=wandb_run,
            use_arcsinh_transform=args.use_arcsinh_transform,
        )

        if obj.best_loss is not None and obj.best_loss < best_overall_loss:
            best_overall_loss = obj.best_loss
            best_overall_seed = seed

        logger.info(
            "Seed %d done — best loss: %.6f, evals: %d | overall best: %.6f (seed %d)",
            seed,
            obj.best_loss,
            obj.eval_count,
            best_overall_loss,
            best_overall_seed,
        )

        if wandb_run is not None:
            wandb_run.finish()

        seed += 1


if __name__ == "__main__":
    main()
