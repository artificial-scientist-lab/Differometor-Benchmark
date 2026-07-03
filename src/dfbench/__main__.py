"""Script to allow running the benchmark from the command line."""
# TODO: Configs using YAML

from __future__ import annotations

import argparse
from pathlib import Path

from dfbench.benchmark import Benchmark, AlgorithmConfig
from dfbench.algorithms import algorithms
from dfbench.core.constants import DATA_DIR
from dfbench.problems import problems


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the benchmark.")
    parser.add_argument(
        "--experiment_name", "-exp", type=str, help="Name of the experiment."
    )
    parser.add_argument(
        "--optimizers",
        "-o",
        nargs="+",
        type=str,
        required=True,
        help="List of optimizers to run.",
        choices=list(algorithms.keys()),
    )
    parser.add_argument(
        "problem",
        type=str,
        help="Name of the problem to run.",
        choices=list(problems.keys()),
    )
    parser.add_argument(
        "--seed", "-s", type=int, default=0, help="Master seed to use for the runs."
    )
    parser.add_argument(
        "--n_runs",
        "-n",
        type=int,
        default=1,
        help="Number of seeds to run for each algorithm.",
    )
    parser.add_argument(
        "--max_evals",
        "-e",
        type=int,
        default=100,
        help="Maximum number of evaluations to run.",
    )
    parser.add_argument(
        "--max_time",
        "-t",
        type=float,
        default=60.0,
        help="Maximum time (in seconds) to run.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0: silent, 1: warnings, 2: info.",
    )
    parser.add_argument(
        "--display_mode",
        "-disp",
        type=str,
        default="live",
        choices=["live", "log"],
        help="Display mode for the results."
        "live: in-place refreshing progress bar, "
        "log: multiline logging of progress.",
    )
    parser.add_argument(
        "--print_every",
        "-p",
        type=int,
        default=10,
        help="Print progress every N evaluations.",
    )
    parser.add_argument(
        "--save_to_file", "-sv", action="store_true", help="Save results to a CSV file."
    )
    parser.add_argument(
        "--output_dir",
        "-out",
        type=Path,
        default=DATA_DIR,
        help="Directory to save results and run data.",
    )
    args = parser.parse_args()

    # Sanity checks
    if not isinstance(args.output_dir, Path):
        args.output_dir = Path(args.output_dir)

    assert all(algo_name in algorithms for algo_name in args.optimizers), (
        "Some optimizers are not recognized. \nAvailable optimizers: {}".format(
            list(algorithms.keys())
        )
    )

    assert args.problem in problems, (
        "Problem {} is not recognized. \nAvailable problems: {}".format(
            args.problem, list(problems.keys())
        )
    )

    # Create the algorithm configurations

    algo_configs: list[AlgorithmConfig] = [
        AlgorithmConfig(
            algorithm=algorithms[algo_name],
        )
        for algo_name in args.optimizers
    ]

    # Create the benchmark
    benchmark = Benchmark(
        problem=problems[args.problem],
        configs=algo_configs,
        random_seed=args.seed,
        n_runs=args.n_runs,
        max_time=args.max_time,
        max_evals=args.max_evals,
        data_dir=args.output_dir,
    )

    # Run the benchmark

    benchmark.run(
        verbose=args.verbose,
        print_every=args.print_every,
        save_csv=args.save_to_file,
        save_run_data=args.save_to_file,
    )
