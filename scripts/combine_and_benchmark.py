#!/usr/bin/env python3
"""Run the benchmark evaluation on pre-combined run data.

Usage:
    python scripts/combine_and_benchmark.py [--data-dir DIR] [--success-loss FLOAT]

Defaults:
    --data-dir      ./data/random_uifo_benchmark/combined
    --success-loss  50.0
"""

import argparse
import json
from pathlib import Path

from dfbench.problems import RandomUIFOProblem
from dfbench.benchmark import Benchmark, AlgorithmConfig


TOTAL_RUNS = 30
MAX_TIME = 7200.0  # 2 hours per run


def run_benchmark(combined_dir: Path, success_loss: float, n_time_samples: int):
    """Run the benchmark evaluation on combined data using Benchmark.run(load_from=...)."""
    print("=" * 70)
    print("Benchmark Evaluation: RandomUIFO")
    print("=" * 70)
    print(f"Data dir:       {combined_dir}")
    print(f"Success loss:   {success_loss}")
    print(f"Time samples:   {n_time_samples}")
    print()

    # Load metadata to read n_runs / max_time if available
    metadata_path = combined_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            meta = json.load(f)
        n_runs = meta.get("n_runs", TOTAL_RUNS)
        max_time = meta.get("max_time", MAX_TIME)
        print(
            f"Loaded metadata: {len(meta.get('algorithms', []))} algorithms, "
            f"{n_runs} runs, max_time={max_time}s"
        )
    else:
        n_runs = TOTAL_RUNS
        max_time = MAX_TIME

    problem = RandomUIFOProblem()

    # Dummy config — not used when loading, but Benchmark requires it
    from dfbench.algorithms import AdamGD

    dummy_configs = [AlgorithmConfig(AdamGD(), {}, name="dummy")]

    benchmark = Benchmark(
        problem=problem,
        success_loss=success_loss,
        configs=dummy_configs,
        n_runs=n_runs,
        max_time=max_time,
        n_time_samples=n_time_samples,
    )

    results = benchmark.run(
        save_csv=True,
        save_run_data=False,
        load_from=str(combined_dir),
        verbose=1,
    )

    benchmark.print_summary(results)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation on pre-combined run data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/random_uifo_benchmark/combined",
        help="Directory containing combined NPZ files and metadata.json",
    )
    parser.add_argument(
        "--success-loss",
        type=float,
        default=0.0,
        help="Loss threshold for success (default: 50.0)",
    )
    parser.add_argument(
        "--n-time-samples",
        type=int,
        default=10,
        help="Number of time sample points for metrics (default: 100)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        return

    run_benchmark(data_dir, args.success_loss, args.n_time_samples)
    print("\nDone!")


if __name__ == "__main__":
    main()
