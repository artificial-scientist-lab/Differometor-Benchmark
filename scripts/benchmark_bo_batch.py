"""Benchmark script for the structured BO / surrogate batch.

Runs all BO algorithms that have their dependencies available against
VoyagerProblem with a moderate time budget. Designed to be executed via
srun on the cluster.

Usage (compute node):
    srun -p a100-galvani --gres=gpu:1 --time=0-02:00 \
        python scripts/benchmark_bo_batch.py

Usage (quick local test):
    python scripts/benchmark_bo_batch.py --quick
"""

from __future__ import annotations

import argparse
import importlib
import sys

from dfbench.problems import VoyagerProblem
from dfbench.benchmark import Benchmark, AlgorithmConfig

# ── Always-available BoTorch algorithms ──────────────────────────────
from dfbench.algorithms.surrogate_based.botorch_qnei import BotorchqNEI
from dfbench.algorithms.surrogate_based.botorch_qkg import BotorchqKG
from dfbench.algorithms.surrogate_based.botorch_rembo import REMBO
from dfbench.algorithms.surrogate_based.botorch_gebo import GEBO
from dfbench.algorithms.surrogate_based.botorch_linebo import LineBO
from dfbench.algorithms.surrogate_based.ax_baxus import BAxUS
from dfbench.algorithms.surrogate_based.turbo_lbfgs import TuRBOLBFGS


def build_configs(*, quick: bool = False):
    """Build algorithm configs, skipping unavailable packages."""
    n_init = 5 if quick else 20
    bo_iters = 3 if quick else 40
    turbo_iters = 3 if quick else 30

    configs = [
        AlgorithmConfig(
            BotorchqNEI(),
            {"n_initial": n_init, "batch_size": 1, "max_iterations": bo_iters},
            name="qNEI",
        ),
        AlgorithmConfig(
            BotorchqKG(),
            {"n_initial": n_init, "batch_size": 1, "max_iterations": bo_iters, "num_fantasies": 8},
            name="qKG",
        ),
        AlgorithmConfig(
            REMBO(),
            {"n_initial": n_init, "max_iterations": bo_iters, "d_embedding": 6},
            name="REMBO",
        ),
        AlgorithmConfig(
            GEBO(),
            {"n_initial": n_init, "max_iterations": bo_iters},
            name="GEBO",
        ),
        AlgorithmConfig(
            LineBO(),
            {"n_initial": n_init, "max_iterations": bo_iters, "line_samples": 10},
            name="LineBO",
        ),
        AlgorithmConfig(
            BAxUS(),
            {"n_initial": n_init, "max_iterations": bo_iters, "d_init": 4},
            name="BAxUS",
        ),
        AlgorithmConfig(
            TuRBOLBFGS(),
            {"turbo_iterations": turbo_iters, "n_initial": n_init, "lbfgs_patience": 200},
            name="TuRBO→L-BFGS",
        ),
    ]

    # Conditional: Ax SAASBO
    if importlib.util.find_spec("ax") is not None:
        from dfbench.algorithms.surrogate_based.ax_saasbo import AxSAASBO

        configs.append(
            AlgorithmConfig(
                AxSAASBO(),
                {
                    "n_initial": n_init,
                    "max_iterations": bo_iters,
                    "num_warmup": 64 if quick else 256,
                    "num_samples": 16 if quick else 128,
                },
                name="SAASBO",
            )
        )

    # Conditional: HEBO
    if importlib.util.find_spec("hebo") is not None:
        from dfbench.algorithms.surrogate_based.hebo_bo import HEBO

        configs.append(
            AlgorithmConfig(
                HEBO(),
                {"batch_size": 1, "max_iterations": n_init + bo_iters},
                name="HEBO",
            )
        )

    # Conditional: SMAC
    if importlib.util.find_spec("smac") is not None:
        from dfbench.algorithms.surrogate_based.smac_bo import SMAC

        configs.append(
            AlgorithmConfig(
                SMAC(),
                {"n_initial": n_init, "max_iterations": bo_iters},
                name="SMAC",
            )
        )

    return configs


def main():
    parser = argparse.ArgumentParser(description="BO batch benchmark")
    parser.add_argument("--quick", action="store_true", help="Tiny budget for local smoke test")
    args = parser.parse_args()

    problem = VoyagerProblem()
    configs = build_configs(quick=args.quick)

    n_runs = 1 if args.quick else 3
    max_time = 60 if args.quick else 300
    n_time_samples = 5 if args.quick else 15
    success_loss = 50

    print("=" * 70)
    print("BO Batch Benchmark")
    print("=" * 70)
    print(f"Problem: {problem.name}  (dim={problem.n_params})")
    print(f"Algorithms: {len(configs)}")
    for c in configs:
        print(f"  - {c.name}")
    print(f"Runs: {n_runs}, max_time: {max_time}s")
    print()

    benchmark = Benchmark(
        problem=problem,
        success_loss=success_loss,
        configs=configs,
        n_runs=n_runs,
        max_time=max_time,
        n_time_samples=n_time_samples,
    )

    results = benchmark.run(save_csv=not args.quick)
    benchmark.print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
