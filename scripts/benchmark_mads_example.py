"""Benchmark example for MADS and OrthoMADS direct-search algorithms.

Demonstrates how to run and compare MADS vs OrthoMADS on VoyagerProblem using
the dfbench Benchmark infrastructure.

Usage::

    python scripts/benchmark_mads_example.py

Both algorithms treat the problem as a rugged-landscape local search problem
and operate entirely in bounded physical space.  The benchmark runs each
algorithm a small number of times to keep the script fast; increase ``n_runs``
and the budgets for a production benchmark.
"""

from dfbench.algorithms import MADS, OrthoMADS
from dfbench.benchmark import AlgorithmConfig, Benchmark
from dfbench.problems import VoyagerProblem

problem = VoyagerProblem()

MAX_EVALS = 300
MAX_TIME = 120.0  # seconds
N_RUNS = 3
SUCCESS_LOSS = 0.1

configs = [
    AlgorithmConfig(
        MADS(poll_size_init=1.0),
        hyperparameters={"opportunistic": False},
        name="MADS_default",
    ),
    AlgorithmConfig(
        MADS(poll_size_init=0.5),
        hyperparameters={"opportunistic": True},
        name="MADS_opp",
    ),
    AlgorithmConfig(
        OrthoMADS(poll_size_init=1.0),
        hyperparameters={"opportunistic": False},
        name="OrthoMADS_default",
    ),
    AlgorithmConfig(
        OrthoMADS(poll_size_init=0.5),
        hyperparameters={"opportunistic": True},
        name="OrthoMADS_opp",
    ),
]

benchmark = Benchmark(
    problem,
    success_loss=SUCCESS_LOSS,
    configs=configs,
    n_runs=N_RUNS,
    max_evals=MAX_EVALS,
    max_time=MAX_TIME,
)

if __name__ == "__main__":
    results = benchmark.run()
    for name, result in results.items():
        best = min(r.best_loss for r in result.runs)
        print(f"{name}: best_loss={best:.4f}")
