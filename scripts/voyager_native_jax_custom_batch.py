"""Benchmark example for native-JAX custom/hybrid gradient-based algorithms."""

from dfbench.algorithms import (
    AdamToLBFGSJAX,
    GaussianSmoothingGDJAX,
    NoisyAdamJAX,
    SGLDJAX,
)
from dfbench.benchmark import AlgorithmConfig, Benchmark
from dfbench.problems import VoyagerProblem


problem = VoyagerProblem()

configs = [
    AlgorithmConfig(
        SGLDJAX(),
        {
            "learning_rate": 0.03,
            "temperature": 0.5,
            "restart_every": 60,
        },
        name="SGLD-JAX",
    ),
    AlgorithmConfig(
        NoisyAdamJAX(),
        {
            "learning_rate": 0.05,
            "noise_std": 0.01,
            "restart_every": 80,
        },
        name="NoisyAdam-JAX",
    ),
    AlgorithmConfig(
        AdamToLBFGSJAX(),
        {
            "adam_learning_rate": 0.05,
            "adam_fraction": 0.6,
            "min_adam_steps": 20,
        },
        name="Adam->LBFGS-JAX",
    ),
    AlgorithmConfig(
        GaussianSmoothingGDJAX(),
        {
            "learning_rate": 0.03,
            "sigma": 0.05,
            "n_directions": 4,
        },
        name="GaussSmooth+GD-JAX",
    ),
]

benchmark = Benchmark(
    problem=problem,
    success_loss=50,
    configs=configs,
    n_runs=3,
    max_time=90,
    n_time_samples=9,
)

results = benchmark.run(save_csv=False)
benchmark.print_summary(results)
