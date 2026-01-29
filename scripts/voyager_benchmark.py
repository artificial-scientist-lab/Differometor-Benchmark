"""Voyager benchmark script for optimization algorithms.

Usage:
    python voyager_benchmark.py -i <config_index> -s <seed>

All algorithms now return Objective instances and use the new Benchmark class.
"""

from dfbench.algorithms import (
    EvoxPSO,
    EvoxES,
    AdamGD,
    NAAdamGD,
    SAGD,
    BotorchBO,
    BotorchTuRBO,
    VAESampling,
)
from dfbench.benchmark import Benchmark, AlgorithmConfig
from dfbench.problems import ConstrainedVoyagerProblem
import argparse

# CLI args: allow selecting which config to run via command line
parser = argparse.ArgumentParser(description="Run a single voyager benchmark config by index")
parser.add_argument("-i", "--index", type=int, default=0, help="index of config to run (0-based)")  # ! Not currently used; runs all configs
parser.add_argument("-s", "--seed", type=int, default=0, help="random seed for benchmark runs")
parser.add_argument("-v", "--verbose", type=int, default=0, help="verbosity level (0=silent, 1=progress)")
args = parser.parse_args()
i = args.index
s = args.seed
v = args.verbose

# Setup
problem = ConstrainedVoyagerProblem()

# All algorithm configurations
configs = [
    # Gradient-based
    AlgorithmConfig(AdamGD(problem, verbose=v), {"learning_rate": 0.1, "patience": 500}, "Adam-0.1"),
    AlgorithmConfig(NAAdamGD(problem, verbose=v), {"noise_anneal_iters": 5000, "noise_schedule": "linear"}, "NAAdam-lin"),
    AlgorithmConfig(NAAdamGD(problem, verbose=v), {"noise_anneal_iters": 5000, "noise_schedule": "exponential"}, "NAAdam-exp"),
    AlgorithmConfig(SAGD(problem, verbose=v), {}, "SAGD-default"),
    AlgorithmConfig(SAGD(problem, verbose=v), {"use_double_annealing": True}, "SAGD-double"),
    # Evolutionary (PSO variants)
    AlgorithmConfig(EvoxPSO(problem, verbose=v, batch_size=125), {"pop_size": 250}, "PSO-250"),
    AlgorithmConfig(EvoxPSO(problem, verbose=v, variant="CSO", batch_size=125), {"pop_size": 500}, "CSO-500"),
    AlgorithmConfig(EvoxPSO(problem, verbose=v, variant="CLPSO", batch_size=125), {"pop_size": 500}, "CLPSO-500"),
    AlgorithmConfig(EvoxPSO(problem, verbose=v, variant="FSPSO", batch_size=125), {"pop_size": 500}, "FSPSO-500"),
    AlgorithmConfig(EvoxPSO(problem, verbose=v, variant="SLPSOGS", batch_size=125), {"pop_size": 50}, "SLPSOGS-50"),
    AlgorithmConfig(EvoxPSO(problem, verbose=v, variant="SLPSOUS", batch_size=125), {"pop_size": 50}, "SLPSOUS-50"),
    # Evolutionary (ES variants)
    AlgorithmConfig(EvoxES(problem, verbose=v, batch_size=50, variant="SNES"), {"pop_size": 50}, "SNES-50"),
    # AlgorithmConfig(EvoxES(problem, batch_size=50, variant="OpenES"), {"pop_size": 50}, "OpenES-50"),
    # Surrogate-based
    AlgorithmConfig(BotorchBO(problem, verbose=v), {}, "BotorchBO"),
    AlgorithmConfig(BotorchTuRBO(problem, verbose=v), {}, "BotorchTuRBO"),
    # Generative
    AlgorithmConfig(VAESampling(problem, verbose=v, batch_size=64, hidden_dim=256, num_blocks=4), {"sampling_time_percentage": 0.5, "top_k": 20}, "VAE-0.5-top2%"),
    AlgorithmConfig(VAESampling(problem, verbose=v, batch_size=64, hidden_dim=256, num_blocks=4), {"sampling_time_percentage": 0.8, "top_k": 20}, "VAE-0.8-top2%"),
]
if i < 0 or i >= len(configs):
    raise IndexError(f"config index {i} out of range (0..{len(configs)-1})")

# Run benchmark using new interface
# max_time=1200 (20 minutes), n_time_samples=80 gives time points every 15 seconds
benchmark = Benchmark(
    problem=problem,
    success_loss=1.0,
    configs=configs,# [configs[i]], for convenience when submitting multiple jobs
    n_runs=3,
    max_time=150.0,
    n_time_samples=10,
    random_baseline_loss=50,  # Random baseline for AUC
    random_seed=s,
)

results = benchmark.run(save_csv=True, save_run_data=True)
benchmark.print_summary(results)
