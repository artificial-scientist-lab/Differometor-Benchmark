"""Differometor Benchmark package.

Provides optimization algorithms, problem definitions, and benchmarking tools.
"""

# Initialize environment variables first
import dfbench.core._init_env  # noqa: F401

# Import protocols
from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)

# Import utilities
from dfbench.core.config import create_parser
from dfbench.core.utils import t2j, j2t, t2j_numpy, j2t_numpy

# Import algorithms
from dfbench.algorithms.evolutionary.evox_pso import EvoxPSO
from dfbench.algorithms.evolutionary.random_search import RandomSearch
from dfbench.algorithms.gradient_based.adam_gd import AdamGD
from dfbench.algorithms.gradient_based.sa_gd import SAGD
from dfbench.algorithms.surrogate_based.botorch_bo import BotorchBO

# Import problems
from dfbench.problems.voyager_problem import VoyagerProblem

# Import benchmarking
from dfbench.benchmark.benchmark import Benchmark, AlgorithmConfig


__all__ = [
    # Protocols
    "ContinuousProblem",
    "OptimizationAlgorithm",
    "AlgorithmType",
    # Algorithms
    "EvoxPSO",
    "RandomSearch",
    "AdamGD",
    "SAGD",
    "BotorchBO",
    # Problems
    "VoyagerProblem",
    # Utilities
    "create_parser",
    "t2j",
    "j2t",
    "t2j_numpy",
    "j2t_numpy",
    # Benchmarking
    "Benchmark",
    "AlgorithmConfig",
]
