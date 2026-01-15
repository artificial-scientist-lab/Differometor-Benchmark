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
from dfbench.algorithms.evolutionary.evox_es import EvoxES
from dfbench.algorithms.evolutionary.evox_pso import EvoxPSO
from dfbench.algorithms.evolutionary.random_search import RandomSearch
from dfbench.algorithms.gradient_based.adam_gd import AdamGD
from dfbench.algorithms.gradient_based.na_adam_gd import NAAdamGD
from dfbench.algorithms.gradient_based.sa_gd import SAGD
from dfbench.algorithms.surrogate_based.botorch_bo import BotorchBO
from dfbench.algorithms.surrogate_based.botorch_turbo import BotorchTuRBO

# Import problems
from dfbench.problems.voyager.voyager_problem import VoyagerProblem
from dfbench.problems.voyager.constrained_voyager_problem import ConstrainedVoyagerProblem
from dfbench.problems.uifo.random_uifo_problem import RandomUIFOProblem
from dfbench.problems.base_problem import OpticalSetupProblem

# Backwards compatibility alias
UIFOProblem = RandomUIFOProblem

# Import benchmarking
from dfbench.benchmark.benchmark import Benchmark, AlgorithmConfig


__all__ = [
    # Protocols
    "ContinuousProblem",
    "OptimizationAlgorithm",
    "AlgorithmType",
    # Algorithms
    "EvoxES",
    "EvoxPSO",
    "RandomSearch",
    "AdamGD",
    "NAAdamGD",
    "SAGD",
    "BotorchBO",
    "BotorchTuRBO",
    # Problems
    "OpticalSetupProblem",
    "VoyagerProblem",
    "ConstrainedVoyagerProblem",
    "RandomUIFOProblem",
    "UIFOProblem",
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
