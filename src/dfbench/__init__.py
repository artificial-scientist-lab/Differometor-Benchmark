"""Differometor Benchmark package.

Provides optimization algorithms, problem definitions, and benchmarking tools.

Usage:
    ### Core classes
    `from dfbench import Objective`

    ### Protocols
    `from dfbench import ContinuousProblem, OptimizationAlgorithm, AlgorithmType`

    ### Algorithms (hierarchical)
    `from dfbench.algorithms import AdamGD, EvoxES, BotorchBO`

    ### Problems (hierarchical)
    `from dfbench.problems import VoyagerProblem, RandomUIFOProblem`

    ### Benchmarking (hierarchical)
    `from dfbench.benchmark import Benchmark, AlgorithmConfig`
"""

# Initialize environment variables first
import dfbench.core._init_env  # noqa: F401

# Import protocols
from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)

# Import core utilities
from dfbench.core.config import create_parser
from dfbench.core.utils import t2j, j2t

# Import Objective for external use
from dfbench.core.objective import Objective


__all__ = [
    # Core
    "Objective",
    # Protocols
    "ContinuousProblem",
    "OptimizationAlgorithm",
    "AlgorithmType",
    # Utilities
    "create_parser",
    "t2j",
    "j2t",
]
