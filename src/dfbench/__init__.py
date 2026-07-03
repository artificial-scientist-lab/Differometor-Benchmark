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
    `from dfbench.problems import VoyagerProblem, UIFOProblem`

    ### Benchmarking (hierarchical)
    `from dfbench.benchmark import Benchmark, AlgorithmConfig`
"""

# Initialize environment variables first
import dfbench.core._init_env  # noqa: F401

# Import protocols
from dfbench.core.problem import ContinuousProblem
from dfbench.benchmark import Benchmark, AlgorithmConfig
from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench.core.parameters import (
    DiscreteParameter,
    FloatParameter,
    IntegerParameter,
    Parameter,
)
from dfbench.core.parameter_config import ParameterConfig
from dfbench.core.search_space import SearchDimension, SearchSpace, TargetRef
from dfbench.core.state import OptimizationState

# Import core utilities
from dfbench.core.config import create_parser
from dfbench.core.utils import t2j, j2t

# Import Objective for external use
from dfbench.core.objective import Objective


__all__ = [
    # Core
    "Objective",
    "Parameter",
    "ParameterConfig",
    "FloatParameter",
    "IntegerParameter",
    "DiscreteParameter",
    "OptimizationState",
    "TargetRef",
    "SearchDimension",
    "SearchSpace",
    # Benchmarking
    "Benchmark",
    "AlgorithmConfig",
    # Protocols
    "ContinuousProblem",
    "OptimizationAlgorithm",
    "AlgorithmType",
    # Utilities
    "create_parser",
    "t2j",
    "j2t",
]
