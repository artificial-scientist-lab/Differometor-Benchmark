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
from dfbench.core.problem import (
    ContinuousProblem,
    build_problem_from_spec,
    register_problem,
    validate_spec_round_trip,
)
from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType

# Import core utilities
from dfbench.core.config import create_parser
from dfbench.core.utils import t2j, j2t

# Import Objective for external use
from dfbench.core.objective import Objective

# Import modular storage components
from dfbench.core.storage import (
    CheckpointManager,
    CheckpointSerializer,
    JsonCheckpointSerializer,
    LocalFilesystemBackend,
    NpzCheckpointSerializer,
    RunDataExporter,
    RunMetadata,
    RunPathResolver,
    RunState,
    StorageBackend,
)


__all__ = [
    # Core
    "Objective",
    # Protocols
    "ContinuousProblem",
    "OptimizationAlgorithm",
    "AlgorithmType",
    # Problem reconstruction
    "build_problem_from_spec",
    "register_problem",
    "validate_spec_round_trip",
    # Utilities
    "create_parser",
    "t2j",
    "j2t",
    # Modular storage
    "CheckpointManager",
    "CheckpointSerializer",
    "JsonCheckpointSerializer",
    "LocalFilesystemBackend",
    "NpzCheckpointSerializer",
    "RunDataExporter",
    "RunMetadata",
    "RunPathResolver",
    "RunState",
    "StorageBackend",
]
