"""Differometor Benchmark package.

Provides optimization algorithms, problem definitions, and benchmarking tools.

## Who imports what

**Submitters** (people writing an optimization algorithm to be evaluated)
only need the top-level namespace::

    from dfbench import Objective
    from dfbench import ContinuousProblem, OptimizationAlgorithm, AlgorithmType
    from dfbench.algorithms import AdamGD, EvoxES, BotorchBO
    from dfbench.problems import VoyagerProblem, UIFOProblem

`Objective` handles checkpointing, history tracking, and budget enforcement
internally; the only submitter-facing storage knob is
``Objective(..., save_to_file_every=N)``. Submitters never need to import
from :mod:`dfbench.core.storage`.

**Organizers** (people running scoring, checkpoints, and leaderboards)
import the modular storage stack and the benchmark harness directly::

    from dfbench.core.storage import CheckpointManager, RunState, validate_run_state
    from dfbench.benchmark import Benchmark, AlgorithmConfig

The problem-reconstruction contract
(:func:`dfbench.core.problem.build_problem_from_spec`,
:func:`dfbench.core.problem.register_problem`,
:func:`dfbench.core.problem.validate_spec_round_trip`) lives in
:mod:`dfbench.core.problem`, not in the storage package.
"""

# Initialize environment variables first
import dfbench.core._init_env  # noqa: F401

# Protocols: part of the submitter-facing surface
from dfbench.core.problem import ContinuousProblem
from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType

# Core utilities a submitter uses in their optimize() loop
from dfbench.core.config import create_parser
from dfbench.core.utils import t2j, j2t

# Objective: the single class a submitter drives
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
