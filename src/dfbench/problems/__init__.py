"""Optimization problems and scripts."""

from collections.abc import Mapping

# Import base class
from dfbench.problems.base_problem import OpticalSetupProblem

# Import power-penalty presets
from dfbench.problems.base_problem import (
    squashed_relu_penalty,
    relu_penalty,
    zero_penalty,
)

# Import problem classes
from dfbench.problems.voyager import (
    VoyagerProblem,
    VoyagerTuningProblem,
    ConstrainedVoyagerProblem,
)
from dfbench.problems.uifo import UIFOProblem, RandomUIFOProblem

__all__ = [
    # Base class
    "OpticalSetupProblem",
    # Penalty presets
    "squashed_relu_penalty",
    "relu_penalty",
    "zero_penalty",
    # Problem classes
    "VoyagerProblem",
    "VoyagerTuningProblem",
    "ConstrainedVoyagerProblem",
    "UIFOProblem",
    "RandomUIFOProblem",
]

problems: Mapping[str, type[OpticalSetupProblem]] = {
    _name: globals()[_name] for _name in __all__ if _name in globals()
}
