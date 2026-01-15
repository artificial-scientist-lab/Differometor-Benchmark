"""Optimization problems and scripts."""

# Import base class
from dfbench.problems.base_problem import OpticalSetupProblem

# Import problem classes
from dfbench.problems.voyager import VoyagerProblem, ConstrainedVoyagerProblem
from dfbench.problems.uifo import RandomUIFOProblem, UIFOProblem

__all__ = [
    # Base class
    "OpticalSetupProblem",
    # Problem classes
    "VoyagerProblem",
    "ConstrainedVoyagerProblem",
    "RandomUIFOProblem",
    "UIFOProblem",
]
