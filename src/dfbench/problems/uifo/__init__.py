"""UIFO optimization problems."""

from dfbench.problems.uifo.random_uifo_problem import RandomUIFOProblem

# Backwards compatibility alias
UIFOProblem = RandomUIFOProblem

__all__ = [
    "RandomUIFOProblem",
    "UIFOProblem",
]
