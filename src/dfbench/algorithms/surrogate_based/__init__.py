"""Surrogate-based optimization algorithms.

These algorithms build a surrogate model (e.g., Gaussian Process) of the
objective function and use it to guide the search for optimal parameters.
"""

from dfbench.algorithms.surrogate_based.botorch_bo import BotorchBO
from dfbench.algorithms.surrogate_based.botorch_turbo import BotorchTuRBO

__all__ = [
    "BotorchBO",
    "BotorchTuRBO",
]
