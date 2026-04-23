"""Surrogate-based optimization algorithms.

These algorithms build a surrogate model (e.g., Gaussian Process) of the
objective function and use it to guide the search for optimal parameters.
"""

from dfbench.algorithms.surrogate_based.ax_baxus import BAxUS
from dfbench.algorithms.surrogate_based.botorch import (
    BotorchBO,
    BotorchTuRBO,
    BotorchqKG,
    BotorchqNEI,
    GEBO,
    LineBO,
    REMBO,
)
from dfbench.algorithms.surrogate_based.hebo_bo import HEBO
from dfbench.algorithms.surrogate_based.restir import ReSTIR
from dfbench.algorithms.surrogate_based.turbo_lbfgs import TuRBOLBFGS

# External-package algorithms: imported only when their dependencies exist.
try:
    from dfbench.algorithms.surrogate_based.ax_saasbo import AxSAASBO
except ImportError:
    pass

try:
    from dfbench.algorithms.surrogate_based.smac_bo import SMAC
except ImportError:
    pass

__all__ = [
    "BotorchBO",
    "BotorchTuRBO",
    "ReSTIR",
    "AxSAASBO",
    "BAxUS",
    "BotorchqNEI",
    "BotorchqKG",
    "REMBO",
    "GEBO",
    "LineBO",
    "HEBO",
    "SMAC",
    "TuRBOLBFGS",
]
