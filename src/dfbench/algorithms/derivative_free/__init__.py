"""Derivative-free optimization algorithms.

Includes mesh-based direct search (OMADS), Powell-style trust-region methods
(PDFO: UOBYQA, NEWUOA, LINCOA; plus Py-BOBYQA), and SciPy classics
(Nelder-Mead, Powell).
"""

from dfbench.algorithms.derivative_free.nelder_mead import NelderMead
from dfbench.algorithms.derivative_free.omads_mads import OmadsMADS, OmadsOrthoMADS
from dfbench.algorithms.derivative_free.pdfo import PDFOLINCOA, PDFONEWUOA, PDFOUOBYQA
from dfbench.algorithms.derivative_free.powell import Powell
from dfbench.algorithms.derivative_free.pybobyqa import PyBOBYQA

__all__ = [
    "NelderMead",
    "OmadsMADS",
    "OmadsOrthoMADS",
    "PDFOLINCOA",
    "PDFONEWUOA",
    "PDFOUOBYQA",
    "Powell",
    "PyBOBYQA",
]
