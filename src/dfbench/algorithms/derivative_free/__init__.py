"""Derivative-free optimization algorithms.

Includes mesh-based direct search (OMADS), Powell-style trust-region methods
(PDFO: UOBYQA, NEWUOA, LINCOA; plus Py-BOBYQA), and SciPy classics
(Nelder-Mead, Powell)."""

from importlib import import_module as _import_module

# Resolve public names on demand so unrelated optional backends stay unloaded.
_EXPORTS = {
    "NelderMead": ".nelder_mead",
    "OmadsMADS": ".omads_mads",
    "OmadsOrthoMADS": ".omads_mads",
    "PDFOLINCOA": ".pdfo.lincoa",
    "PDFONEWUOA": ".pdfo.newuoa",
    "PDFOUOBYQA": ".pdfo.uobyqa",
    "Powell": ".powell",
    "PyBOBYQA": ".pybobyqa",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = _import_module(_EXPORTS[name], package=__name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
