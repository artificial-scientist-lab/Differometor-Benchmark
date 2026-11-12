"""Global search algorithms (stochastic, multi-start methods).

Wraps SciPy's ``basinhopping`` and ``dual_annealing`` for benchmark use."""

from importlib import import_module as _import_module

# Resolve public names on demand so unrelated optional backends stay unloaded.
_EXPORTS = {
    "BasinHopping": ".basin_hopping",
    "DualAnnealing": ".dual_annealing",
    "RandomSearch": ".random_search",
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
