"""Surrogate-based optimization algorithms.

These algorithms build a surrogate model (e.g., Gaussian Process) of the
objective function and use it to guide the search for optimal parameters."""

from importlib import import_module as _import_module

# Resolve public names on demand so unrelated optional backends stay unloaded.
_EXPORTS = {
    "BotorchBO": ".botorch.botorch_bo",
    "BotorchTuRBO": ".botorch.botorch_turbo",
    "ReSTIR": ".restir",
    "AxSAASBO": ".ax_saasbo",
    "BAxUS": ".ax_baxus",
    "BotorchQNEI": ".botorch.botorch_qnei",
    "BotorchqKG": ".botorch.botorch_qkg",
    "REMBO": ".botorch.botorch_rembo",
    "GEBO": ".botorch.botorch_gebo",
    "LineBO": ".botorch.botorch_linebo",
    "HEBO": ".hebo_bo",
    "SMAC": ".smac_bo",
    "TuRBOLBFGS": ".turbo_lbfgs",
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
