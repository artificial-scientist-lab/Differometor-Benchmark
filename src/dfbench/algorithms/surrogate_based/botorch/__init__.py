"""BoTorch-based surrogate algorithms.

Bundles all algorithms whose surrogate model is built on top of the
BoTorch / GPyTorch stack (vanilla GP-EI BO, TuRBO, qNEI, qKG, REMBO,
GEBO, LineBO, ...)."""

from importlib import import_module as _import_module

# Resolve public names on demand so unrelated optional backends stay unloaded.
_EXPORTS = {
    "BotorchBO": ".botorch_bo",
    "BotorchTuRBO": ".botorch_turbo",
    "BotorchqKG": ".botorch_qkg",
    "BotorchQNEI": ".botorch_qnei",
    "GEBO": ".botorch_gebo",
    "LineBO": ".botorch_linebo",
    "REMBO": ".botorch_rembo",
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
