"""SciPy-based gradient optimizers."""

from importlib import import_module as _import_module

# Resolve public names on demand so unrelated optional backends stay unloaded.
_EXPORTS = {
    "BFGS": ".bfgs",
    "COBYLA": ".cobyla",
    "COBYQA": ".cobyqa",
    "Dogleg": ".dogleg",
    "LBFGSB": ".lbfgsb",
    "NewtonCG": ".newton_cg",
    "NonlinearCG": ".nonlinear_cg",
    "SLSQP": ".slsqp",
    "SR1": ".sr1",
    "TNC": ".tnc",
    "TrustConstr": ".trust_constr",
    "TrustKrylov": ".trust_krylov",
    "TrustNCG": ".trust_ncg",
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
