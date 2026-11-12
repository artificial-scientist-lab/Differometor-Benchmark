"""Evolutionary optimization algorithms."""

from importlib import import_module as _import_module

# Resolve public names on demand so unrelated optional backends stay unloaded.
_EXPORTS = {
    "EvoxES": ".evox_es",
    "EvoxPSO": ".evox_pso",
    "NevergradNGOpt": ".nevergrad.ngopt",
    "NevergradOnePlusOne": ".nevergrad.oneplusone",
    "NevergradTBPSA": ".nevergrad.tbpsa",
    "PyCMACMAES": ".pycma_cmaes",
    "PyCMAActiveCMAES": ".pycma_cmaes",
    "PyCMAIPOP": ".pycma_cmaes",
    "PyCMABIPOP": ".pycma_cmaes",
    "CMAESSepCMA": ".cmaes_sep_cma",
    "EvosaxMAES": ".evosax_es",
    "EvosaxLMMAES": ".evosax_es",
    "JAXOnePlusOneES": ".jax_es",
    "JAXMuLambdaES": ".jax_es",
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
