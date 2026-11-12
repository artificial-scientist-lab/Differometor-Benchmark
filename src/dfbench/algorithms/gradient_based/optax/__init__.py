"""Optax-based gradient optimizers."""

from importlib import import_module as _import_module

# Resolve public names on demand so unrelated optional backends stay unloaded.
_EXPORTS = {
    "OptaxAdam": ".adam",
    "OptaxAdamW": ".adamw",
    "OptaxAdaBelief": ".adabelief",
    "OptaxAdafactor": ".adafactor",
    "OptaxAMSGrad": ".amsgrad",
    "OptaxAdaGrad": ".adagrad",
    "OptaxAdaDelta": ".adadelta",
    "OptaxAdaMax": ".adamax",
    "OptaxAdaMaxW": ".adamaxw",
    "OptaxAdan": ".adan",
    "OptaxLion": ".lion",
    "OptaxLAMB": ".lamb",
    "OptaxNadam": ".nadam",
    "OptaxNadamW": ".nadamw",
    "OptaxRMSProp": ".rmsprop",
    "OptaxRProp": ".rprop",
    "OptaxRAdam": ".radam",
    "OptaxSGD": ".sgd",
    "OptaxSGDM": ".sgd",
    "OptaxNAG": ".sgd",
    "OptaxNoisySGD": ".noisy_sgd",
    "OptaxPolyakSGD": ".polyak_sgd",
    "OptaxSAM": ".sam",
    "OptaxSophia": ".sophia",
    "OptaxLookahead": ".lookahead",
    "OptaxScheduleFreeAdam": ".schedule_free_adam",
    "OptaxYogi": ".yogi",
    "OptaxNovoGrad": ".novograd",
    "OptaxOGD": ".ogd",
    "OptaxOAdam": ".oadam",
    "OptaxSignSGD": ".sign",
    "OptaxSignum": ".sign",
    "OptaxSM3": ".sm3",
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
