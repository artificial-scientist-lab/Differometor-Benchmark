"""Gradient-based optimization algorithms.

Organized into two sub-packages:
  - ``scipy``:  Wrappers around ``scipy.optimize.minimize``.
  - ``optax``:  Wrappers around Optax first-order optimizers.

Flat modules in this package provide custom-loop algorithms
(:mod:`adam_gd`, :mod:`lbfgs_gd`, :mod:`sa_gd`, :mod:`na_adam_gd`,
:mod:`optax_lbfgs`) and a hybrid / variance-reduced JAX batch
(:mod:`custom_jax`: SGLD, ASAM, GD->L-BFGS hand-off, etc.)."""

from importlib import import_module as _import_module

# Resolve public names on demand so unrelated optional backends stay unloaded.
_EXPORTS = {
    "BFGS": ".scipy.bfgs",
    "COBYLA": ".scipy.cobyla",
    "COBYQA": ".scipy.cobyqa",
    "Dogleg": ".scipy.dogleg",
    "LBFGSB": ".scipy.lbfgsb",
    "NewtonCG": ".scipy.newton_cg",
    "NonlinearCG": ".scipy.nonlinear_cg",
    "SLSQP": ".scipy.slsqp",
    "SR1": ".scipy.sr1",
    "TNC": ".scipy.tnc",
    "TrustConstr": ".scipy.trust_constr",
    "TrustKrylov": ".scipy.trust_krylov",
    "TrustNCG": ".scipy.trust_ncg",
    "OptaxAdam": ".optax.adam",
    "OptaxAdamW": ".optax.adamw",
    "OptaxAdaBelief": ".optax.adabelief",
    "OptaxAdafactor": ".optax.adafactor",
    "OptaxAMSGrad": ".optax.amsgrad",
    "OptaxAdaGrad": ".optax.adagrad",
    "OptaxAdaDelta": ".optax.adadelta",
    "OptaxAdaMax": ".optax.adamax",
    "OptaxAdaMaxW": ".optax.adamaxw",
    "OptaxAdan": ".optax.adan",
    "OptaxLion": ".optax.lion",
    "OptaxLAMB": ".optax.lamb",
    "OptaxNadam": ".optax.nadam",
    "OptaxNadamW": ".optax.nadamw",
    "OptaxRMSProp": ".optax.rmsprop",
    "OptaxRProp": ".optax.rprop",
    "OptaxRAdam": ".optax.radam",
    "OptaxSGD": ".optax.sgd",
    "OptaxSGDM": ".optax.sgd",
    "OptaxNAG": ".optax.sgd",
    "OptaxNoisySGD": ".optax.noisy_sgd",
    "OptaxPolyakSGD": ".optax.polyak_sgd",
    "OptaxSAM": ".optax.sam",
    "OptaxSophia": ".optax.sophia",
    "OptaxLookahead": ".optax.lookahead",
    "OptaxScheduleFreeAdam": ".optax.schedule_free_adam",
    "OptaxYogi": ".optax.yogi",
    "OptaxNovoGrad": ".optax.novograd",
    "OptaxOGD": ".optax.ogd",
    "OptaxOAdam": ".optax.oadam",
    "OptaxSignSGD": ".optax.sign",
    "OptaxSignum": ".optax.sign",
    "OptaxSM3": ".optax.sm3",
    "AdamGD": ".adam_gd",
    "LBFGSGD": ".lbfgs_gd",
    "NAAdamGD": ".na_adam_gd",
    "OptaxLBFGS": ".optax_lbfgs",
    "SAGD": ".sa_gd",
    "SGLDJAX": ".custom_jax",
    "ASAMJAX": ".custom_jax",
    "AdamToLBFGSJAX": ".custom_jax",
    "EntropySGDJAX": ".custom_jax",
    "SGHMCJAX": ".custom_jax",
    "ARCJAX": ".custom_jax",
    "OGDJAX": ".custom_jax",
    "OAdamJAX": ".custom_jax",
    "PerturbedGDJAX": ".custom_jax",
    "NoisyAdamJAX": ".custom_jax",
    "GDRestartsJAX": ".custom_jax",
    "GaussianSmoothingGDJAX": ".custom_jax",
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
