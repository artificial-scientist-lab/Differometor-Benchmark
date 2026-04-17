"""Optax-based gradient optimizers."""

from dfbench.algorithms.gradient_based.optax.adam import OptaxAdam
from dfbench.algorithms.gradient_based.optax.adamw import OptaxAdamW
from dfbench.algorithms.gradient_based.optax.adabelief import OptaxAdaBelief
from dfbench.algorithms.gradient_based.optax.adafactor import OptaxAdafactor
from dfbench.algorithms.gradient_based.optax.amsgrad import OptaxAMSGrad
from dfbench.algorithms.gradient_based.optax.adagrad import OptaxAdaGrad
from dfbench.algorithms.gradient_based.optax.adadelta import OptaxAdaDelta
from dfbench.algorithms.gradient_based.optax.adamax import OptaxAdaMax
from dfbench.algorithms.gradient_based.optax.adamaxw import OptaxAdaMaxW
from dfbench.algorithms.gradient_based.optax.adan import OptaxAdan
from dfbench.algorithms.gradient_based.optax.lion import OptaxLion
from dfbench.algorithms.gradient_based.optax.lamb import OptaxLAMB
from dfbench.algorithms.gradient_based.optax.nadam import OptaxNadam
from dfbench.algorithms.gradient_based.optax.nadamw import OptaxNadamW
from dfbench.algorithms.gradient_based.optax.rmsprop import OptaxRMSProp
from dfbench.algorithms.gradient_based.optax.rprop import OptaxRProp
from dfbench.algorithms.gradient_based.optax.radam import OptaxRAdam
from dfbench.algorithms.gradient_based.optax.sgd import OptaxSGD, OptaxSGDM, OptaxNAG
from dfbench.algorithms.gradient_based.optax.noisy_sgd import OptaxNoisySGD
from dfbench.algorithms.gradient_based.optax.polyak_sgd import OptaxPolyakSGD
from dfbench.algorithms.gradient_based.optax.sam import OptaxSAM
from dfbench.algorithms.gradient_based.optax.sophia import OptaxSophia
from dfbench.algorithms.gradient_based.optax.lookahead import OptaxLookahead
from dfbench.algorithms.gradient_based.optax.schedule_free_adam import OptaxScheduleFreeAdam
from dfbench.algorithms.gradient_based.optax.yogi import OptaxYogi
from dfbench.algorithms.gradient_based.optax.novograd import OptaxNovoGrad
from dfbench.algorithms.gradient_based.optax.ogd import OptaxOGD
from dfbench.algorithms.gradient_based.optax.oadam import OptaxOAdam
from dfbench.algorithms.gradient_based.optax.sign import OptaxSignSGD, OptaxSignum
from dfbench.algorithms.gradient_based.optax.sm3 import OptaxSM3

__all__ = [
    "OptaxAdam",
    "OptaxAdamW",
    "OptaxAdaBelief",
    "OptaxAdafactor",
    "OptaxAMSGrad",
    "OptaxAdaGrad",
    "OptaxAdaDelta",
    "OptaxAdaMax",
    "OptaxAdaMaxW",
    "OptaxAdan",
    "OptaxLion",
    "OptaxLAMB",
    "OptaxNadam",
    "OptaxNadamW",
    "OptaxRMSProp",
    "OptaxRProp",
    "OptaxRAdam",
    "OptaxSGD",
    "OptaxSGDM",
    "OptaxNAG",
    "OptaxNoisySGD",
    "OptaxPolyakSGD",
    "OptaxSAM",
    "OptaxSophia",
    "OptaxLookahead",
    "OptaxScheduleFreeAdam",
    "OptaxYogi",
    "OptaxNovoGrad",
    "OptaxOGD",
    "OptaxOAdam",
    "OptaxSignSGD",
    "OptaxSignum",
    "OptaxSM3",
]
