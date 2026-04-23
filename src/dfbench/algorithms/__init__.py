"""Optimization algorithms."""

from dfbench.algorithms.derivative_free.omads_mads import OmadsMADS, OmadsOrthoMADS
from dfbench.algorithms.evolutionary.evox_es import EvoxES
from dfbench.algorithms.evolutionary.evox_pso import EvoxPSO
from dfbench.algorithms.evolutionary.nevergrad.ngopt import NevergradNGOpt
from dfbench.algorithms.evolutionary.nevergrad.oneplusone import NevergradOnePlusOne
from dfbench.algorithms.evolutionary.nevergrad.tbpsa import NevergradTBPSA
from dfbench.algorithms.evolutionary.random_search import RandomSearch
from dfbench.algorithms.gradient_based.misc.adam_gd import AdamGD
from dfbench.algorithms.gradient_based.scipy.bfgs import BFGS
from dfbench.algorithms.gradient_based.scipy.cobyla import COBYLA
from dfbench.algorithms.gradient_based.scipy.cobyqa import COBYQA
from dfbench.algorithms.gradient_based.scipy.dogleg import Dogleg
from dfbench.algorithms.gradient_based.misc.lbfgs_gd import LBFGSGD
from dfbench.algorithms.gradient_based.scipy.lbfgsb import LBFGSB
from dfbench.algorithms.gradient_based.misc.na_adam_gd import NAAdamGD
from dfbench.algorithms.gradient_based.scipy.newton_cg import NewtonCG
from dfbench.algorithms.gradient_based.scipy.nonlinear_cg import NonlinearCG
from dfbench.algorithms.gradient_based.misc.sa_gd import SAGD
from dfbench.algorithms.gradient_based.scipy.slsqp import SLSQP
from dfbench.algorithms.gradient_based.scipy.sr1 import SR1
from dfbench.algorithms.gradient_based.scipy.tnc import TNC
from dfbench.algorithms.gradient_based.scipy.trust_constr import TrustConstr
from dfbench.algorithms.gradient_based.scipy.trust_krylov import TrustKrylov
from dfbench.algorithms.gradient_based.scipy.trust_ncg import TrustNCG
from dfbench.algorithms.surrogate_based.botorch_bo import BotorchBO
from dfbench.algorithms.surrogate_based.botorch_turbo import BotorchTuRBO
from dfbench.algorithms.surrogate_based.restir import ReSTIR
from dfbench.algorithms.generative.vae_sampling import VAESampling

# Optax
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
from dfbench.algorithms.gradient_based.misc.optax_lbfgs import OptaxLBFGS

__all__ = [
    "OmadsMADS",
    "OmadsOrthoMADS",
    "EvoxES",
    "EvoxPSO",
    "NevergradNGOpt",
    "NevergradOnePlusOne",
    "NevergradTBPSA",
    "RandomSearch",
    "AdamGD",
    "BFGS",
    "COBYLA",
    "COBYQA",
    "Dogleg",
    "LBFGSGD",
    "LBFGSB",
    "NAAdamGD",
    "NewtonCG",
    "NonlinearCG",
    "SAGD",
    "SLSQP",
    "SR1",
    "TNC",
    "TrustConstr",
    "TrustKrylov",
    "TrustNCG",
    "BotorchBO",
    "BotorchTuRBO",
    "ReSTIR",
    "VAESampling",
    # Optax batch
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
    "OptaxLBFGS",
]
