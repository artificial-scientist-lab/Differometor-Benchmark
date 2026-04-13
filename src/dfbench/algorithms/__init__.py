"""Optimization algorithms."""

from dfbench.algorithms.evolutionary.evox_es import EvoxES
from dfbench.algorithms.evolutionary.evox_pso import EvoxPSO
from dfbench.algorithms.evolutionary.random_search import RandomSearch
from dfbench.algorithms.gradient_based.adam_gd import AdamGD
from dfbench.algorithms.gradient_based.bfgs import BFGS
from dfbench.algorithms.gradient_based.cobyla import COBYLA
from dfbench.algorithms.gradient_based.cobyqa import COBYQA
from dfbench.algorithms.gradient_based.dogleg import Dogleg
from dfbench.algorithms.gradient_based.lbfgs_gd import LBFGSGD
from dfbench.algorithms.gradient_based.lbfgsb import LBFGSB
from dfbench.algorithms.gradient_based.na_adam_gd import NAAdamGD
from dfbench.algorithms.gradient_based.newton_cg import NewtonCG
from dfbench.algorithms.gradient_based.nonlinear_cg import NonlinearCG
from dfbench.algorithms.gradient_based.sa_gd import SAGD
from dfbench.algorithms.gradient_based.slsqp import SLSQP
from dfbench.algorithms.gradient_based.sr1 import SR1
from dfbench.algorithms.gradient_based.tnc import TNC
from dfbench.algorithms.gradient_based.trust_constr import TrustConstr
from dfbench.algorithms.gradient_based.trust_krylov import TrustKrylov
from dfbench.algorithms.gradient_based.trust_ncg import TrustNCG
from dfbench.algorithms.surrogate_based.botorch_bo import BotorchBO
from dfbench.algorithms.surrogate_based.botorch_turbo import BotorchTuRBO
from dfbench.algorithms.surrogate_based.restir import ReSTIR
from dfbench.algorithms.generative.vae_sampling import VAESampling

# Optax batch
from dfbench.algorithms.gradient_based.optax_adam import OptaxAdam
from dfbench.algorithms.gradient_based.optax_adamw import OptaxAdamW
from dfbench.algorithms.gradient_based.optax_adabelief import OptaxAdaBelief
from dfbench.algorithms.gradient_based.optax_adafactor import OptaxAdafactor
from dfbench.algorithms.gradient_based.optax_amsgrad import OptaxAMSGrad
from dfbench.algorithms.gradient_based.optax_adagrad import OptaxAdaGrad
from dfbench.algorithms.gradient_based.optax_adadelta import OptaxAdaDelta
from dfbench.algorithms.gradient_based.optax_adamax import OptaxAdaMax
from dfbench.algorithms.gradient_based.optax_adamaxw import OptaxAdaMaxW
from dfbench.algorithms.gradient_based.optax_adan import OptaxAdan
from dfbench.algorithms.gradient_based.optax_lion import OptaxLion
from dfbench.algorithms.gradient_based.optax_lamb import OptaxLAMB
from dfbench.algorithms.gradient_based.optax_nadam import OptaxNadam
from dfbench.algorithms.gradient_based.optax_nadamw import OptaxNadamW
from dfbench.algorithms.gradient_based.optax_rmsprop import OptaxRMSProp
from dfbench.algorithms.gradient_based.optax_rprop import OptaxRProp
from dfbench.algorithms.gradient_based.optax_radam import OptaxRAdam
from dfbench.algorithms.gradient_based.optax_sgd import OptaxSGD, OptaxSGDM, OptaxNAG
from dfbench.algorithms.gradient_based.optax_noisy_sgd import OptaxNoisySGD
from dfbench.algorithms.gradient_based.optax_polyak_sgd import OptaxPolyakSGD
from dfbench.algorithms.gradient_based.optax_sam import OptaxSAM
from dfbench.algorithms.gradient_based.optax_sophia import OptaxSophia
from dfbench.algorithms.gradient_based.optax_lookahead import OptaxLookahead
from dfbench.algorithms.gradient_based.optax_schedule_free_adam import OptaxScheduleFreeAdam
from dfbench.algorithms.gradient_based.optax_yogi import OptaxYogi
from dfbench.algorithms.gradient_based.optax_novograd import OptaxNovoGrad
from dfbench.algorithms.gradient_based.optax_ogd import OptaxOGD
from dfbench.algorithms.gradient_based.optax_oadam import OptaxOAdam
from dfbench.algorithms.gradient_based.optax_sign import OptaxSignSGD, OptaxSignum
from dfbench.algorithms.gradient_based.optax_sm3 import OptaxSM3
from dfbench.algorithms.gradient_based.optax_lbfgs import OptaxLBFGS

__all__ = [
    "EvoxES",
    "EvoxPSO",
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
