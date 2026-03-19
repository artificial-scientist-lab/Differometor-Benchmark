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
]
