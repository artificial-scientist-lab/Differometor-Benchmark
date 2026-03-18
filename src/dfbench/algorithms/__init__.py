"""Optimization algorithms."""

from dfbench.algorithms.derivative_free.basin_hopping import BasinHopping
from dfbench.algorithms.derivative_free.dual_annealing import DualAnnealing
from dfbench.algorithms.derivative_free.nelder_mead import NelderMead
from dfbench.algorithms.derivative_free.powell import Powell
from dfbench.algorithms.evolutionary.evox_es import EvoxES
from dfbench.algorithms.evolutionary.evox_pso import EvoxPSO
from dfbench.algorithms.evolutionary.random_search import RandomSearch
from dfbench.algorithms.gradient_based.adam_gd import AdamGD
from dfbench.algorithms.gradient_based.lbfgs_gd import LBFGSGD
from dfbench.algorithms.gradient_based.na_adam_gd import NAAdamGD
from dfbench.algorithms.gradient_based.sa_gd import SAGD
from dfbench.algorithms.surrogate_based.botorch_bo import BotorchBO
from dfbench.algorithms.surrogate_based.botorch_turbo import BotorchTuRBO
from dfbench.algorithms.surrogate_based.restir import ReSTIR
from dfbench.algorithms.generative.vae_sampling import VAESampling

__all__ = [
    "BasinHopping",
    "DualAnnealing",
    "NelderMead",
    "Powell",
    "EvoxES",
    "EvoxPSO",
    "RandomSearch",
    "AdamGD",
    "LBFGSGD",
    "NAAdamGD",
    "SAGD",
    "BotorchBO",
    "BotorchTuRBO",
    "ReSTIR",
    "VAESampling",
]
