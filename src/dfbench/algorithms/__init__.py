"""Optimization algorithms."""

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
from dfbench.algorithms.derivative_free.pdfo_uobyqa import PDFOUOBYQA
from dfbench.algorithms.derivative_free.pdfo_newuoa import PDFONEWUOA
from dfbench.algorithms.derivative_free.pdfo_lincoa import PDFOLINCOA
from dfbench.algorithms.derivative_free.pybobyqa import PyBOBYQA

__all__ = [
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
    "PDFOUOBYQA",
    "PDFONEWUOA",
    "PDFOLINCOA",
    "PyBOBYQA",
]
