"""Optimization algorithms."""

from dfbench.algorithms.evolutionary.evox_es import EvoxES
from dfbench.algorithms.evolutionary.evox_pso import EvoxPSO
from dfbench.algorithms.evolutionary.random_search import RandomSearch
from dfbench.algorithms.evolutionary.pycma_cmaes import (
    PyCMACMAES,
    PyCMAActiveCMAES,
    PyCMAIPOP,
    PyCMABIPOP,
)
from dfbench.algorithms.evolutionary.cmaes_sep_cma import CMAESSepCMA
from dfbench.algorithms.evolutionary.evosax_es import EvosaxMAES, EvosaxLMMAES
from dfbench.algorithms.evolutionary.jax_es import JAXOnePlusOneES, JAXMuLambdaES
from dfbench.algorithms.gradient_based.adam_gd import AdamGD
from dfbench.algorithms.gradient_based.lbfgs_gd import LBFGSGD
from dfbench.algorithms.gradient_based.na_adam_gd import NAAdamGD
from dfbench.algorithms.gradient_based.sa_gd import SAGD
from dfbench.algorithms.surrogate_based.botorch_bo import BotorchBO
from dfbench.algorithms.surrogate_based.botorch_turbo import BotorchTuRBO
from dfbench.algorithms.surrogate_based.restir import ReSTIR
from dfbench.algorithms.generative.vae_sampling import VAESampling

__all__ = [
    "EvoxES",
    "EvoxPSO",
    "RandomSearch",
    "PyCMACMAES",
    "PyCMAActiveCMAES",
    "PyCMAIPOP",
    "PyCMABIPOP",
    "CMAESSepCMA",
    "EvosaxMAES",
    "EvosaxLMMAES",
    "JAXOnePlusOneES",
    "JAXMuLambdaES",
    "AdamGD",
    "LBFGSGD",
    "NAAdamGD",
    "SAGD",
    "BotorchBO",
    "BotorchTuRBO",
    "ReSTIR",
    "VAESampling",
]
