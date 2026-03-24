"""Optimization algorithms."""

from dfbench.algorithms.evolutionary.evox_es import EvoxES
from dfbench.algorithms.evolutionary.evox_pso import EvoxPSO
from dfbench.algorithms.evolutionary.random_search import RandomSearch
from dfbench.algorithms.gradient_based.adam_gd import AdamGD
from dfbench.algorithms.gradient_based.custom_jax import (
    ARCJAX,
    ASAMJAX,
    AdamToLBFGSJAX,
    EntropySGDJAX,
    GDRestartsJAX,
    GaussianSmoothingGDJAX,
    NoisyAdamJAX,
    OAdamJAX,
    OGDJAX,
    PerturbedGDJAX,
    SGHMCJAX,
    SGLDJAX,
)
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
    "AdamGD",
    "SGLDJAX",
    "ASAMJAX",
    "AdamToLBFGSJAX",
    "EntropySGDJAX",
    "SGHMCJAX",
    "ARCJAX",
    "OGDJAX",
    "OAdamJAX",
    "PerturbedGDJAX",
    "NoisyAdamJAX",
    "GDRestartsJAX",
    "GaussianSmoothingGDJAX",
    "LBFGSGD",
    "NAAdamGD",
    "SAGD",
    "BotorchBO",
    "BotorchTuRBO",
    "ReSTIR",
    "VAESampling",
]
