"""Section 12 — Package-level import smoke tests.

Tests 12.1–12.5: verify that all public API symbols are importable.
"""

import pytest


# 12.1
class TestCoreImports:
    def test_import_objective(self):
        from dfbench import Objective  # noqa: F401

    # 12.2
    def test_import_protocols(self):
        from dfbench import ContinuousProblem, OptimizationAlgorithm, AlgorithmType  # noqa: F401

    # 12.3
    def test_import_algorithms(self):
        from dfbench.algorithms import (
            AdamGD,
            SAGD,
            NAAdamGD,
            LBFGSGD,
            EvoxES,
            EvoxPSO,
            RandomSearch,
            BotorchBO,
            BotorchTuRBO,
            ReSTIR,
            VAESampling,
        )  # noqa: F401

    # 12.4
    def test_import_problems(self):
        from dfbench.problems import (
            VoyagerProblem,
            ConstrainedVoyagerProblem,
            UIFOProblem,
        )  # noqa: F401

    def test_import_random_uifo_alias(self):
        """RandomUIFOProblem backwards-compat alias is importable."""
        from dfbench.problems import RandomUIFOProblem  # noqa: F401

    # 12.5
    def test_import_benchmark(self):
        from dfbench.benchmark import Benchmark, AlgorithmConfig  # noqa: F401
