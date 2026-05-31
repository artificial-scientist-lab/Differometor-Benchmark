"""Section 6 — OptimizationAlgorithm ABC and AlgorithmType enum.

Tests 6.1–6.6.
"""

from __future__ import annotations

import inspect

import pytest

from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective


class TestAlgorithmType:
    def test_enum_members(self):
        """6.1 AlgorithmType exposes the folder-aligned members."""
        expected = {
            "GRADIENT_BASED",
            "EVOLUTIONARY",
            "DERIVATIVE_FREE",
            "GLOBAL_SEARCH",
            "SURROGATE_BASED",
            "GENERATIVE",
        }
        actual = {m.name for m in AlgorithmType}
        assert expected.issubset(actual)

    def test_prepare_signature_order(self):
        """6.1b prepare() keeps the documented argument order."""
        parameters = list(inspect.signature(OptimizationAlgorithm.prepare).parameters)
        assert parameters[:5] == [
            "self",
            "obj",
            "unbounded",
            "random_seed",
            "algorithm_str",
        ]

    def test_enum_values_are_strings(self):
        """6.2 All enum values are snake_case strings."""
        for member in AlgorithmType:
            assert isinstance(member.value, str)
            assert member.value == member.value.lower()


class TestOptimizationAlgorithmABC:
    def test_cannot_instantiate(self):
        """6.3 Cannot directly instantiate the ABC."""
        with pytest.raises(TypeError):
            OptimizationAlgorithm()

    def test_incomplete_subclass_fails(self):
        """6.4 Subclass without optimize() raises TypeError on instantiation."""

        class PartialAlgo(OptimizationAlgorithm):
            algorithm_str = "partial"
            algorithm_type = AlgorithmType.GRADIENT_BASED

            def __init__(self):
                pass

        with pytest.raises(TypeError):
            PartialAlgo()

    def test_prepare_returns_seed_and_key(self, mock_problem):
        """6.5 prepare() returns (int, jax.Array) and sets obj attributes."""

        class DummyAlgo(OptimizationAlgorithm):
            algorithm_str = "dummy"
            algorithm_type = AlgorithmType.GRADIENT_BASED

            def __init__(self):
                pass

            def optimize(self, objective, init_params=None, random_seed=None, **kw):
                pass

        algo = DummyAlgo()
        obj = Objective(mock_problem)
        seed, key = algo.prepare(obj, unbounded=False, random_seed=123)
        assert seed == 123
        assert key.shape == (2,) or key.shape == ()  # depends on JAX version
        assert obj.algorithm_str == "dummy"
        assert obj.unbounded is False

    def test_prepare_generates_seed_when_none(self, mock_problem):
        """6.6 prepare() with random_seed=None generates a seed."""

        class DummyAlgo(OptimizationAlgorithm):
            algorithm_str = "dummy2"
            algorithm_type = AlgorithmType.GRADIENT_BASED

            def __init__(self):
                pass

            def optimize(self, objective, init_params=None, random_seed=None, **kw):
                pass

        algo = DummyAlgo()
        obj = Objective(mock_problem)
        seed, key = algo.prepare(obj, unbounded=False, random_seed=None)
        assert isinstance(seed, int)
        assert algo._random_seed == seed
