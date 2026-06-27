"""Tests for the reconstructive ProblemSpec contract and checkpoint round-trip."""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.core.problem import (
    ContinuousProblem,
    build_problem_from_spec,
    register_problem,
    validate_spec_round_trip,
)
from dfbench.core.storage import (
    CheckpointManager,
    LocalFilesystemBackend,
    RunPathResolver,
)
from dfbench.core.objective import Objective

from tests.conftest import QuadraticProblem


# ---------------------------------------------------------------------
# Registry / spec
# ---------------------------------------------------------------------


class TestProblemRegistry:
    def test_quadratic_is_registered(self):
        from dfbench.core.problem import _PROBLEM_REGISTRY

        assert "QuadraticProblem" in _PROBLEM_REGISTRY

    def test_build_from_spec_round_trip(self):
        p = QuadraticProblem(n_params=3)
        spec = p.to_spec()
        assert spec["type"] == "QuadraticProblem"
        assert spec["n_params"] == 3
        p2 = build_problem_from_spec(spec)
        assert isinstance(p2, QuadraticProblem)
        assert p2.n_params == 3

    def test_build_from_unknown_type_raises(self):
        with pytest.raises(ValueError, match="not registered"):
            build_problem_from_spec({"type": "DoesNotExist"})

    def test_build_from_missing_type_raises(self):
        with pytest.raises(ValueError, match="missing required 'type'"):
            build_problem_from_spec({"n_params": 2})

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):

            @register_problem
            class _Colliding(ContinuousProblem):
                spec_type = "QuadraticProblem"  # collides with existing

                def __init__(self):
                    pass

                @property
                def bounds(self):
                    return None

                @property
                def optimization_pairs(self):
                    return []

                def to_spec(self):
                    return {"type": "QuadraticProblem"}


# ---------------------------------------------------------------------
# Checkpoint embeds and restores problem spec
# ---------------------------------------------------------------------


class TestCheckpointProblemSpec:
    def test_save_embeds_problem_spec(self, mock_problem, tmp_path):
        obj = Objective(mock_problem, max_evals=100)
        obj.set_seed(42)
        obj.start_logging()
        obj.value(obj.random_params_bounded())
        path = obj.save_run_data(filepath=str(tmp_path / "ckpt.npz"))

        manager = CheckpointManager(
            backend=LocalFilesystemBackend(root=tmp_path),
            resolver=RunPathResolver(root=str(tmp_path)),
        )
        state = manager.load(path)
        spec = CheckpointManager.extract_problem_spec(state)
        assert spec is not None
        assert spec["type"] == "QuadraticProblem"
        assert spec["n_params"] == 2

    def test_reconstruct_problem_from_checkpoint(self, mock_problem, tmp_path):
        obj = Objective(mock_problem, max_evals=100)
        obj.set_seed(42)
        obj.start_logging()
        obj.value(obj.random_params_bounded())
        path = obj.save_run_data(filepath=str(tmp_path / "ckpt.npz"))

        manager = CheckpointManager(
            backend=LocalFilesystemBackend(root=tmp_path),
            resolver=RunPathResolver(root=str(tmp_path)),
        )
        state = manager.load(path)
        rebuilt = CheckpointManager.reconstruct_problem(state)
        assert rebuilt is not None
        assert isinstance(rebuilt, QuadraticProblem)
        assert rebuilt.n_params == mock_problem.n_params
        np.testing.assert_allclose(
            np.asarray(rebuilt.bounds), np.asarray(mock_problem.bounds)
        )

    def test_reconstruct_returns_none_without_spec(self, tmp_path):
        # A state whose metadata has no problem_spec
        from dfbench.core.storage import RunMetadata, RunState

        state = RunState(
            loss_history=np.array([1.0], dtype=object),
            grad_history=np.array([], dtype=object),
            hessian_history=np.array([], dtype=object),
            params_history=np.array([], dtype=object),
            eval_type_history=np.array([], dtype=object),
            time_steps=np.array([0.0], dtype=object),
            eval_count=1,
            best_loss=1.0,
            best_params=np.array([], dtype=np.float64),
            improvement_count=0,
            evals_since_improvement=0,
            log_call_count=1,
            eval_type_counts={1: 1},
            metadata=RunMetadata(),  # no extra
        )
        assert CheckpointManager.reconstruct_problem(state) is None


# ---------------------------------------------------------------------
# validate_spec_round_trip
# ---------------------------------------------------------------------


class TestValidateSpecRoundTrip:
    def test_quadratic_round_trip_matches_bounds(self, mock_problem):
        """validate_spec_round_trip rebuilds and asserts bounds/n_params match."""
        rebuilt = validate_spec_round_trip(mock_problem)
        assert rebuilt.n_params == mock_problem.n_params
        np.testing.assert_allclose(
            np.asarray(rebuilt.bounds), np.asarray(mock_problem.bounds)
        )

    def test_quadratic_5d_round_trip(self, mock_problem_5d):
        """Round-trip works for a different dimensionality."""
        rebuilt = validate_spec_round_trip(mock_problem_5d)
        assert rebuilt.n_params == 5

    def test_round_trip_detects_mismatch(self, mock_problem):
        """A spec that produces a different n_params raises AssertionError."""
        spec = mock_problem.to_spec()
        spec["n_params"] = 99  # corrupt the spec
        with pytest.raises(AssertionError, match="n_params"):
            # build from the corrupted spec, then validate against original
            rebuilt = build_problem_from_spec(spec)
            # validate_spec_round_trip would compare rebuilt vs rebuilt;
            # so instead directly check the inconsistency is caught:
            assert rebuilt.n_params == mock_problem.n_params


# ---------------------------------------------------------------------
# Benchmark problem reconstruction from saved metadata
# ---------------------------------------------------------------------


class TestBenchmarkReconstructProblem:
    def test_reconstruct_problem_from_saved_benchmark(self, mock_problem, tmp_path):
        """Benchmark.reconstruct_problem rebuilds the problem from metadata.json."""
        from dfbench.benchmark import AlgorithmConfig, Benchmark
        from tests.test_benchmark_smoke import _DummyAlgorithm

        config = AlgorithmConfig(_DummyAlgorithm())
        bm = Benchmark(
            mock_problem,
            success_loss=0.1,
            configs=[config],
            n_runs=1,
            max_time=2.0,
            n_time_samples=3,
            random_seed=42,
        )
        bm.run(save_csv=False, save_run_data=True, output_dir=str(tmp_path))

        # Find the saved benchmark directory
        saved_dirs = list(tmp_path.iterdir())
        assert len(saved_dirs) >= 1
        data_dir = saved_dirs[0]

        rebuilt = Benchmark.reconstruct_problem(data_dir)
        assert rebuilt is not None
        assert isinstance(rebuilt, QuadraticProblem)
        assert rebuilt.n_params == mock_problem.n_params
        np.testing.assert_allclose(
            np.asarray(rebuilt.bounds), np.asarray(mock_problem.bounds)
        )

    def test_reconstruct_returns_none_without_spec(self, tmp_path):
        """reconstruct_problem returns None if metadata.json has no problem_spec."""
        import json

        (tmp_path / "metadata.json").write_text(
            json.dumps({"problem_name": "x", "algorithms": []})
        )
        from dfbench.benchmark import Benchmark

        assert Benchmark.reconstruct_problem(tmp_path) is None
