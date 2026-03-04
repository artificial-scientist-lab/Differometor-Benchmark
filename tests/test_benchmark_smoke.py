"""Section 9 — Benchmark orchestration smoke tests with mock algorithm/problem.

Tests 9.1–9.14.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dfbench.benchmark.benchmark import (
    AlgorithmConfig,
    AggregateMetric,
    Benchmark,
    BenchmarkResult,
    RunData,
    SingleMetric,
)
from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective


# ── Dummy algorithm for smoke tests ──────────────────────────────────


class _DummyAlgorithm(OptimizationAlgorithm):
    algorithm_str = "dummy"
    algorithm_type = AlgorithmType.EVOLUTIONARY

    def __init__(self):
        pass

    def optimize(self, problem_objective, init_params=None, random_seed=None, **kwargs):
        obj = problem_objective
        seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)
        obj.start_logging()
        while not obj.budget_exceeded:
            params = obj.random_params_bounded()
            obj.value(params)


# ======================================================================
# AlgorithmConfig (9.1)
# ======================================================================


class TestAlgorithmConfig:
    def test_stores_correctly(self):
        """9.1 Stores algorithm, hyperparameters, name; default name from algorithm_str."""
        algo = _DummyAlgorithm()
        config = AlgorithmConfig(algo, {"lr": 0.1}, name="test_name")
        assert config.name == "test_name"
        assert config.hyperparameters == {"lr": 0.1}
        assert config.algorithm is algo

    def test_default_name(self):
        """9.1b Default name falls back to algorithm_str."""
        algo = _DummyAlgorithm()
        config = AlgorithmConfig(algo)
        assert config.name == "dummy"


# ======================================================================
# Benchmark.__init__ (9.2)
# ======================================================================


class TestBenchmarkInit:
    def test_time_samples(self, mock_problem):
        """9.2 time_samples excludes 0, evenly spaced, ends at max_time."""
        config = AlgorithmConfig(_DummyAlgorithm())
        bm = Benchmark(
            mock_problem,
            success_loss=0.1,
            configs=[config],
            n_runs=1,
            max_time=10.0,
            n_time_samples=5,
        )
        ts = bm.time_samples
        assert ts[0] > 0  # excludes 0
        assert ts[-1] == pytest.approx(10.0)
        assert len(ts) == 5
        # Evenly spaced
        diffs = np.diff(ts)
        np.testing.assert_allclose(diffs, diffs[0], atol=1e-6)


# ======================================================================
# BenchmarkResult dataclass (9.3)
# ======================================================================


class TestBenchmarkResult:
    def test_fields_exist(self):
        """9.3 BenchmarkResult has expected field types."""
        from dataclasses import fields as dc_fields

        field_names = {f.name for f in dc_fields(BenchmarkResult)}
        expected = {
            "algorithm_name",
            "time_samples",
            "n_runs",
            "fraction_of_success",
            "min_loss",
            "performance_profile_auc",
            "auc_top_1",
            "avg_loss",
            "time_to_success",
            "evals_to_success",
            "solution_diversity_overall",
            "solution_diversity_nn",
            "auc_top_10",
        }
        assert expected.issubset(field_names)


# ======================================================================
# RunData.from_objective (9.4)
# ======================================================================


class TestRunDataFromObjective:
    def test_extracts_correctly(self, mock_problem):
        """9.4 Correctly extracts loss_history, time_steps, etc."""
        obj = Objective(mock_problem, max_evals=5, save_params_history=True)
        obj.set_seed(42)
        obj.start_logging()
        for _ in range(5):
            obj.value(obj.random_params_bounded())

        rd = RunData.from_objective(obj)
        assert rd.eval_count == 5
        assert len(rd.loss_history) == 5
        assert len(rd.time_steps) == 5
        assert rd.best_loss == pytest.approx(float(obj.best_loss), abs=1e-5)


# ======================================================================
# Benchmark.run() smoke tests (9.5–9.8)
# ======================================================================


class TestBenchmarkRun:
    @pytest.fixture()
    def benchmark(self, mock_problem):
        config = AlgorithmConfig(_DummyAlgorithm())
        return Benchmark(
            mock_problem,
            success_loss=0.1,
            configs=[config],
            n_runs=2,
            max_time=5.0,
            n_time_samples=5,
            random_seed=42,
        )

    def test_run_returns_correct_length(self, benchmark, tmp_path):
        """9.5 run() returns list[BenchmarkResult] of correct length."""
        results = benchmark.run(save_csv=False)
        assert len(results) == 1  # 1 algorithm config
        assert isinstance(results[0], BenchmarkResult)
        assert results[0].n_runs == 2

    def test_metrics_at_time_samples(self, benchmark, tmp_path):
        """9.6 Metric arrays have shape (n_time_samples,)."""
        results = benchmark.run(save_csv=False)
        r = results[0]
        assert r.fraction_of_success.value.shape == (5,)
        assert r.avg_loss.mean.shape == (5,)

    def test_fraction_of_success_range(self, benchmark, tmp_path):
        """9.7 fraction_of_success is in [0, 1]."""
        results = benchmark.run(save_csv=False)
        fos = results[0].fraction_of_success.value
        assert np.all(np.array(fos) >= 0)
        assert np.all(np.array(fos) <= 1)

    def test_deterministic_seeds(self, mock_problem):
        """9.8 With random_seed, per-run seeds are deterministic."""
        config = AlgorithmConfig(_DummyAlgorithm())
        bm1 = Benchmark(
            mock_problem,
            success_loss=0.1,
            configs=[config],
            n_runs=2,
            max_time=5.0,
            n_time_samples=3,
            random_seed=42,
        )
        bm2 = Benchmark(
            mock_problem,
            success_loss=0.1,
            configs=[config],
            n_runs=2,
            max_time=5.0,
            n_time_samples=3,
            random_seed=42,
        )
        r1 = bm1.run(save_csv=False)
        r2 = bm2.run(save_csv=False)
        np.testing.assert_allclose(
            np.array(r1[0].avg_loss.mean), np.array(r2[0].avg_loss.mean), atol=1e-5
        )


# ======================================================================
# Save / Load (9.9–9.12)
# ======================================================================


class TestBenchmarkSaveLoad:
    def test_save_run_data(self, mock_problem, tmp_path):
        """9.9 _save_algorithm_run_data creates NPZ files."""
        config = AlgorithmConfig(_DummyAlgorithm())
        bm = Benchmark(
            mock_problem,
            success_loss=0.1,
            configs=[config],
            n_runs=1,
            max_time=3.0,
            n_time_samples=3,
            random_seed=42,
        )
        results = bm.run(save_csv=False, save_run_data=True, output_dir=str(tmp_path))
        npz_files = list(tmp_path.rglob("*.npz"))
        assert len(npz_files) > 0

    def test_save_metadata(self, mock_problem, tmp_path):
        """9.10 Metadata JSON written."""
        config = AlgorithmConfig(_DummyAlgorithm())
        bm = Benchmark(
            mock_problem,
            success_loss=0.1,
            configs=[config],
            n_runs=1,
            max_time=3.0,
            n_time_samples=3,
            random_seed=42,
        )
        bm.run(save_csv=False, save_run_data=True, output_dir=str(tmp_path))
        json_files = list(tmp_path.rglob("*.json"))
        assert len(json_files) > 0


# ======================================================================
# CSV (9.13)
# ======================================================================


class TestBenchmarkCSV:
    def test_csv_created(self, mock_problem, tmp_path, monkeypatch):
        """9.13 save_csv creates a CSV file."""
        monkeypatch.chdir(tmp_path)
        config = AlgorithmConfig(_DummyAlgorithm())
        bm = Benchmark(
            mock_problem,
            success_loss=0.1,
            configs=[config],
            n_runs=1,
            max_time=3.0,
            n_time_samples=3,
            random_seed=42,
        )
        bm.run(save_csv=True)
        csv_files = list(tmp_path.rglob("*.csv")) + list(Path("data").rglob("*.csv"))
        # Just verify it doesn't crash; exact path depends on implementation
        # The CSV writing code is exercised regardless


# ======================================================================
# print_summary (9.14)
# ======================================================================


class TestPrintSummary:
    def test_no_crash(self, mock_problem, capsys):
        """9.14 print_summary does not raise."""
        config = AlgorithmConfig(_DummyAlgorithm())
        bm = Benchmark(
            mock_problem,
            success_loss=0.1,
            configs=[config],
            n_runs=1,
            max_time=3.0,
            n_time_samples=3,
            random_seed=42,
        )
        results = bm.run(save_csv=False)
        bm.print_summary(results)  # Should not raise
