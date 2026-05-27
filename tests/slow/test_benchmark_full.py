"""Section 13 — End-to-end benchmark integration (needs srun).

Marked @slow — must be run via srun on the cluster.
"""

from __future__ import annotations


import numpy as np
import pytest

from dfbench.benchmark.benchmark import AlgorithmConfig, Benchmark


pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def voyager_problem():
    from dfbench.problems import VoyagerProblem

    return VoyagerProblem()


class TestEndToEnd:
    def test_benchmark_two_algos(self, voyager_problem, tmp_path):
        """13.2 Full Benchmark.run() with 2 algorithms, 2 runs each."""
        from dfbench.algorithms import AdamGD, RandomSearch

        configs = [
            AlgorithmConfig(AdamGD(), name="Adam"),
            AlgorithmConfig(RandomSearch(), name="RS"),
        ]
        bm = Benchmark(
            voyager_problem,
            success_loss=0.1,
            configs=configs,
            n_runs=2,
            max_time=30.0,
            n_time_samples=5,
            random_seed=42,
        )
        results = bm.run(save_csv=True, output_dir=str(tmp_path))
        assert len(results) == 2
        for r in results:
            assert r.n_runs == 2

    def test_save_load_round_trip(self, voyager_problem, tmp_path):
        """13.3 save_run_data then load_from produces consistent results."""
        from dfbench.algorithms import RandomSearch

        configs = [AlgorithmConfig(RandomSearch(), name="RS")]
        bm = Benchmark(
            voyager_problem,
            success_loss=0.1,
            configs=configs,
            n_runs=2,
            max_time=30.0,
            n_time_samples=5,
            random_seed=42,
        )
        results_original = bm.run(
            save_csv=False, save_run_data=True, output_dir=str(tmp_path)
        )

        # _prepare_save_dir creates a timestamped subdirectory inside output_dir
        run_dirs = list(tmp_path.glob("*/metadata.json"))
        assert len(run_dirs) == 1, (
            f"Expected one run directory, found: {[p.parent for p in run_dirs]}"
        )
        run_dir = run_dirs[0].parent

        # Reload
        bm2 = Benchmark(
            voyager_problem,
            success_loss=0.1,
            configs=configs,
            n_runs=2,
            max_time=30.0,
            n_time_samples=5,
        )
        results_loaded = bm2.run(save_csv=False, load_from=str(run_dir))
        np.testing.assert_allclose(
            np.array(results_original[0].avg_loss.mean),
            np.array(results_loaded[0].avg_loss.mean),
            atol=1e-5,
        )
