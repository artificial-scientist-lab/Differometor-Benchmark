"""Section 7 + 13 (integration) — Algorithm integration tests with real problems.

Marked @slow — must be run via srun on the cluster.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.core.objective import Objective


pytestmark = pytest.mark.slow


ALL_ALGORITHM_CLASSES = None  # Populated lazily below


def _get_algorithm_classes():
    global ALL_ALGORITHM_CLASSES
    if ALL_ALGORITHM_CLASSES is None:
        from dfbench.algorithms import (
            AdamGD,
            SAGD,
            NAAdamGD,
            LBFGSGD,
            RandomSearch,
            EvoxES,
            EvoxPSO,
            BotorchBO,
            BotorchTuRBO,
            ReSTIR,
            VAESampling,
        )

        ALL_ALGORITHM_CLASSES = [
            AdamGD,
            SAGD,
            NAAdamGD,
            LBFGSGD,
            RandomSearch,
            EvoxES,
            EvoxPSO,
            BotorchBO,
            BotorchTuRBO,
            ReSTIR,
            VAESampling,
        ]
    return ALL_ALGORITHM_CLASSES


@pytest.fixture(scope="module")
def voyager_problem():
    from dfbench.problems import VoyagerProblem

    return VoyagerProblem()


class TestAlgorithmIntegration:
    """13.1 For each algorithm: optimize on VoyagerProblem with small budget."""

    @staticmethod
    def _optimize_kwargs(algo_cls):
        """Return extra kwargs required by certain algorithm classes."""
        if algo_cls.__name__ in ("BotorchBO", "BotorchTuRBO"):
            return {"max_iterations": 10, "n_initial": 5}
        return {}

    @pytest.mark.parametrize(
        "algo_idx",
        [
            *range(9),
            pytest.param(
                9,
                marks=pytest.mark.xfail(
                    reason="ReSTIR vmap warmup is incompatible with VoyagerProblem's"
                    " differometor simulate() integer indexing under JAX vmap",
                    strict=False,
                ),
            ),
            10,
        ],
    )
    def test_optimize_finite_loss(self, voyager_problem, algo_idx):
        classes = _get_algorithm_classes()
        algo_cls = classes[algo_idx]
        algo = algo_cls()
        obj = Objective(voyager_problem, max_time=30, max_evals=50)
        algo.optimize(obj, random_seed=42, **self._optimize_kwargs(algo_cls))
        assert obj.best_loss is not None
        assert np.isfinite(float(obj.best_loss))
        assert len(obj.loss_history) > 0
