"""Integration tests on a real problem.

For every algorithm in the uniform registry we run a short optimisation
on ``VoyagerProblem`` and check it produces a finite loss. This catches
problems that the mock quadratic cannot — JAX-vmap incompatibilities,
JIT recompilation issues, gradient explosions on real landscapes, etc.

These tests are slow and require GPU/Differometor;
"""

from __future__ import annotations

import numpy as np
import pytest

from dfbench.core.objective import Objective
from tests.test_algorithms_uniform import REGISTRY, AlgoSpec


pytestmark = pytest.mark.slow


# Algorithms known to fail on VoyagerProblem for reasons unrelated to the
# algorithm itself (e.g. internal vmap incompatibility with the
# differometor simulate's integer indexing). They are still parametrised
# but xfailed with an explicit reason.
KNOWN_VOYAGER_FAILURES = {
    "ReSTIR": "ReSTIR vmap warmup is incompatible with VoyagerProblem's "
              "differometor simulate() integer indexing under JAX vmap",
}


def _params():
    out = []
    for spec in REGISTRY:
        marks = []
        reason = KNOWN_VOYAGER_FAILURES.get(spec.cls.__name__)
        if reason is not None:
            marks.append(pytest.mark.xfail(reason=reason, strict=False))
        out.append(pytest.param(spec, id=spec.cls.__name__, marks=marks))
    return out


@pytest.fixture(scope="module")
def voyager_problem():
    from dfbench.problems import VoyagerProblem
    return VoyagerProblem()


@pytest.mark.parametrize("spec", _params())
def test_voyager_finite_loss(spec: AlgoSpec, voyager_problem):
    obj = Objective(voyager_problem, max_time=30, max_evals=50)
    spec.cls().optimize(
        obj,
        random_seed=42,
        **spec.extra_kwargs(50),
    )
    assert obj.best_loss is not None
    assert np.isfinite(float(obj.best_loss))
    assert len(obj.loss_history) > 0
