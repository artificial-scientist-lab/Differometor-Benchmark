"""Uniform algorithm tests.

Every algorithm exported by ``dfbench.algorithms`` is parametrised through
the same set of tests. Family-specific quirks (extra kwargs, bounded vs.
unbounded search space, known-broken combinations) are encoded in the
registry below so the test bodies themselves stay identical.

Two goals drive this:

* Every algorithm is held to the same contract — it produces evaluations,
  leaves the ``Objective`` in a consistent state, respects bounds, etc.
* Adding a new algorithm requires nothing more than appending one entry to
  ``REGISTRY``; the full set of tests then runs against it automatically.

Algorithm-specific behaviour (e.g. the SciPy adapter cache, the SAGD
transition probability, the VAE forward pass) belongs in the focused test
modules and is out of scope here.

See ``docs/Testing.md`` for the conventions this file follows and the
process for extending it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pytest

from dfbench.algorithms import (
    AdamGD,
    BFGS,
    BotorchBO,
    BotorchTuRBO,
    COBYLA,
    COBYQA,
    Dogleg,
    EvoxES,
    EvoxPSO,
    LBFGSB,
    LBFGSGD,
    NAAdamGD,
    NewtonCG,
    NevergradNGOpt,
    NevergradOnePlusOne,
    NevergradTBPSA,
    NonlinearCG,
    OmadsMADS,
    OmadsOrthoMADS,
    OptaxAdaBelief,
    OptaxAdaDelta,
    OptaxAdaGrad,
    OptaxAdaMax,
    OptaxAdaMaxW,
    OptaxAdafactor,
    OptaxAMSGrad,
    OptaxAdam,
    OptaxAdamW,
    OptaxAdan,
    OptaxLAMB,
    OptaxLBFGS,
    OptaxLion,
    OptaxLookahead,
    OptaxNAG,
    OptaxNadam,
    OptaxNadamW,
    OptaxNoisySGD,
    OptaxNovoGrad,
    OptaxOAdam,
    OptaxOGD,
    OptaxPolyakSGD,
    OptaxRAdam,
    OptaxRMSProp,
    OptaxRProp,
    OptaxSAM,
    OptaxSGD,
    OptaxSGDM,
    OptaxSM3,
    OptaxScheduleFreeAdam,
    OptaxSignSGD,
    OptaxSignum,
    OptaxSophia,
    OptaxYogi,
    RandomSearch,
    ReSTIR,
    SAGD,
    SLSQP,
    SR1,
    TNC,
    TrustConstr,
    TrustKrylov,
    TrustNCG,
    VAESampling,
)
from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective


# ── Registry ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AlgoSpec:
    """Per-algorithm metadata for the uniform test parametrisation.

    Attributes:
        cls: The algorithm class (``OptimizationAlgorithm`` subclass).
        family: Expected ``algorithm_type``.
        unbounded: Expected value of ``obj.unbounded`` after ``optimize()``.
        extra_kwargs: Builder for any kwargs that ``optimize()`` needs to
            produce at least one evaluation under a small budget (e.g.
            BoTorch counts iterations rather than evals and needs an explicit
            ``n_initial``). Receives ``max_evals`` so kwargs can be scaled.
        skip: Optional reason; if set the algorithm is skipped entirely.
        xfail: Optional reason; if set the test is run but expected to fail.
            ``strict=False`` so an unexpected pass does not break CI.
    """

    cls: type[OptimizationAlgorithm]
    family: AlgorithmType
    unbounded: bool
    extra_kwargs: Callable[[int], dict[str, Any]] = field(default=lambda _: {})
    skip: str | None = None
    xfail: str | None = None


def _botorch_kwargs(max_evals: int) -> dict[str, Any]:
    """Return kwargs that keep BoTorch inside the test budget.

    BoTorch counts iterations rather than individual evaluations, so without
    explicit caps it will exceed a small ``max_evals`` budget on the very
    first call.
    """
    return {"max_iterations": 2, "n_initial": min(5, max(2, max_evals // 4))}


# Order: misc gradient → optax → scipy → evolutionary → surrogate → generative.
# To add a new algorithm, append one AlgoSpec entry here.
REGISTRY: list[AlgoSpec] = [
    # -- misc gradient-based --------------------------------------------------
    AlgoSpec(AdamGD,        AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(SAGD,          AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(NAAdamGD,      AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(LBFGSGD,       AlgorithmType.GRADIENT_BASED, unbounded=True),
    # -- optax ----------------------------------------------------------------
    AlgoSpec(OptaxAdam,            AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxAdamW,           AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxAdaBelief,       AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxAdafactor,       AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxAMSGrad,         AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxAdaGrad,         AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxAdaDelta,        AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxAdaMax,          AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxAdaMaxW,         AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxAdan,            AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxLion,            AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxLAMB,            AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxNadam,           AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxNadamW,          AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxRMSProp,         AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxRProp,           AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxRAdam,           AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxSGD,             AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxSGDM,            AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxNAG,             AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxNoisySGD,        AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxPolyakSGD,       AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxSAM,             AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxSophia,          AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxLookahead,       AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxScheduleFreeAdam, AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxYogi,            AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxNovoGrad,        AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxOGD,             AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxOAdam,           AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxSignSGD,         AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxSignum,          AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxSM3,             AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(OptaxLBFGS,           AlgorithmType.GRADIENT_BASED, unbounded=True),
    # -- scipy: unbounded (sigmoid-mapped) ------------------------------------
    AlgoSpec(BFGS,         AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(NonlinearCG,  AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(NewtonCG,     AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(TrustNCG,     AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(TrustKrylov,  AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(Dogleg,       AlgorithmType.GRADIENT_BASED, unbounded=True),
    # -- scipy: native bounds -------------------------------------------------
    AlgoSpec(LBFGSB,       AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(TrustConstr,  AlgorithmType.GRADIENT_BASED, unbounded=False),
    AlgoSpec(TNC,          AlgorithmType.GRADIENT_BASED, unbounded=True),
    AlgoSpec(SLSQP,        AlgorithmType.GRADIENT_BASED, unbounded=False),
    AlgoSpec(COBYQA,       AlgorithmType.GRADIENT_BASED, unbounded=False),
    AlgoSpec(COBYLA,       AlgorithmType.GRADIENT_BASED, unbounded=False),
    AlgoSpec(SR1,          AlgorithmType.GRADIENT_BASED, unbounded=False),
    # -- evolutionary ---------------------------------------------------------
    AlgoSpec(RandomSearch, AlgorithmType.EVOLUTIONARY, unbounded=False),
    AlgoSpec(
        EvoxES, AlgorithmType.EVOLUTIONARY, unbounded=False,
        xfail="EvoxES default variant (CMA-ES) hits a torch.compile / dynamo "
              "aliasing bug inside evox on torch >= 2.6. Tracked upstream; "
              "other EvoxES variants are exercised in their own tests.",
    ),
    AlgoSpec(EvoxPSO,      AlgorithmType.EVOLUTIONARY, unbounded=False),
    AlgoSpec(NevergradOnePlusOne, AlgorithmType.EVOLUTIONARY, unbounded=False),
    AlgoSpec(NevergradTBPSA,      AlgorithmType.EVOLUTIONARY, unbounded=False),
    AlgoSpec(NevergradNGOpt,      AlgorithmType.EVOLUTIONARY, unbounded=False),
    AlgoSpec(OmadsMADS,      AlgorithmType.EVOLUTIONARY, unbounded=False),
    AlgoSpec(OmadsOrthoMADS, AlgorithmType.EVOLUTIONARY, unbounded=False),
    # -- surrogate-based ------------------------------------------------------
    AlgoSpec(BotorchBO,    AlgorithmType.SURROGATE_BASED, unbounded=False,
             extra_kwargs=_botorch_kwargs),
    AlgoSpec(BotorchTuRBO, AlgorithmType.SURROGATE_BASED, unbounded=False,
             extra_kwargs=_botorch_kwargs),
    AlgoSpec(ReSTIR,       AlgorithmType.SURROGATE_BASED, unbounded=False),
    # -- generative -----------------------------------------------------------
    AlgoSpec(VAESampling,  AlgorithmType.GENERATIVE, unbounded=True),
]


# Algorithms that produce the same trajectory for the same seed.
# Population-based and async-compiled methods are excluded; their
# reproducibility is checked in their own focused tests where applicable.
DETERMINISTIC_ALGORITHMS = {
    AdamGD, NAAdamGD, LBFGSGD,
    OptaxAdam, OptaxSGD, OptaxLBFGS,
    BFGS, LBFGSB, NewtonCG,
    RandomSearch,
}


# ── Pytest plumbing ───────────────────────────────────────────────────


def _id(spec: AlgoSpec) -> str:
    return spec.cls.__name__


def _apply_marks(spec: AlgoSpec, *, run_required: bool = True) -> pytest.param:
    """Build a ``pytest.param`` with any skip/xfail marks from the spec.

    Pass ``run_required=False`` for tests that only inspect class metadata
    and never call ``optimize()``; this suppresses xfail marks that only
    apply at runtime.
    """
    marks = []
    if spec.skip is not None:
        marks.append(pytest.mark.skip(reason=spec.skip))
    if spec.xfail is not None and run_required:
        marks.append(pytest.mark.xfail(reason=spec.xfail, strict=False))
    return pytest.param(spec, id=_id(spec), marks=marks)


ALL_PARAMS = [_apply_marks(s) for s in REGISTRY]
ALL_PARAMS_STATIC = [_apply_marks(s, run_required=False) for s in REGISTRY]
GRADIENT_PARAMS = [_apply_marks(s) for s in REGISTRY
                   if s.family == AlgorithmType.GRADIENT_BASED]
EVOLUTIONARY_PARAMS = [_apply_marks(s) for s in REGISTRY
                       if s.family == AlgorithmType.EVOLUTIONARY]
SURROGATE_PARAMS = [_apply_marks(s) for s in REGISTRY
                    if s.family == AlgorithmType.SURROGATE_BASED]
GENERATIVE_PARAMS = [_apply_marks(s) for s in REGISTRY
                     if s.family == AlgorithmType.GENERATIVE]
DETERMINISTIC_PARAMS = [_apply_marks(s) for s in REGISTRY
                        if s.cls in DETERMINISTIC_ALGORITHMS]


# Budget used throughout. Small enough to keep the test run CPU-friendly.
DEFAULT_MAX_EVALS = 30
DEFAULT_MAX_TIME = 60.0


def _run(spec: AlgoSpec, mock_problem, *, max_evals: int = DEFAULT_MAX_EVALS,
         seed: int = 42) -> Objective:
    """Run ``optimize`` and return the resulting Objective.

    Shared by all test methods so that budget, seed, and problem are
    consistent and failures across tests are directly comparable.
    """
    obj = Objective(mock_problem, max_evals=max_evals, max_time=DEFAULT_MAX_TIME)
    algo = spec.cls()
    algo.optimize(obj, random_seed=seed, **spec.extra_kwargs(max_evals))
    return obj


# ── Tests ─────────────────────────────────────────────────────────────


class TestStaticMetadata:
    """Checks on class-level attributes that do not require running the algorithm."""

    @pytest.mark.parametrize("spec", ALL_PARAMS_STATIC)
    def test_algorithm_str_is_nonempty_string(self, spec: AlgoSpec):
        algo = spec.cls()
        assert isinstance(algo.algorithm_str, str)
        assert algo.algorithm_str

    @pytest.mark.parametrize("spec", ALL_PARAMS_STATIC)
    def test_algorithm_type_matches_registry(self, spec: AlgoSpec):
        assert spec.cls.algorithm_type == spec.family


class TestSmokeRun:
    """Basic contract checks: every algorithm must produce a usable Objective."""

    @pytest.mark.parametrize("spec", ALL_PARAMS)
    def test_eval_count_positive(self, spec: AlgoSpec, mock_problem):
        obj = _run(spec, mock_problem)
        assert obj.eval_count > 0

    @pytest.mark.parametrize("spec", ALL_PARAMS)
    def test_best_loss_finite(self, spec: AlgoSpec, mock_problem):
        obj = _run(spec, mock_problem)
        assert obj.best_loss is not None
        assert np.isfinite(float(obj.best_loss))

    @pytest.mark.parametrize("spec", ALL_PARAMS)
    def test_loss_history_non_empty(self, spec: AlgoSpec, mock_problem):
        obj = _run(spec, mock_problem)
        assert len(obj.loss_history) > 0

    @pytest.mark.parametrize("spec", ALL_PARAMS)
    def test_loss_history_finite(self, spec: AlgoSpec, mock_problem):
        obj = _run(spec, mock_problem)
        losses = np.asarray([float(l) for l in obj.loss_history if l is not None])
        assert losses.size > 0
        assert np.all(np.isfinite(losses))

    @pytest.mark.parametrize("spec", ALL_PARAMS)
    def test_best_loss_matches_history_minimum(self, spec: AlgoSpec, mock_problem):
        obj = _run(spec, mock_problem)
        losses = [float(l) for l in obj.loss_history if l is not None]
        assert float(obj.best_loss) == pytest.approx(min(losses), abs=1e-5)

    @pytest.mark.parametrize("spec", ALL_PARAMS)
    def test_time_steps_monotonic(self, spec: AlgoSpec, mock_problem):
        obj = _run(spec, mock_problem)
        ts = list(obj.time_steps)
        assert len(ts) > 0
        for prev, curr in zip(ts, ts[1:]):
            assert curr >= prev

    @pytest.mark.parametrize("spec", ALL_PARAMS)
    def test_budget_is_respected(self, spec: AlgoSpec, mock_problem):
        """Eval count must stay within 2× the budget.

        The slack covers batched algorithms (population, surrogate) that
        may overshoot by one full block at the end.
        """
        obj = _run(spec, mock_problem)
        assert obj.eval_count <= 2 * DEFAULT_MAX_EVALS

    @pytest.mark.parametrize("spec", ALL_PARAMS)
    def test_unbounded_flag_matches_registry(self, spec: AlgoSpec, mock_problem):
        obj = _run(spec, mock_problem)
        assert obj.unbounded is spec.unbounded

    @pytest.mark.parametrize("spec", ALL_PARAMS)
    def test_algorithm_str_recorded_on_objective(self, spec: AlgoSpec, mock_problem):
        obj = _run(spec, mock_problem)
        assert obj.algorithm_str == spec.cls().algorithm_str


class TestBoundsContract:
    """``best_params_bounded`` must lie inside the problem bounds.

    For sigmoid-mapped algorithms the Objective's inverse mapping guarantees
    this. For algorithms that receive native bounds directly, the algorithm
    itself is responsible. Either way, downstream callers see parameters
    inside the declared box.
    """

    @pytest.mark.parametrize("spec", ALL_PARAMS)
    def test_best_params_bounded_inside_box(self, spec: AlgoSpec, mock_problem):
        obj = _run(spec, mock_problem)
        bp = np.asarray(obj.best_params_bounded)
        bounds = np.asarray(mock_problem.bounds)
        assert np.all(bp >= bounds[0] - 1e-6)
        assert np.all(bp <= bounds[1] + 1e-6)

    @pytest.mark.parametrize(
        "spec",
        [_apply_marks(s) for s in REGISTRY if s.unbounded is False],
    )
    def test_native_bounded_raw_params_inside_box(self, spec: AlgoSpec, mock_problem):
        """For native-bounded algorithms, the raw ``best_params`` must also
        lie inside the box — there is no sigmoid postprocessing to fall back on.
        """
        obj = _run(spec, mock_problem)
        bp = np.asarray(obj.best_params)
        bounds = np.asarray(mock_problem.bounds)
        assert np.all(bp >= bounds[0] - 1e-6)
        assert np.all(bp <= bounds[1] + 1e-6)


class TestDeterminism:
    """Two runs with the same seed must produce the same loss trajectory.

    Limited to algorithms whose internal RNG is fully determined by the
    seed argument. Population-based and async-compiled backends are excluded.
    """

    @pytest.mark.parametrize("spec", DETERMINISTIC_PARAMS)
    def test_same_seed_same_trajectory(self, spec: AlgoSpec, mock_problem):
        obj1 = _run(spec, mock_problem, max_evals=15, seed=42)
        obj2 = _run(spec, mock_problem, max_evals=15, seed=42)
        h1 = np.asarray([float(l) for l in obj1.loss_history])
        h2 = np.asarray([float(l) for l in obj2.loss_history])
        assert h1.shape == h2.shape
        np.testing.assert_allclose(h1, h2, atol=1e-5)
