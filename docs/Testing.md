# Testing

The test suite lives under `tests/`. It is organised so that the things
that should hold for *every* algorithm are checked once, in one place,
against every algorithm, and everything else is a focused module that
exercises a specific piece of code.

This page documents the conventions we follow when writing or extending
tests, what the existing suite covers, and the rationale behind a few
choices that are not obvious from the code.

## Layout

```
tests/
├── conftest.py                       shared fixtures + mock problem
├── test_algorithms_uniform.py        cross-algorithm baseline (registry-driven)
├── test_algorithm_protocol.py        the OptimizationAlgorithm ABC
├── test_algorithms_unit.py           algorithm-specific helpers, knobs, and edge cases
├── test_scipy_wrapper.py             SciPyObjectiveAdapter behaviour
├── test_objective_*.py               Objective invariants
├── test_problem_*.py                 Problem invariants
├── test_benchmark_smoke.py           Benchmark orchestration
└── slow/
    └── test_algorithms_integration.py    real-problem (Voyager) integration
```

The `slow/` directory contains tests marked `@pytest.mark.slow`; they are
excluded from the default `pytest` run and intended for the cluster.

## The uniform algorithm suite

`tests/test_algorithms_uniform.py` is the single source of truth for the
contract every `OptimizationAlgorithm` must satisfy. It is driven by a
registry (one `AlgoSpec` per algorithm) and parametrises the same
set of checks over every entry.

For each algorithm the suite verifies:

* the `algorithm_str` and `algorithm_type` are well-formed and match the
  registry's expectation,
* a short `optimize()` call on the mock quadratic produces evaluations,
  a finite `best_loss`, and a non-empty `loss_history`,
* every entry in `loss_history` is finite,
* `best_loss` equals `min(loss_history)` (i.e. tracking is consistent),
* `time_steps` is monotonically non-decreasing,
* the eval budget is respected up to a generous slack (twice the
  budget; this allows for batched algorithms whose last block overruns
  by one block),
* `obj.unbounded` after the run matches what the algorithm declares,
* `obj.algorithm_str` matches the algorithm's class attribute,
* `obj.best_params_bounded` lies inside the problem's box, and
* for algorithms that work in native bounded space, the raw
  `obj.best_params` also lies inside the box.

A small subset of algorithms (the ones whose internal RNG is fully
captured by the seed we pass in) are additionally checked for
reproducibility (same seed, same loss trajectory).

### Adding a new algorithm

Append one entry to `REGISTRY` in
[tests/test_algorithms_uniform.py](../tests/test_algorithms_uniform.py).
Use the `AlgorithmType` matching the algorithm's `src/dfbench/algorithms/`
subfolder.
All tests then run against your algorithm automatically:

```python
REGISTRY: list[AlgoSpec] = [
    ...
    AlgoSpec(MyAlgorithm, AlgorithmType.GRADIENT_BASED, unbounded=True),
]
```

If your algorithm needs extra kwargs to make progress under a small
budget (BoTorch is the canonical example: it counts iterations, not
evaluations), pass an `extra_kwargs` builder:

```python
AlgoSpec(
    MyAlgorithm,
    AlgorithmType.SURROGATE_BASED,
    unbounded=False,
    extra_kwargs=lambda max_evals: {"n_initial": min(5, max_evals // 4)},
)
```

If your algorithm is reproducibly broken in the current environment
(e.g. an upstream torch/JAX bug rather than your code), register it with
an `xfail` reason. Do not add `pytest.mark.xfail` decorators on the
tests themselves; the decorators belong on the registry entry, where
they describe the algorithm rather than the test.

```python
AlgoSpec(
    BrokenAlgo, AlgorithmType.EVOLUTIONARY, unbounded=False,
    xfail="Upstream torch.compile aliasing bug, see #123.",
)
```

Use `skip` instead of `xfail` only when running the algorithm would
crash the interpreter or hang the suite.

### What does *not* belong in the uniform suite

Anything that only applies to one algorithm. Examples that already live
in `test_algorithms_unit.py`:

* the SAGD transition-probability formula,
* SAM's `rho` parameter or Lookahead's `inner_optimizer_name`,
* Dogleg requiring a dense Hessian,
* the SR1 wrapper passing a ``scipy.optimize.SR1`` strategy.

One class per algorithm, named after the algorithm.

## Writing tests

A few conventions that make the suite predictable.

### Use `mock_problem`, not real problems

The mock problem in `tests/conftest.py` is a 2-parameter quadratic with
sigmoid bounding. It is fast, deterministic, and has a known optimum at
the origin. Use it for everything except integration tests that
explicitly need a real problem (Voyager, UIFO, ...); those go under
`tests/slow/` and are run on the cluster.

```python
def test_my_algorithm(mock_problem):
    obj = Objective(mock_problem, max_evals=30, max_time=60)
    MyAlgorithm().optimize(obj, random_seed=42)
    ...
```

### Always seed

Every test that runs an algorithm should pass `random_seed=...` (we use
`42` by convention). Tests without seeds are flaky and waste time.

### Use small budgets

Default budgets in the suite are `max_evals=30`, `max_time=60`. Larger
budgets are fine when the test needs them (the Optax improvement
subset uses 50 evals), but most invariants do not require more than
a handful of steps.

### Assert the contract, not the algorithm

Tests in the uniform suite assert things every `OptimizationAlgorithm`
should satisfy. Tests in algorithm-specific files assert things unique
to that algorithm. If you find yourself writing the same assertion
against several algorithms, lift it into the uniform suite; that is
the whole point.

### Failure messages should identify the algorithm

When parametrising over algorithm classes, give pytest an `id` so the
test name carries the algorithm name:

```python
@pytest.mark.parametrize("cls", MY_LIST, ids=lambda c: c.__name__)
def test_something(cls, mock_problem):
    ...
```

### Slow tests

Anything that needs Differometor, GPU, or a real problem goes in
`tests/slow/` and gets `pytestmark = pytest.mark.slow` at module level.
The default `pytest` run does not pick these up. We run them
periodically on the cluster:

```
srun -p a100-galvani --gres=gpu:1 --time=0-00:50 \
    pytest tests/slow -m slow
```

## Running the suite

```
pytest                                  # fast tests
pytest -m slow                          # slow tests only (run via srun)
pytest tests/test_algorithms_uniform.py # uniform algorithm tests
pytest -k "MyAlgorithm"                 # one algorithm, all uniform tests
pytest --collect-only -q                # list tests without running
```

The fast tests finish in roughly two minutes on a laptop CPU; the
uniform algorithm tests account for most of that time.

## On xfails

xfails are recorded with a one-line reason that explains *what* is
broken and *where* the bug lives. We use `strict=False` so a passing
xfail does not break CI. When the upstream bug is fixed, the next
person who notices the `XPASS` removes the mark.

We do not use xfail to silence flakiness. If a test is flaky, the test
is wrong; fix the test or the algorithm.

## On determinism

Most JAX-based algorithms are deterministic given a seed. PyTorch-based
algorithms are not, because of nondeterministic CUDA kernels and the
way `torch.compile` reorders work. Algorithms whose determinism we
actually exercise are listed in `DETERMINISTIC_ALGORITHMS` in the
uniform suite; if you add an algorithm whose seed reproducibly fixes
the trajectory, add it there too.

## Current status

As of the latest sweep on `main`, the fast-test suite reports
**1308 passed, 15 skipped, 11 xfailed** for
`pytest tests/test_algorithms_uniform.py tests/test_algorithms_unit.py
tests/test_bo_batch.py tests/test_dfo_algorithms.py`.

The non-pass results all have a single root cause each:

* **11 xfailed: all `EvoxES`.** The 11 uniform-suite test cases applied
  to `EvoxES` are all marked xfail because the default variant
  (CMA-ES) trips a `torch.compile` / dynamo aliasing bug inside
  `evox` on torch >= 2.6. The bug is upstream; the other `EvoxES`
  variants are exercised in their own dedicated tests.
* **15 skipped** = **12 + 3**:
  * **12** are `ARCJAX` across the same uniform-suite tests; `ARCJAX`
    is intentionally exposed but raises `NotImplementedError`. Its
    expected-failure behaviour is covered in
    `tests/test_custom_jax_batch.py` instead.
  * **3** are in `tests/test_bo_batch.py`, gated on optional
    dependencies that are not installed in the default environment:
    `ax-platform` (for `AxSAASBO`), `HEBO`, and `SMAC3`. Installing
    those packages re-enables the corresponding tests.

There are no unexpected failures; every skip and xfail carries an
explanatory reason in the registry / `pytest.mark.skipif`.
