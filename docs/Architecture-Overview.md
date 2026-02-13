# Architecture Overview

## Module Map

```
src/dfbench/
├── __init__.py               # Public API surface
├── core/
│   ├── _init_env.py          # Pre-import environment setup (matplotlib, HPC)
│   ├── algorithm.py          # OptimizationAlgorithm ABC + AlgorithmType enum
│   ├── config.py             # CLI argument parser helper
│   ├── objective.py          # Objective wrapper (central piece)
│   ├── problem.py            # ContinuousProblem ABC
│   └── utils.py              # torch↔jax conversion, inverse sigmoid
├── algorithms/
│   ├── evolutionary/         # RandomSearch, EvoxPSO, EvoxES
│   ├── gradient_based/       # AdamGD, SAGD, NAAdamGD, LBFGSGD
│   ├── surrogate_based/      # BotorchBO, BotorchTuRBO, ReSTIR
│   └── generative/           # VAESampling
├── problems/
│   ├── base_problem.py       # OpticalSetupProblem (shared optics logic)
│   ├── voyager/              # VoyagerProblem, ConstrainedVoyagerProblem
│   └── uifo/                 # RandomUIFOProblem
└── benchmark/
    ├── benchmark.py          # Benchmark orchestrator, AlgorithmConfig
    └── metrics.py            # Per-run, aggregation, and multi-run metrics
```

---

## Separation of Concerns

The framework is organised around **three strict boundaries**:

### 1. Problem Layer (`core/problem.py`, `problems/`)

A problem defines *what* is being optimised. Every problem subclasses `ContinuousProblem` and exposes:

| Attribute | Purpose |
|-----------|---------|
| `objective_function` | Loss in **bounded** parameter space — used by evolutionary / surrogate algorithms |
| `sigmoid_objective_function` | Loss in **unbounded** space via sigmoid transform — used by gradient methods |
| `bounds` | `(2, n_params)` lower / upper limits |
| `optimization_pairs` | `[(component, property), …]` mapping each parameter index to a Differometor component |

**Rationale — two objective functions:** Gradient-based optimisers work best in unconstrained $(-\infty, +\infty)$ space where gradients flow smoothly through every point. Population-based methods, on the other hand, naturally respect bound constraints by sampling and clamping. Providing both variants lets each family operate in its natural domain without adapter code in every algorithm.

### 2. Objective Layer (`core/objective.py`)

`Objective` is the **sole interface** between any algorithm and its problem. It transparently:

- Dispatches to the correct objective function (bounded or sigmoid)
- Pre-compiles `jax.grad`, `jax.value_and_grad`, and `jax.vmap` variants
- Records every evaluation with aligned loss / gradient / params / timestamp histories
- Enforces wall-clock time and evaluation-count budgets
- Provides deterministic random sampling via a splittable JAX PRNG

**Rationale — why a wrapper instead of bare functions?** Without it, every algorithm would need to independently implement timing, budget checks, history logging, checkpointing, and bounded↔unbounded transforms. This both duplicates code and makes cross-algorithm comparison unreliable because each implementation might measure time or count evaluations slightly differently.

### 3. Algorithm Layer (`core/algorithm.py`, `algorithms/`)

An algorithm defines *how* to search. Every algorithm subclasses `OptimizationAlgorithm` and provides:

| Attribute / Method | Purpose |
|--------------------|---------|
| `algorithm_str` | Unique identifier (e.g. `"adam_gd"`, `"evox_cmaes"`) |
| `algorithm_type` | One of `GRADIENT_BASED`, `EVOLUTIONARY`, `SURROGATE_BASED`, `GENERATIVE` |
| `optimize(problem_objective, …)` | Main entry point — receives a pre-configured `Objective`, runs the loop, returns it |

Algorithms **never** create their own `Objective`; they receive one from the caller (or from the `Benchmark` harness). This inversion of control ensures the harness can set budget limits, select seeds, and configure history storage uniformly.

---

## Data Flow

```
                      Algorithm.optimize()
                            │
         ┌──────────────────┼──────────────────────┐
         │                  │                       │
   obj.value(p)    obj.value_and_grad(p)   obj.vmap_value(batch)
         │                  │                       │
         └──────────┬───────┘───────────────────────┘
                    │
             Objective._func / _value_and_grad_func / _vmap_func
                    │
                    ▼
         ┌─────────────────────┐
         │  problem.objective  │   (or sigmoid variant)
         │  _function(params)  │
         └─────────┬───────────┘
                   │
                   ▼
           Differometor.simulate()
                   │
                   ▼
            scalar loss value
                   │
       ┌───────────┼───────────────────┐
       │           │                   │
  _log_time()  _log_evals()     _log_to_file()
       │           │                   │
       ▼           ▼                   ▼
  _time_steps  _loss_history      periodic NPZ
               _params_history    checkpoint
               _grad_history
               _best_loss / _best_params
```

Every call to `obj.value()`, `obj.value_and_grad()`, or any `vmap_*` variant follows this exact pipeline. The algorithm receives the computed result; the logging is a side-effect invisible to the caller.

---

## Benchmark Orchestration

```
Benchmark.run()
    │
    ├─ for each AlgorithmConfig:
    │     ├─ for each run (1 … n_runs):
    │     │     ├─ Create Objective (with budget, seed)
    │     │     ├─ algorithm.optimize(objective, **hyperparams)
    │     │     └─ RunData.from_objective(objective)
    │     └─ AlgorithmRunData (list of RunData)
    │
    ├─ for each AlgorithmRunData:
    │     └─ _evaluate_algorithm() → BenchmarkResult
    │           └─ at each time sample t:
    │                 • slice histories at t
    │                 • compute per-run metrics
    │                 • aggregate across runs
    │
    └─ Save CSV + optional NPZ run data
```

The `Benchmark` class:

1. **Injects dependencies** — creates `Objective` instances with uniform budget and seed settings, then passes them to each algorithm.
2. **Collects raw data** — `RunData.from_objective()` extracts numpy arrays from each finished `Objective`.
3. **Evaluates at time checkpoints** — metrics are computed at `n_time_samples` evenly-spaced wall-clock times so algorithms with different per-evaluation costs remain comparable.
4. **Supports reload** — `save_run_data=True` persists raw histories to NPZ; `load_from=` re-evaluates metrics without re-running algorithms.

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **JAX for physics, PyTorch for some algorithms** | Differometor is a JAX project. Some optimisation libraries (EvoX, BoTorch) require PyTorch tensors. The `t2j` / `j2t` utilities bridge the gap with negligible overhead. |
| **Sigmoid bounding for gradient methods** | Gradient descent in clipped-bounded space produces zero gradients at boundaries. The sigmoid map $\sigma(x) \cdot (\text{ub} - \text{lb}) + \text{lb}$ keeps gradients nonzero everywhere. |
| **Wall-clock time as primary budget** | Evaluation cost varies across problems (12 ms for Voyager, 500 ms for UIFO). Time-based budgets make cross-problem comparisons meaningful. |
| **Time-sampled metrics** | Evaluating metrics at fixed time points (not iteration counts) normalises for per-eval cost differences between algorithms. |
| **Atomic checkpoints** | Long HPC jobs are killed without warning. Writing to `.tmp.npz` and then calling `os.replace` avoids half-written files. |
| **`_init_env.py` setting `MPLCONFIGDIR`** | On shared HPC filesystems, matplotlib's default config directory may be read-only. Setting a temp directory before any import prevents cryptic crashes. |
| **`AlgorithmType` enum** | The benchmark uses the type to decide whether an algorithm should receive an `unbounded=True` objective (gradient-based) or `unbounded=False` (everything else), removing a common source of misconfiguration. |
| **Reduced history properties** | Batched algorithms produce `(batch, …)` shaped histories. The `*_reduced` properties collapse each batch to a single representative (argmin of loss) so downstream analysis code never needs to handle ragged shapes. |
