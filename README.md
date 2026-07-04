# Differometor Benchmark

A benchmarking framework for optimization algorithms on gravitational-wave detector design problems, built on top of the [Differometor](https://github.com/artificial-scientist-lab/Differometor) simulator.

> **For detailed documentation, see the [Wiki](docs/Home.md).**

## TL;DR (I want to try my own algorithm)

This is how to create a raw script that tests your algorithm logic on a problem. Adding an algorithm as a class to the codebase is really not harder than this which would result in easier hyperparam testing (through short scripts you could then create) and the ability to add it to the benchmarking tool. But start from a script like that as you can copy that logic into the class later on.

All you need is the `Objective` wrapper. It handles evaluation tracking, budget enforcement, and history logging, you just write the optimization logic.

```python
from dfbench import Objective
from dfbench.problems import VoyagerProblem

# Pick a problem
problem = VoyagerProblem()

# Wrap that problem inside the Objective wrapper for loss and time tracking
obj = Objective(problem, unbounded=True, max_time=120, max_evals=1000)

# JIT warmup (doesn't count against budget)
obj.warmup_value_and_grad()

# Start logging loss and time
obj.start_logging()

# Your optimization loop, that's it.
params = obj.random_params_unbounded()
while not obj.budget_exceeded:

    # --- Your Optimization here ---
    loss, grad = obj.value_and_grad(params) # for example
    params = params - 0.1 * grad  # or any update rule

    # No need to log losses or params.

print(f"Best loss: {obj.best_loss}")
print(f"Best params: {obj.best_params_bounded}")
obj.plot_loss()
obj.save_run_to_file("my_run.npz")
```

**A loss below 0 means your solution beats the real Voyager detector's sensitivity.** (On `VoyagerProblem` without physical constraints — you might be burning mirrors.)

### Evaluation Methods

The problems are JAX-based and differentiable up to second order. Use whichever method fits your algorithm:

- `obj.value(params)`
- `obj.value_and_grad(params)`
- `obj.grad(params)`
- `obj.hessian(params)`
- `obj.value_grad_and_hessian(params)`
- `obj.vmap_value(batch)`
- `obj.vmap_value_and_grad(batch)`
- `obj.vmap_grad(batch)`
- `obj.vmap_hessian(batch)`
- `obj.vmap_value_grad_and_hessian(batch)`

### PyTorch Users

```python
from dfbench import t2j, j2t

params_jax = t2j(params_torch)       # Torch → JAX
losses_torch = j2t(obj.vmap_value(params_jax))  # JAX → Torch
```

This adds negligible overhead compared to the objective function itself.

### Available Problems

| Problem | Speed | Notes |
|---------|-------|-------|
| `VoyagerProblem` | ~12 ms/eval (A100) | Lightweight optimization of the Voyager Setup, good for prototyping, not physics-constrained. Loss < 0 achievable. |
| `VoyagerTuningProblem` | ~12 ms/eval (A100) | Tuning-only Voyager optimization (6 parameters on key mirrors). Lightweight and good for quick experiments. |
| `ConstrainedVoyagerProblem` | ~25 ms/eval (A100) | The same setup but physically constrained. Loss < 0 very difficult. |
| `UIFOProblem` | ~500 ms/eval (A100) | Full 3x3 UIFO setup (constrained). Loss < 0 hard but doable. |

Both constrained problems accept a `power_penalty_fn(value, threshold)` callable to control how power-constraint violations are penalized.  Built-in presets: `squashed_relu_penalty` (default), `relu_penalty`, `zero_penalty`. Feel free to try own ones.

All problems also support `bounds_overrides` (e.g. `{"tuning": (0, 45)}`) to narrow default property bounds, and expose `problem.print_bounds()` to inspect effective bounds.

See [Problems](docs/Problems.md) for details on loss computation, parameter meanings, and constraints.

---

## Installation

### From PyPI

```bash
pip install dfbench                    # Core package and Differometor problems
pip install "dfbench[optax,scipy]"      # Common local optimizers
pip install "dfbench[evolution]"        # CMA, EvoX, Nevergrad, Evosax
pip install "dfbench[bo]"               # BoTorch/Ax surrogate optimizers
pip install "dfbench[dfo,smac]"         # Derivative-free and SMAC optimizers
pip install "dfbench[all]"              # All optimizer backends
pip install "dfbench[cuda13]"           # CUDA 13 JAX support
pip install "dfbench[analysis]"         # Notebook/profiling tools
```

### From Source With `uv` (recommended for development)

[uv](https://uv.dev/) handles virtual environments and dependency resolution automatically.

```bash
uv sync                                  # CPU-only
uv sync --group cuda13                   # With GPU support (cuda12 also possible)
uv sync --group analysis                 # With analysis tools (profiling, notebooks)
uv sync --group cuda13 --group analysis  # Everything
```

### From Source With `pip`

```bash
pip install -e .                         # CPU-only editable install
pip install -e ".[all,cuda13,analysis]"  # All backends, CUDA 13, and analysis extras
```

See [Installation](docs/Installation.md) for GPU setup details and HPC notes.

---

## Architecture

```
OptimizationAlgorithm.optimize()
         │
         ▼
   ┌───────────┐      records losses, params, grads, timestamps
   │ Objective │ ──►  enforces time / eval budgets
   └─────┬─────┘      bounded ↔ unbounded sigmoid transform
         │
         ▼
  ContinuousProblem        (VoyagerProblem, VoyagerTuningProblem, ConstrainedVoyagerProblem, UIFOProblem)
         │
         ▼
  Differometor Simulator   (JAX-based interferometer physics)
```

**Design Idea:** Algorithms never create their own `Objective`, they receive a pre-configured one. This lets the benchmark harness or user script control budgets, seeds, and history settings uniformly. The algorithm only has to implement its optimization logic.

See [Architecture Overview](docs/Architecture-Overview.md) for full design details.

---

## Project Structure

```
src/dfbench/
├── core/
│   ├── problem.py        # ContinuousProblem ABC + ProblemSpec
│   ├── algorithm.py       # OptimizationAlgorithm ABC + AlgorithmType enum
│   ├── objective.py       # Objective wrapper (central piece)
│   └── utils.py           # torch↔jax conversion, inverse sigmoid
├── algorithms/
│   ├── derivative_free/   # OMADS, PDFO/Py-BOBYQA, NelderMead, Powell
│   ├── global_search/     # SciPy BasinHopping, DualAnnealing
│   ├── evolutionary/      # RandomSearch, EvoxPSO, EvoxES, Nevergrad, CMA family
│   ├── gradient_based/
│   │   ├── optax/         # 34 Optax-based optimizers (OptaxAdam, OptaxLAMB, ...)
│   │   ├── scipy/         # 13 SciPy-based optimizers (BFGS, TNC, SLSQP, …)
│   │   ├── custom_jax.py  # Native-JAX custom/hybrid batch (SGLD, ASAM, GD→L-BFGS, …)
│   │   └── *.py           # Custom-loop algorithms (AdamGD, LBFGSGD, SAGD, NAAdamGD, OptaxLBFGS)
│   ├── surrogate_based/
│   │   ├── botorch/       # BotorchBO, BotorchTuRBO, BotorchqNEI, BotorchqKG,
│   │   │                  #   REMBO, GEBO, LineBO  (+ shared _botorch_common.py)
│   │   ├── ax_baxus.py / ax_saasbo.py     # Ax/BoTorch high-dim BO
│   │   ├── hebo_bo.py / smac_bo.py        # External BO packages
│   │   ├── turbo_lbfgs.py                  # TuRBO + L-BFGS refinement
│   │   └── restir.py                       # GPU-native kNN surrogate
│   └── generative/        # VAESampling
├── problems/
│   ├── voyager/           # VoyagerProblem, VoyagerTuningProblem, ConstrainedVoyagerProblem
    └── uifo/             # UIFOProblem
└── benchmark/
    ├── benchmark.py       # Benchmark orchestrator
    └── metrics.py         # Metric computation functions
```

---

## Quick Start

### Running a Single Algorithm

```python
from dfbench import Objective
from dfbench.problems import VoyagerProblem
from dfbench.algorithms import AdamGD

problem = VoyagerProblem()

# The caller creates the Objective with budget and tracking settings
obj = Objective(problem, max_time=120, max_evals=50000, verbose=1)

# The algorithm receives the Objective and mutates it in place
optimizer = AdamGD()
optimizer.optimize(
    objective=obj,
    learning_rate=0.1,
    patience=1000,
    random_seed=42,
)

# Access results
print(f"Best loss: {obj.best_loss}")
print(f"Best params: {obj.best_params_bounded}")
print(f"Evaluations: {obj.eval_count}")
obj.plot_loss()  # Also saves JSONs of losses and best params
```

### Running a Benchmark

The `Benchmark` class handles `Objective` creation, seed management, and metric computation automatically.

```python
from dfbench.problems import VoyagerProblem
from dfbench.benchmark import Benchmark, AlgorithmConfig
from dfbench.algorithms import AdamGD, RandomSearch, EvoxES

problem = VoyagerProblem()

configs = [
    AlgorithmConfig(AdamGD(), {"learning_rate": 0.1}, name="Adam"),
    AlgorithmConfig(RandomSearch(batch_size=100), name="Random"),
    AlgorithmConfig(EvoxES(variant="CMAES"), {"pop_size": 100}, name="CMA-ES"),
]

benchmark = Benchmark(
    problem=problem,
    success_loss=0.1,
    configs=configs,
    n_runs=20,
    max_time=300,
)

results = benchmark.run(save_csv=True, save_run_data=True)
benchmark.print_summary(results)
```

- `save_csv`: Writes a CSV with all metrics computed at evenly-spaced time points.
- `save_run_data`: Persists raw loss/params/time histories to NPZ files for later re-evaluation.

See [Benchmarking](docs/Benchmarking.md) for full configuration options and [Metrics Reference](docs/Metrics-Reference.md) for what gets computed.

### Native-JAX Custom/Hybrid Batch

The gradient-based package now also includes native-JAX custom/hybrid classes:

- `SGLDJAX`, `ASAMJAX`, `AdamToLBFGSJAX`, `EntropySGDJAX`, `SGHMCJAX`
- `OGDJAX`, `OAdamJAX`, `PerturbedGDJAX`, `NoisyAdamJAX`
- `GDRestartsJAX`, `GaussianSmoothingGDJAX`

`ARCJAX` is currently exposed but intentionally raises `NotImplementedError`
to fail loudly until a stable, benchmark-fair ARC implementation is available.

All of the above default to unbounded optimization mode and rely on
`Objective` for logging/budget tracking. For a ready-to-run benchmark example,
see `scripts/voyager_native_jax_custom_batch.py`.

---

## How to Add an Algorithm (as a Class)

The interface is designed to make this as simple as possible. You write the optimization logic; `Objective` handles everything else (timing, logging, budget enforcement, file I/O).

> **Full step-by-step tutorial:** [Implementing a New Algorithm](docs/Implementing-a-New-Algorithm.md)

### The Contract

1. Subclass `OptimizationAlgorithm`
2. Declare `algorithm_str` and `algorithm_type`
3. Implement `optimize(objective, ...) → None`
4. Use `Objective` for all function evaluations
5. The `Objective` is mutated in place, thereby no return is needed

Please create a branch called `algorithm/my-algo` for the pull request.

### Minimal Template

```python
import secrets
import numpy as np
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm, AlgorithmType
from dfbench import Objective


class MyAlgorithm(OptimizationAlgorithm):
    """My optimization algorithm."""

    algorithm_str = "my_algorithm"
    algorithm_type = AlgorithmType.EVOLUTIONARY  # match one of the algorithms/ subfolders

    def __init__(self, batch_size: int = 50) -> None:
        """Algorithm-level meta-parameters that don't change between runs."""
        self.batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        max_iterations: int | None = None,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        patience: int = 1000,
        **kwargs,
    ) -> None:
        # 1. Setup + seed all RNGs
        obj = objective
        random_seed, key = self.prepare(
            obj,
            unbounded=False,
            random_seed=random_seed,
        )
        torch.manual_seed(random_seed)  # for frameworks beyond np/jax

        # 3. Initialize parameters
        if init_params is None:
            params = obj.random_params_bounded(n_samples=self.batch_size)
        else:
            params = init_params

        # 4. JIT warmup (before start_logging, compilation time is free)
        obj.warmup_vmap_value(batch_size=self.batch_size)

        # 5. Start the clock
        obj.start_logging()

        # 6. Optimization loop
        iteration = 0
        while not obj.budget_exceeded:
            if max_iterations is not None and iteration >= max_iterations:
                break

            losses = obj.vmap_value(params)  # automatically logged

            # ... your update logic here ...
            key, subkey = jax.random.split(key)

            if obj.evals_since_improvement > patience:
                break
            iteration += 1

        # 7. Done, Objective is mutated in place
```

### Key Points

- **`__init__` takes only algorithm meta-parameters** (batch size, network architecture, etc.), not the problem, not the budget.
- **`optimize()` receives a pre-configured `Objective`**, the algorithm does not create it.
- **`prepare()`** is called as `prepare(obj, unbounded, random_seed, algorithm_str=None, **kwargs)`. It configures the Objective space mode and algorithm identifier, seeds `np.random` and JAX, and returns `(random_seed, key)`. For PyTorch-based algorithms, call `torch.manual_seed(random_seed)` afterwards.
- **Choose `unbounded`:** Set to `True` if your algorithm benefits from smooth unconstrained space (via sigmoid transform). Most evolutionary, derivative-free, global-search, and surrogate methods use `False` (bounded space).
- **JIT warmup before `start_logging()`**, compilation time doesn't count against the budget. Use the matching `warmup_*()` helper; the single-point helpers take no arguments, while batched helpers take the batch size you will use during optimization.
- **`budget_exceeded`** checks both time and eval limits, please use it as your loop condition.

### Evaluation Methods

| Method | When to use | What gets logged |
|--------|-------------|------------------|
| `obj.value(params)` | Loss only | loss, params |
| `obj.value_and_grad(params)` | Gradient-based optimization | loss, grad, params |
| `obj.grad(params)` | Gradient only (rare) | grad, params, **no loss** |
| `obj.hessian(params)` | Exact second-order information | hessian, params, **no loss** |
| `obj.value_grad_and_hessian(params)` | Newton-style / second-order methods | loss, grad, hessian, params |
| `obj.vmap_value(batch)` | Population evaluation | batch losses, batch params |
| `obj.vmap_value_and_grad(batch)` | Batched gradient methods | batch losses, grads, params |
| `obj.vmap_hessian(batch)` | Batched second-order methods | batch hessians, batch params |
| `obj.vmap_value_grad_and_hessian(batch)` | Batched second-order methods | batch losses, grads, hessians, params |
| `obj.log_evaluation(...)` | Custom JIT'd loop | whatever you pass, including optional Hessians |

### Register It

Add your import to `src/dfbench/algorithms/<category>/__init__.py` and `src/dfbench/algorithms/__init__.py`.

### The Objective Wrapper

`Objective` handles all tracking transparently. Here's what's available:

```python
# Budget checking
while not obj.budget_exceeded:        # main loop condition
    if obj.evals_since_improvement > patience:
        break                          # early stopping

# Random parameter generation
params = obj.random_params()                      # active bounded/unbounded space
params = obj.random_params_bounded()              # shape: (n_params,)
batch = obj.random_params_bounded(n_samples=100)  # shape: (100, n_params)
params = obj.random_params_unbounded()            # for unbounded space

# Results
obj.best_loss               # best (minimum) loss found
obj.best_params_bounded     # best params in physical (bounded) space
obj.eval_count              # total evaluations performed
obj.loss_history            # full loss history
obj.time_steps              # elapsed time at each evaluation
```

See [Objective API Reference](docs/Objective-API-Reference.md) for the complete interface.

---

## Built-in Algorithms

| Algorithm | Type | Key Strength |
|-----------|------|-------------|
| `AdamGD` | Gradient | Fast convergence on smooth landscapes |
| `SAGD` | Gradient | Escapes local minima via stochastic ascent |
| `NAAdamGD` | Gradient | Noise-based exploration with annealing |
| `LBFGSGD` | Gradient | Second-order curvature information |
| `BFGS`, `LBFGSB`, `NonlinearCG`, `NewtonCG` | Gradient | Classical SciPy gradient and quasi-Newton methods |
| `TrustNCG`, `TrustKrylov`, `TrustConstr`, `Dogleg`, `SR1` | Gradient | Trust-region and constrained SciPy methods |
| `TNC`, `SLSQP`, `COBYQA`, `COBYLA` | Gradient | Bounded physical-space SciPy solvers |
| `RandomSearch` | Evolutionary | Unbiased baseline, no hyperparameters |
| `EvoxPSO` | Evolutionary | Swarm intelligence, many variants (CLPSO, CSO, ...) |
| `EvoxES` | Evolutionary | CMA-ES, OpenES, XNES, and more (EvoX backend) |
| `PyCMACMAES` | Evolutionary | Vanilla CMA-ES (pycma backend) |
| `PyCMAActiveCMAES` | Evolutionary | Active CMA-ES with negative weight updates (pycma) |
| `PyCMAIPOP` | Evolutionary | IPOP-CMA-ES: increasing-population restarts (pycma) |
| `PyCMABIPOP` | Evolutionary | BIPOP-CMA-ES: bi-population restart strategy (pycma) |
| `CMAESCMA` | Evolutionary | Full-covariance CMA-ES (cmaes package) |
| `CMAESSepCMA` | Evolutionary | sep-CMA-ES with diagonal covariance (cmaes package) |
| `EvosaxMAES` | Evolutionary | Matrix Adaptation ES (evosax backend) |
| `EvosaxLMMAES` | Evolutionary | Limited-Memory MA-ES for high dimensions (evosax) |
| `JAXOnePlusOneES` | Evolutionary | (1+1)-ES with 1/5 rule, native JAX |
| `JAXMuLambdaES` | Evolutionary | (μ,λ)-ES with truncation selection, native JAX |
| `OmadsMADS`, `OmadsOrthoMADS` | Derivative-Free | MADS / OrthoMADS direct search (OMADS) |
| `PDFOUOBYQA`, `PDFONEWUOA`, `PDFOLINCOA`, `PyBOBYQA` | Derivative-Free | Powell-style trust-region DFO (PDFO + Py-BOBYQA) |
| `NelderMead`, `Powell` | Derivative-Free | SciPy classical simplex / direction-set search |
| `BasinHopping`, `DualAnnealing` | Global Search | SciPy stochastic global optimization |
| `NevergradOnePlusOne`, `NevergradTBPSA`, `NevergradNGOpt` | Evolutionary | Nevergrad rugged-landscape baselines |
| `BotorchBO` | Surrogate | Sample-efficient Bayesian Optimization |
| `BotorchTuRBO` | Surrogate | Trust-region BO for high dimensions |
| `BotorchqNEI`, `BotorchqKG` | Surrogate | Noise-aware / lookahead BoTorch acquisitions |
| `BAxUS`, `AxSAASBO` | Surrogate | High-dim Ax/BoTorch BO (subspace / sparse-axis) |
| `REMBO`, `GEBO`, `LineBO` | Surrogate | Embedding / gradient / line-search BO variants |
| `TuRBOLBFGS` | Surrogate | TuRBO basin-finding + L-BFGS refinement |
| `HEBO`, `SMAC` | Surrogate | External BO packages (HEBO, SMAC3) |
| `ReSTIR` | Surrogate | GPU-native kNN surrogate, scales to 100k+ candidates |
| `VAESampling` | Generative | Latent-space compression + BO |

See [Algorithms](docs/Algorithms.md) for hyperparameter details and usage examples.

---

## Examples

Execution scripts in `./scripts/`:
- `voyager_adam_gd.py`: single-algorithm run
- `voyager_benchmark.py`: full benchmark with multiple algorithms
- `voyager_cma_family.py`: all ten CMA-family algorithms on VoyagerProblem
- `voyager_scipy_benchmark.py`: SciPy gradient / trust / constrained batch

Reference implementations worth reading:
- `gradient_based/adam_gd.py`: gradient-based pattern (custom loop)
- `gradient_based/optax/adam.py`: Optax wrapper pattern (minimal subclass)
- `gradient_based/scipy/_common.py`: shared SciPy wrapper, caching, and budget handling
- `evolutionary/random_search.py`: simplest batched example
- `evolutionary/evox_es.py`: wrapping an external library (EvoX/PyTorch)
- `evolutionary/pycma_cmaes.py`: wrapping pycma (ask/tell, restart strategies)
- `evolutionary/jax_es.py`: native JAX ES without external library
- `surrogate_based/botorch/botorch_bo.py`: surrogate-based with BoTorch

---

## Wiki

For in-depth documentation beyond this README:

| Page | Content |
|------|---------|
| [Architecture Overview](docs/Architecture-Overview.md) | Design, module map, data-flow diagrams |
| [Objective API Reference](docs/Objective-API-Reference.md) | Complete `Objective` class reference |
| [Problems](docs/Problems.md) | Loss computation, parameter meanings, constraints |
| [Algorithms](docs/Algorithms.md) | All built-in algorithms with hyperparameters |
| [Implementing a New Algorithm](docs/Implementing-a-New-Algorithm.md) | Full step-by-step contributor tutorial |
| [Benchmarking](docs/Benchmarking.md) | Running benchmarks, saving/loading results |
| [Metrics Reference](docs/Metrics-Reference.md) | Every benchmark metric explained |
| [Utilities & Helpers](docs/Utilities-and-Helpers.md) | `t2j`/`j2t`, CLI config, inverse sigmoid |
| [Installation](docs/Installation.md) | Environment setup, GPU support, HPC notes |
| [FAQ](docs/FAQ.md) | Common pitfalls and troubleshooting |
