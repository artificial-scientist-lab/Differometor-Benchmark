# Differometor Benchmark

A benchmarking framework for optimization algorithms on gravitational wave detector problems.

Please see this external documentation as not complete yet. The internal documentation should be decent.

## TL;DR (I want to try my own algorithm)

It's simple! Just use `Objective` to wrap your optimization loop:

```python
import jax.numpy as jnp
from dfbench import Objective
from dfbench.problems import VoyagerProblem

# Pick a problem
problem = VoyagerProblem()

# Create objective wrapper (handles all tracking)
obj = Objective(problem, unbounded=True, max_time=120, max_evals=1000)

# Warmup JIT, then start logging
_ = obj.value(jnp.zeros(problem.n_params))
obj.start_logging()

# Your optimization loop, that's it.
while not obj.budget_exceeded:
    ...
    params = ...  # your algorithm loop
    loss = obj.value(params)  # automatically tracked
    ...

print(f"Best loss: {obj.best_loss}")
print(f"Best params: {obj.best_params}")

obj.plot_loss()
obj.save_run_to_file("my_run.npz")
```

**Losses below 0 mean your solution beats the real Voyager detector's sensitivity!** ...except when using `VoyagerProblem` where there are no physical constraints -- you could be burning mirrors for example.

### Objective functions
The problem is written in Jax and differntiable! This means there are multiple objective functions that can be used for various kinds of optimizations:
- `obj.value(params)`
- `obj.value_and_grad(params)`
- `obj.grad(params)`
- `obj.vmap_value(params_batch)`
- `obj.vmap_value_and_grad(params_batch)`
- `obj.vmap_grad(params_batch)`

### Torch Users
The objective function is Jax-based. So to convert arrays between packages, you can use integrated conversion functions:
```python
from dfbench import t2j, j2t

# Torch to Jax
params_jax = t2j(params_torch)
loss = obj.value(params_jax)

params_batch_jax = t2j(params_batch_torch)
losses = j2t(obj.vmap_value(params_batch_jax))
```
This adds negligible overhead compared to the execution time of the objective function.
### Available Problems

| Problem | Speed | Notes |
|---------|-------|-------|
| `VoyagerProblem` | ~12ms/eval (A100) | Lightweight, good for prototyping. Loss < 0 is achievable. |
| `ConstrainedVoyagerProblem` | ~25ms/eval (A100) | Adds physical constraints. Loss < 0 is very difficult. |
| `RandomUIFOProblem` | ~500ms/eval (A100) | Computationally intensive. Loss < 0 is hard but doable. |

## Installation
### With `uv`

I recommend using `uv` (https://uv.dev/) for managing the environment.

Basic install:
```bash
uv sync
```

With dev dependencies (testing, profiling, notebooks):
```bash
uv sync --group dev
```

With CUDA 12 support to use GPUs:
```bash
uv sync --group cuda12
```

With everything (CUDA + dev):
```bash
uv sync --group cuda12 --dev
```

### With `pip`

I didn't try installing it with `pip` yet but it should definitely work too:

Basic install:
```bash
pip install -e .
```

With everything:
```bash
pip install -e ".[cuda12,dev]"
```


## Project Structure

Everything lives in `src/dfbench/`:

```
src/dfbench/
├── core/
│   ├── protocols.py      <-- Base classes (OptimizationAlgorithm, ContinuousProblem)
│   ├── objective.py      <-- Objective wrapper for tracking & logging
│   └── utils.py          <-- Utility functions (torch<->jax conversion, etc.)
├── algorithms/
│   ├── evolutionary/     <-- Random search, EvoX ES/PSO
│   ├── gradient_based/   <-- Adam, SA-GD, NA-Adam
│   ├── surrogate_based/  <-- BoTorch BO, TuRBO
│   └── generative/       <-- VAE-based sampling
├── problems/
│   └── voyager/          <-- Voyager interferometer problem
└── benchmark/
    ├── benchmark.py      <-- Benchmarking logic
    └── metrics.py        <-- Metric computation functions
```

## Quick Start

### Running a Single Algorithm

```python
from dfbench.problems import VoyagerProblem
from dfbench.algorithms import AdamGD

# Our problem needing to be optimized
problem = VoyagerProblem()
# Our optimization algorithm
optimizer = AdamGD(problem, verbose=1)

# Call optimize() of the algorithm to return
# an object containing all logged run data
obj = optimizer.optimize(
    max_time=120,           # Time budget in seconds
    learning_rate=0.1,      # Algorithm hyperparams
    patience=1000,
)

# Access results
print(f"Best loss: {obj.best_loss}")
print(f"Best params: {obj.best_params_bounded}")
print(f"Evaluations: {obj.eval_count}")
```

### Running a Benchmark

```python
from dfbench.problems import VoyagerProblem
from dfbench.benchmark import Benchmark, AlgorithmConfig
from dfbench.algorithms import AdamGD, RandomSearch, EvoxES

problem = VoyagerProblem()

configs = [
    AlgorithmConfig(AdamGD(problem), {"learning_rate": 0.1}, name="Adam"),
    AlgorithmConfig(RandomSearch(problem), {"n_samples": 10000}, name="Random"),
    AlgorithmConfig(EvoxES(problem, variant="CMAES"), {"pop_size": 100}, name="CMA-ES"),
]

benchmark = Benchmark(
    problem=problem,
    success_loss=0.1,
    configs=configs,
    n_runs=20,
    max_time=300,
    n_time_samples=5
)

results = benchmark.run(save_csv=True, save_run_data=True)
benchmark.print_summary(results)
```
Clarification:
- `save_csv`: Save a csv that can get parsed where at every `time_sample` all benchmarking metrics got calculated for all algorithms.
- `save_run_data`: Save all loss and parameter logs of all algorithms to a directory using npz files.

## How to Add an Algorithm

Implementing an algorithm correctly into the code with all logging and returning requirements requires minimally more effort than just writing a script thanks to the .

### 1. Create your file

Put it in the right folder under `algorithms/`. For example:
```
src/dfbench/algorithms/evolutionary/my_algo.py
```

### 2. Implement the Algorithm

All algorithms must:
1. Inherit from `OptimizationAlgorithm`
2. Use the `Objective` wrapper for evaluation and tracking
3. Return the `Objective` instance from `optimize()`

```python
from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)
from dfbench import Objective

class MyAlgorithm(OptimizationAlgorithm):
    
    algorithm_str: str = "my_algo"  # unique identifier
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY
    
    def __init__(
        self,
        problem: ContinuousProblem,
        verbose: int = 0,
        save_params_history: bool = True,
        save_batched_losses: bool = True,    # For batching algorithms
        save_batched_params: bool = False,   # Off by default (memory heavy)
    ):
        self._problem = problem
        self._verbose = verbose
        self._save_params_history = save_params_history
        self._save_batched_losses = save_batched_losses
        self._save_batched_params = save_batched_params
    
    def optimize(
        self,
        max_time: float | None = None,
        verbose: int | None = None,
        print_every: int = 100,
        **kwargs,
    ) -> Objective:
        # Create Objective wrapper
        obj = Objective(
            self._problem,
            unbounded=False,  # True for gradient-based (sigmoid space)
            max_time=max_time,
            max_evals=10000,
            save_params_history=self._save_params_history,
            save_batched_losses_history=self._save_batched_losses,
            save_batched_history=self._save_batched_params,
            print_every=print_every,
            verbose=verbose if verbose is not None else self._verbose,
            algorithm_str=self.algorithm_str,
        )
        
        # Warmup JIT
        _ = obj.value(jnp.zeros(self._problem.n_params))
        
        # Start tracking time
        obj.start_logging()
        
        # Your optimization loop
        while not obj.budget_exceeded:
            params = ...  # generate/update params
            loss = obj.value(params)  # automatically tracked!
            
            # For batched evaluation:
            # losses = obj.vmap_value(params_batch)
            
            # For gradient-based:
            # loss, grad = obj.value_and_grad(params)
        
        return obj
```

### 3. The Objective Wrapper

The `Objective` class handles all tracking automatically:

```python
# Single evaluation
loss = obj.value(params)

# Batched evaluation (for evolutionary algorithms)
losses = obj.vmap_value(params_batch)  # shape: (batch_size, n_params)

# Gradient-based (uses sigmoid_objective_function internally)
loss, grad = obj.value_and_grad(params)

# Check budget
if obj.budget_exceeded:
    break

# Early stopping helper
if obj.evals_since_improvement > patience:
    break
```

**Properties:** See full reference below in [Objective Properties Reference](#objective-properties-reference).

### 4. Bounded vs Unbounded Optimization

**Bounded (`unbounded=False`):**
- Use `obj.value()` or `obj.vmap_value()`
- Params must be within `problem.bounds`
- Good for: evolutionary, surrogate-based algorithms

**Unbounded (`unbounded=True`):**
- Use `obj.value_and_grad()` for gradient-based optimization
- Params can be any real number (-∞, +∞)
- Sigmoid bounding applied internally
- Use `obj.best_params_bounded` to get final params in bounded space
- Good for: gradient descent methods

### 5. Register it

Add import to `src/dfbench/algorithms/<category>/__init__.py` and `src/dfbench/algorithms/__init__.py`.

## Examples
Execution scripts in `./scripts/`:
- `voyager_adam_gd.py` - simple single-algorithm run
- `voyager_benchmark.py` - full benchmark with multiple algorithms

Look at existing implementations:

- `adam_gd.py` - gradient-based pattern
- `random_search.py` - simplest batched example
- `evox_es.py` - wraps external library (EvoX)
- `botorch_bo.py` - surrogate-based with BoTorch
- `vae_sampling.py` - generative model + BO hybrid


## Objective Properties Reference

All properties of the `Objective` class, organized by category.

### Problem & Configuration

| Property | Type | Description |
|----------|------|-------------|
| `bounds` | `Array[2, n_params]` | Lower and upper bounds for parameters |
| `n_params` | `int` | Number of parameters in the optimization problem |
| `problem` | `ContinuousProblem` | The underlying optimization problem instance |

### Budget Tracking (Evaluations)

| Property | Type | Description |
|----------|------|-------------|
| `eval_count` | `int` | Total number of objective evaluations performed |
| `evals_left` | `int \| None` | Evaluations remaining before budget exceeded (None if no limit) |
| `evals_exceeded` | `bool` | Whether evaluation budget has been exceeded |
| `evals_progress_fraction` | `float` | Fraction of eval budget used (0.0 to 1.0) |

### Budget Tracking (Time)

| Property | Type | Description |
|----------|------|-------------|
| `time_elapsed` | `float` | Seconds since `start_logging()` was called |
| `time_left` | `float \| None` | Seconds remaining before time budget exceeded (None if no limit) |
| `time_exceeded` | `bool` | Whether time budget has been exceeded |
| `time_progress_fraction` | `float` | Fraction of time budget used (0.0 to 1.0) |
| `budget_exceeded` | `bool` | Whether **any** budget (time OR evals) has been exceeded |

### Best Results

| Property | Type | Description |
|----------|------|-------------|
| `best_loss` | `float \| None` | Best (minimum) loss found so far |
| `best_params` | `Array[n_params] \| None` | Parameters for best loss (raw, possibly unbounded) |
| `best_params_bounded` | `Array[n_params] \| None` | Best params transformed to bounded space (**use for final output**) |

### Current State

| Property | Type | Description |
|----------|------|-------------|
| `current_loss` | `float \| Array[batch]` | Most recent loss from last evaluation |
| `current_params` | `Array[n_params] \| Array[batch, n_params]` | Most recent params from last evaluation |

### History (Raw)

These return copies of internal lists. May contain batched entries if `save_batched_*` was enabled.

| Property | Type | Description |
|----------|------|-------------|
| `loss_history` | `list[float \| Array[batch]]` | All loss values computed |
| `grad_history` | `list[Array[n_params] \| Array[batch, n_params] \| None]` | All gradient values (if saved) |
| `params_history` | `list[Array[n_params] \| Array[batch, n_params]]` | All parameter values evaluated (raw) |
| `params_history_bounded` | `list[Array[...] \| None]` | Params history transformed to bounded space |
| `time_steps` | `list[float]` | Elapsed time at each evaluation |

### History (Reduced)

These **always** return non-batched values, safe for benchmark analysis. Batches are reduced by selecting the entry with minimum loss (or minimum gradient norm as fallback).

| Property | Type | Description |
|----------|------|-------------|
| `loss_history_reduced` | `list[float]` | Losses with batches reduced to `nanmin` |
| `params_history_reduced` | `list[Array[n_params] \| None]` | Params with batches reduced to `argmin(loss)` |
| `params_history_reduced_bounded` | `list[Array[n_params] \| None]` | Reduced params in bounded space |
| `grad_history_reduced` | `list[Array[n_params] \| None]` | Grads with batches reduced to `argmin(loss)` |

### Progress Counters

| Property | Type | Description |
|----------|------|-------------|
| `improvement_count` | `int` | Number of times a new best loss was found |
| `evals_since_improvement` | `int` | Evaluations since last improvement (useful for early stopping) |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `start_logging()` | `() -> None` | Start the optimization timer. Call before beginning optimization. |
| `reset()` | `() -> None` | Reset all history and counters. |
| `value(params)` | `(Array[n_params]) -> float` | Evaluate objective at params |
| `grad(params)` | `(Array[n_params]) -> Array[n_params]` | Compute gradient at params |
| `value_and_grad(params)` | `(...) -> (float, Array)` | Compute both value and gradient (efficient) |
| `vmap_value(params_batch)` | `(Array[batch, n_params]) -> Array[batch]` | Batched evaluation |
| `vmap_grad(params_batch)` | `(...) -> Array[batch, n_params]` | Batched gradient |
| `vmap_value_and_grad(params_batch)` | `(...) -> (Array, Array)` | Batched value and gradient |
| `plot_loss()` | `() -> Figure` | Plot loss history |
| `save_run_data(algo_name, filepath)` | `(...) -> Path` | Save to compressed NPZ file |

## About the Problems

The Problems are interferometer optimizations where we minimize sensitivity across a frequency range.

The parameters represent properties of components (laser power, mirror reflectivity, distance). Different setups have differnt components. 

Read more about and see the setups at the [Differometor Repository](https://github.com/artificial-scientist-lab/Differometor)!

- `VoyagerProblem` is using the voyager setup. 
- `ConstrainedVoyagerProblem` is the same setup but also regularizes the loss by physical constraints. 
- `RandomUIFOProblem` is the full "Quasi-universal interferometer" setup where specific components get chosen randomly each time (really, take a look at the README of the repo above). This problem has the same physical constraints.
- A Problem using a specific encoding for a specific UIFO is coming soon...

### About the Loss

For the two constrained problems, the loss is the order of magnitude of the difference in sensitivity between the current setup and the real voyager setup. This is the reason why achieving a loss below 0 for the `ConstrainedVoyagerProblem` is so hard. Due to the UIFO being overparameterized, it's easier.

## About ConstrainedVoyagerProblem


## Notes

- Everything is JAX-based, use `jax.jit` and `jax.vmap` for performance
- JIT warmup happens in the algorithm before `obj.start_logging()`
- For torch<->jax conversion, use `t2j` and `j2t` from `dfbench.core.utils`
- Default saving config: batched losses ON, batched params OFF (memory efficiency)

Let me know if something doesn't work or needs clarification!
