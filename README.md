# Differometor Benchmark

Quick overview of the project structure for adding new optimization algorithms.

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
│   └── protocols.py      <-- Base classes live here (OptimizationAlgorithm, ContinuousProblem)
├── algorithms/
│   ├── evolutionary/     <-- PSO, random search
│   ├── gradient_based/   <-- Adam, SA-GD
│   └── surrogate_based/  <-- Bayesian optimization
├── problems/
│   └── voyager_problem.py
└── benchmark/
    └── benchmark.py      <-- Benchmarking logic
```

## How to Add an Algorithm

### 1. Create your file

Put it in the right folder under `algorithms/`. For example:
```
src/dfbench/algorithms/evolutionary/my_algo.py
```

### 2. Inherit from OptimizationAlgorithm

Look at `src/dfbench/core/protocols.py` for the base class. The gist:

```python
from dfbench.core.protocols import (
    ContinuousProblem,
    OptimizationAlgorithm,
    AlgorithmType,
)

class MyAlgorithm(OptimizationAlgorithm):
    
    algorithm_str: str = "my_algo"  # unique identifier
    algorithm_type: AlgorithmType = AlgorithmType.EVOLUTIONARY
    
    def __init__(self, problem: ContinuousProblem, ...):
        self._problem = problem
        # setup stuff
    
    def optimize(self, wall_times=None, return_best_params_history=False, ...):
        # your optimization loop
        # return: (best_params, best_params_history, losses, wall_time_indices)
        pass
```

### 3. The two objective functions

The problem gives you two objective functions:

- `problem.objective_function` - expects params within bounds, use this for evolutionary/population-based stuff
- `problem.sigmoid_objective_function` - expects unbounded params (-inf, +inf), applies sigmoid internally. Use for gradient-based methods.

### 4. Wall time tracking

For benchmarking we track progress at specific wall times. The pattern is:

```python
from collections import deque
import time

wall_time_indices = []
wall_times_remaining = deque(sorted(wall_times))
max_wall_time = wall_times_remaining[-1]

start_time = time.time()
iteration = 0

while (time.time() - start_time) < max_wall_time:
    elapsed = time.time() - start_time
    
    # record which iteration we're at when we pass each checkpoint in wall_times
    while wall_times_remaining and elapsed >= wall_times_remaining[0]:
        wall_time_indices.append(iteration) # add current iteration to wall_time_indices
        wall_times_remaining.popleft()
    
    # ... do your optimization step ...
    iteration += 1
```

### 5. Register it

Add import to `src/dfbench/algorithms/<category>/__init__.py` and `src/dfbench/__init__.py`.

## Examples

Look at existing implementations:

- `random_search.py` - simplest example, good starting point
- `adam_gd.py` - gradient-based pattern
- `evox_pso.py` - more complex, wraps external library

## Running a Benchmark

```python
from dfbench import (
    MyAlgorithm,
    VoyagerProblem,
    Benchmark,
    AlgorithmConfig,
)

problem = VoyagerProblem()

configs = [
    AlgorithmConfig(MyAlgorithm(problem), {"hyperparam": value}, "MyAlgo-v1"),
]

benchmark = Benchmark(
    problem=problem,
    success_loss=0,
    configs=configs,
    n_runs=10,
    wall_time_steps=[30, 60, 120],
)

results = benchmark.run_benchmark()
benchmark.print_summary(results)
```

There's also `scripts/voyager_benchmark.py` with a working example.

## About VoyagerProblem

The `VoyagerProblem` is what we're currently optimizing - it's an interferometer setup where we're trying to minimize sensitivity across a frequency range.

**Bounds:** The problem has `bounds` property that's shape `[2, n_params]` - first row is lower bounds, second row is upper bounds. Each parameter has different ranges depending on what it is (reflectivity is 0-1, angles are -180 to 180, power/mass can be 0.01-200, etc.). Check `voyager_problem.py` to see which parameters are being optimized.

**Output function:** If you call `problem.output_to_files(best_params, losses, ...)` it'll dump JSON files with the params/losses and also generate plots (loss curve, sensitivity curve). Useful for debugging but you don't need it for benchmarking since the benchmark handles saving data.

## Notes

- Everything is JAX-based, use `jax.jit` and `jax.vmap` for performance
    - (Only if possible) Warmup JIT in `__init__`.
- If you need torch<->jax conversion, there's `t2j_numpy` and `j2t_numpy` in `dfbench.core.utils`. Don't use `t2j` and `j2t` for now, they turned out to be painfully slow.

Let me definitely know if something doesn't work, is unclear or needs refactoring (seems inconsistent etc.).
