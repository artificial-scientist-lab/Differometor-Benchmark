# Differometor Benchmark — Wiki

**dfbench** is a benchmarking framework for comparing optimization algorithms on gravitational-wave detector design problems built on top of the [Differometor](https://github.com/artificial-scientist-lab/Differometor) simulator.

The framework provides a standardised `Objective` wrapper that sits between every algorithm and every problem so that new contributors can focus entirely on their optimization logic while getting fair, reproducible, and fully-tracked runs for free.

---

## Quick Navigation

| Page | What you'll find |
|------|-----------------|
| [Architecture Overview](Architecture-Overview) | High-level design, module map, and data-flow diagram |
| [Installation](Installation) | Environment setup with `uv` or `pip`, GPU support |
| [Objective API Reference](Objective-API-Reference) | Complete reference for the `Objective` wrapper class |
| [Problems](Problems) | Available optimization problems and how they work |
| [Algorithms](Algorithms) | Catalogue of built-in algorithms and their parameters |
| [Implementing a New Algorithm](Implementing-a-New-Algorithm) | Step-by-step tutorial for contributors |
| [Benchmarking](Benchmarking) | Running benchmarks, metrics, and result analysis |
| [Metrics Reference](Metrics-Reference) | Detailed description of every benchmark metric |
| [Utilities & Helpers](Utilities-and-Helpers) | Conversion functions, CLI config, environment init |
| [FAQ](FAQ) | Common pitfalls and answers |

---

## Overivew of the Algorithm Architecture

```
OptimizationAlgorithm
         │
         ▼
   ┌───────────┐      records losses, params, grads, timestamps
   │ Objective │ ──►  enforces time / eval budgets
   └─────┬─────┘      bounded ↔ unbounded sigmoid transform
         │
         ▼
  ContinuousProblem        (VoyagerProblem, VoyagerTuningProblem, UIFOProblem, …)
         │
         ▼
  Differometor Simulator   (JAX-based interferometer physics)
```

Every algorithm talks **only** to `Objective`. Every problem implements `ContinuousProblem`. The `Benchmark` harness orchestrates multiple runs and computes standardised metrics.

---

## Quickstart

### Standalone (no algorithm class needed)

```python
from dfbench import Objective
from dfbench.problems import VoyagerProblem

problem = VoyagerProblem()
obj = Objective(problem, unbounded=True, max_time=120)

obj.set_seed(42)
obj.warmup_value_and_grad()   # JIT warmup
obj.start_logging()

params = obj.random_params_unbounded()
while not obj.budget_exceeded:
    loss, grad = obj.value_and_grad(params)
    params = params - 0.1 * grad

print(obj.best_loss, obj.best_params_bounded)
```

### Using a built-in algorithm

```python
from dfbench import Objective
from dfbench.problems import VoyagerProblem
from dfbench.algorithms import AdamGD

problem = VoyagerProblem()
obj = Objective(problem, max_time=120, verbose=1)

optimizer = AdamGD()
obj = optimizer.optimize(problem_objective=obj, learning_rate=0.1, random_seed=42)

print(obj.best_loss, obj.best_params_bounded)
```

**A loss below 0 means the optimized detector beats the real Voyager design's sensitivity.**

---

## Project Context

Differometor is a differentiable frequency-domain interferometer simulator. This benchmark exists to answer the question: *Which optimization strategy finds the best gravitational-wave detector designs, and how quickly?*

Because the simulator is written in JAX, every problem is automatically differentiable, batchable via `jax.vmap`, and JIT-compilable. The benchmark exploits all three properties.

See the [Differometor README](https://github.com/artificial-scientist-lab/Differometor) for details about the physics simulator itself.
