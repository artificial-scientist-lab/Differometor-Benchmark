# FAQ and Troubleshooting

---

## Algorithm Implementation

### My gradient method diverges to NaN

1. **Learning rate too high.** Differometor loss landscapes are steep. Start with `lr=0.01` or lower.
2. **Not using unbounded mode.** Gradient-based methods must set `unbounded=True` so sigmoid bounding provides smooth gradients everywhere.
3. **Using `obj.grad()` instead of `obj.value_and_grad()`.** `grad()` does **not** log a loss — use `value_and_grad()` to get both.

### Bounded vs. unbounded — which do I use?

| Algorithm type | `unbounded` | Why |
|----------------|-------------|-----|
| Gradient-based | `True` | Sigmoid bounding gives smooth, unconstrained gradients. |
| Evolutionary | `False` | Populations naturally respect bound constraints. |
| Surrogate-based | `False` | GP/BO acquisitions work in bounded space. |
| Generative | Either | Depends on internal representation. |

The `Benchmark` harness sets this automatically based on `algorithm_type`.

### How do I convert between bounded and unbounded parameters?

```python
from dfbench import inverse_sigmoid_bounding

# bounded → unbounded
unbounded_params = inverse_sigmoid_bounding(bounded_params, problem.bounds)

# unbounded → bounded is done automatically during evaluation when obj.unbounded=True
```

### My algorithm uses PyTorch

```python
from dfbench import t2j, j2t

params_jax = t2j(params_torch)       # PyTorch → JAX
losses = obj.vmap_value(params_jax)
losses_torch = j2t(losses)            # JAX → PyTorch
```

The conversion goes through NumPy and adds negligible overhead.

---

## Benchmarking

### How many runs should I use?

At least 30 for basic statistics, 100+ for reliable confidence intervals. The default is 100.

### Can I re-evaluate metrics without re-running?

Yes. Save run data with `save_run_data=True`, then reload with different settings:

```python
benchmark = Benchmark(problem, success_loss=0.05, ..., n_runs=100, max_time=300)
results = benchmark.run(load_from="./data/benchmark_run_data/VoyagerProblem_...")
```

### What does `random_baseline_loss` do?

Enables **normalized AUC**: each run's AUC is divided by the AUC of a hypothetical constant-loss algorithm, then log-scaled. Set it to the expected loss from random parameter sampling.

---

## HPC and Environment

### matplotlib crashes with `PermissionError`

dfbench auto-redirects `MPLCONFIGDIR` to a temp directory at import time. Ensure you import dfbench before matplotlib:

```python
import dfbench        # sets MPLCONFIGDIR
import matplotlib     # now safe
```

### JAX uses too much GPU memory

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

This makes JAX allocate memory on demand instead of pre-allocating 75% of GPU memory.

### Multiple jobs fail on the same GPU node

Use SLURM's `--gres=gpu:1` to isolate GPU access, or set `CUDA_VISIBLE_DEVICES` to assign specific GPUs.

---

## Data and I/O

### Where are results saved?

| Data type | Default path |
|-----------|-------------|
| Benchmark CSV | `./data/benchmark_results/` |
| Run data (NPZ) | `./data/benchmark_run_data/` |
| Objective run data | `./data/objective_run_data/` |
| Problem outputs | `./data/problem_output/` |

### `Objective.save_run_data()` vs. benchmark saving

`Objective.save_run_data()` saves a single run (for development/debugging). The benchmark's `save_run_data` flag saves all runs for all algorithms in a structured directory with metadata for later re-evaluation.

### Legacy data format

The benchmark loader auto-detects and converts legacy files that used `all_wall_time_indices` instead of per-evaluation `time_steps`.
