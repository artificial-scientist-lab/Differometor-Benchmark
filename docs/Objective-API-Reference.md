# Objective API Reference

`Objective` is the central class of the benchmark. It wraps a `ContinuousProblem` and acts as the only interface between an optimization algorithm and the underlying physics simulation. Every function evaluation, gradient computation, and random sample goes through `Objective`, which transparently records everything needed for reproducible benchmarking.

If manual handling of everything is desired, `Objective` still offers the `ContinuousProblem` itself as an instance which has all the pure JAX-functions 

**Import:**

```python
from dfbench import Objective
```

---

## Constructor

```python
Objective(
    problem: ContinuousProblem,
    unbounded: bool = False,
    max_evals: int | None = None,
    max_time: float | None = None,
    save_time_steps: bool = True,
    save_params_history: bool = True,
    save_grad_history: bool = False,
    save_hessian_history: bool = False,
    save_batched_losses_history: bool = False,
    save_batched_grads_history: bool = False,
    save_batched_hessians_history: bool = False,
    save_batched_history: bool = False,
    save_eval_type_history: bool = False,
    verbose: int = 0,
    print_every: int = 100,
    algorithm_str: str | None = None,
    save_to_file_every: int | None = None,
    unit_mapping: Callable | None = None,
    inverse_unit_mapping: Callable | None = None,
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `problem` | `ContinuousProblem` | *required* | The optimization problem to wrap. |
| `unbounded` | `bool` | `False` | If `True`, the objective evaluates through the sigmoid-bounded variant (`sigmoid_objective_function`) so algorithms can search in $(-\infty, +\infty)^\text{n\_params}$ space. If `False`, evaluates the plain `objective_function` in bounded space. |
| `max_evals` | `int \| None` | `None` | Maximum number of function evaluations. `None` = unlimited. Batched evaluations are counted as how many parameters were given. |
| `max_time` | `float \| None` | `None` | Maximum wall-clock seconds beginning at the time `obj.start_logging()` was called. `None` = unlimited. |
| `save_time_steps` | `bool` | `True` | Record elapsed-time timestamp for each evaluation. |
| `save_params_history` | `bool` | `True` | Record the parameter vector at each evaluation. |
| `save_grad_history` | `bool` | `False` | Record gradient vectors. Off by default to save memory. |
| `save_hessian_history` | `bool` | `False` | Record Hessian matrices. Off by default because this can become memory-heavy quickly. |
| `save_batched_losses_history` | `bool` | `False` | When using `vmap_*` methods, store the full `(batch,)` loss vector instead of just the minimum loss of the batch. |
| `save_batched_grads_history` | `bool` | `False` | Store full batched gradient arrays. Careful: As the gradients are of dim `(n_params,)`, the batched version is of dim `(batch, n_params)` with the batched history being `(n_evals, batch, n_params)`! |
| `save_batched_hessians_history` | `bool` | `False` | Store full batched Hessian arrays with shape `(batch, n_params, n_params)`. Use sparingly for large batches or high-dimensional problems. |
| `save_batched_history` | `bool` | `False` | Also stores batched params and enables the full batched loss / grad / Hessian histories when those derivative histories are enabled. |
| `save_eval_type_history` | `bool` | `False` | Record a bitmask for each evaluation indicating whether it was a value, grad, hessian, combined call, and/or batched call. |
| `verbose` | `int` | `0` | Verbosity level. `0` = silent; `1` = periodic progress prints; `2` is WIP. |
| `print_every` | `int` | `100` | When `verbose ≥ 1`, print a progress summary every N evaluations. |
| `algorithm_str` | `str \| None` | `None` | If `None`, this is set by the algorithm via `prepare()` of `OptimizationAlgorithm`. Optional identifier string used in file names and logs. |
| `save_to_file_every` | `int \| None` | `None` | Automatically checkpoint to an NPZ-file every N evaluations. `None` disables auto-saving. The time spent saving is excluded from the elapsed-time clock. |
| `unit_mapping` | `Callable \| None` | `None` | Optional function mapping unbounded params to the **[0, 1] range**. Can be scalar (e.g. `jax.nn.sigmoid`) or element-wise vector. The Objective handles scaling to actual bounds: `bounded = lb + (ub - lb) * f(x)`. If omitted, the default sigmoid is used. |
| `inverse_unit_mapping` | `Callable \| None` | `None` | Inverse of the forward mapping, mapping [0, 1] → unbounded space. The Objective normalises bounded params to [0, 1] before calling this: `unbounded = f_inv((bounded - lb) / (ub - lb))`. Must be provided whenever `unit_mapping` is provided. |

### Choosing `unbounded`

| `unbounded` | Objective function used | Example algorithms |
|-------------|-------------------------|--------------------|
| `False` | `problem.objective_function` | Random Search, PSO, CMA-ES, Bayesian Optimization |
| `True` | `problem.sigmoid_objective_function` | Some gradient-based methods (Adam, L-BFGS, SA-GD, NA-Adam in their current implementations) |

### Choosing a bounded/unbounded mapping (important)

If you are new, use the defaults first:

- Leave both mapping arguments as `None`
- Set `unbounded=True` only if your optimizer expects unconstrained search space
- The default pair is sigmoid + inverse-sigmoid (logit)

Use a custom mapping only if you know exactly why you need it.

Rules:

- If you pass `unit_mapping`, you must also pass `inverse_unit_mapping`
- The forward mapping must produce values in **[0, 1]**; the Objective scales to actual bounds via `bounded = lb + (ub - lb) * f(x)`
- The inverse mapping receives values already normalised to [0, 1] by the Objective: `unbounded = f_inv((bounded - lb) / (ub - lb))`
- The inverse should satisfy approximately: `inverse(forward(x)) ≈ x` in the range you optimize over
- Both callables can be **scalar** functions (e.g. `jax.nn.sigmoid`) or **element-wise vector** functions — JAX broadcasts element-wise operations, so both work. The Objective uses `jax.vmap` for batching regardless

Minimal custom example (scalar function):

```python
import jax
from dfbench import Objective

# sigmoid maps (-inf, +inf) -> (0, 1) — perfect for the [0,1] contract
obj = Objective(
    problem,
    unbounded=True,
    unit_mapping=jax.nn.sigmoid,
    inverse_unit_mapping=lambda x: jax.numpy.log(x / (1.0 - x)),
)
```

Element-wise vector example (different mapping per dimension):

```python
import jax.numpy as jnp

def forward(x):
    # Per-dimension [0,1] mapping; x is shape (n_params,)
    return jnp.where(x > 0, 1 - jnp.exp(-x), jnp.exp(x)) * 0.5 + 0.5

def inverse(x):
    x = jnp.clip(x, 1e-7, 1.0 - 1e-7)
    centered = 2.0 * (x - 0.5)
    return jnp.where(centered > 0, -jnp.log(1 - centered), jnp.log(centered + 1))

obj = Objective(
    problem,
    unbounded=True,
    unit_mapping=forward,
    inverse_unit_mapping=inverse,
)
```

You do **not** need to handle bounds scaling — the Objective does that automatically.

---

## Evaluation Methods

All evaluation methods automatically record their results in the internal history. They are the primary way algorithms should interact with the objective.

### Single-point evaluation

```python
obj.value(params)              # → float
obj.grad(params)               # → Array[n_params]
obj.hessian(params)            # → Array[n_params, n_params]
obj.value_and_grad(params)     # → (float, Array[n_params])
obj.value_grad_and_hessian(params)  # → (float, Array[n_params], Array[n_params, n_params])
```

- `value(params)` — Evaluates the loss at `params`. Logs loss and params.
- `grad(params)` — Computes the gradient. Logs grad and params, but **not** a loss value (the loss is not computed).
- `hessian(params)` — Computes the exact Hessian. Logs Hessian and params, but **not** a loss value.
- `value_and_grad(params)` — Computes both in a single forward+backward pass. Logs all three. **Preferred when you need both loss and gradient** because it is more efficient than calling `value` and `grad` separately and it logs the loss.
- `value_grad_and_hessian(params)` — Computes loss, gradient, and Hessian together and logs all four.

### Batched evaluation

```python
obj.vmap_value(params_batch)              # → Array[batch]
obj.vmap_grad(params_batch)               # → Array[batch, n_params]
obj.vmap_hessian(params_batch)            # → Array[batch, n_params, n_params]
obj.vmap_value_and_grad(params_batch)     # → (Array[batch], Array[batch, n_params])
obj.vmap_value_grad_and_hessian(params_batch)
# → (Array[batch], Array[batch, n_params], Array[batch, n_params, n_params])
```

Convenience aliases:

```python
obj.batched_value(…)            # same as vmap_value
obj.batched_grad(…)             # same as vmap_grad
obj.batched_hessian(…)          # same as vmap_hessian
obj.batched_value_and_grad(…)   # same as vmap_value_and_grad
obj.batched_value_grad_and_hessian(…)  # same as vmap_value_grad_and_hessian
```

Batched methods use `jax.vmap` and evaluate the entire batch as **one** history entry. The eval counter is incremented by the batch size. When `save_batched_losses_history` is off (default), only the batch minimum loss is stored.

### Callable shorthand

```python
loss = obj(params)   # equivalent to obj.value(params)
```

### Manual logging

```python
obj.log_evaluation(params=…, loss=…, grad=…, hessian=…)
```

For algorithms with custom JIT-compiled evaluation loops that can't call `obj.value()` directly. Accepts the same `params`, `loss`, `grad`, `hessian` arguments and performs identical history recording.

---

## Lifecycle Methods

### `start_logging()`

Starts the wall-clock timer. **Must be called after JIT warmup and before the optimization loop.** All `time_steps` and budget checks are relative to this moment.

```python
# Typical sequence
obj.warmup_value_and_grad()           # warmup
obj.start_logging()                    # timer starts NOW
while not obj.budget_exceeded:
    …
```

### `warmup_*()`

`Objective` provides no-argument warmup helpers for every evaluation path:

```python
obj.warmup_value()
obj.warmup_grad()
obj.warmup_hessian()
obj.warmup_value_and_grad()
obj.warmup_value_grad_and_hessian()
obj.warmup_vmap_value()
obj.warmup_vmap_grad()
obj.warmup_vmap_hessian()
obj.warmup_vmap_value_and_grad()
obj.warmup_vmap_value_grad_and_hessian()
```

Each helper executes the matching path **twice** on deterministic parameters and must be called before `start_logging()`. The batched variants use a deterministic batch of size 2.

### `reset()`

Clears all histories, resets counters, and prepares for a completely fresh run. Does **not** change the problem, bounds, or budget limits.

### `set_seed(seed: int)`

Initializes the internal JAX PRNG key. Subsequent calls to `random_params_bounded()` and `random_params_unbounded()` consume and split this key automatically, guaranteeing identical initial samples across runs with the same seed.

This is to facilitate uniform initialization across algorithms.

```python
obj.set_seed(42)
p1 = obj.random_params_bounded(100)   # deterministic
p2 = obj.random_params_bounded(100)   # different from p1 but reproducible
obj.set_seed(42)
p3 = obj.random_params_bounded(100)   # identical to p1
```

### `set_space_mode(unbounded, unit_mapping=None, inverse_unit_mapping=None)`

Switches between bounded and unbounded mode before optimization starts.

- Must be called before `start_logging()`
- Re-binds all internal JAX evaluation paths (`value`, `grad`, `hessian`, all `vmap_*`)
- Can optionally replace the mapping pair at the same time
- Custom mappings follow the same [0, 1] contract as the constructor: the forward function maps to [0, 1], the Objective handles bounds scaling

```python
# default sigmoid mapping
obj.set_space_mode(True)

# custom mapping pair (scalar functions work)
import jax
obj.set_space_mode(
    True,
    unit_mapping=jax.nn.sigmoid,
    inverse_unit_mapping=lambda x: jax.numpy.log(x / (1.0 - x)),
)
```

---

## Random Sampling

### `random_params_bounded(n_samples=1, rng_key=None)`

Returns uniform random samples inside `problem.bounds`.

| Argument | Default | Description |
|----------|---------|-------------|
| `n_samples` | `1` | How many vectors to draw. Returns shape `(n_params,)` when 1, `(n_samples, n_params)` otherwise. |
| `rng_key` | `None` | Optional manual JAX key. If `None`, uses the internal key set by `set_seed()`. |

### `random_params_unbounded(n_samples=1, rng_key=None)`

Generates samples uniform in the bounded space then maps them to unbounded space using:

- your custom `inverse_unit_mapping` if provided — the Objective normalises bounded samples to [0, 1] first, then calls your inverse
- otherwise the default inverse sigmoid (logit)

With the matching forward mapping, this round-trip holds:

```python
bounded ≈ lb + (ub - lb) * forward(random_params_unbounded(...))
```

---

## Properties

### Problem & Configuration

| Property | Type | Description |
|----------|------|-------------|
| `bounds` | `Array[2, n_params]` | Lower and upper parameter bounds (or $\pm\infty$ when unbounded). |
| `n_params` | `int` | Number of optimizable parameters. |
| `problem` | `ContinuousProblem` | The wrapped problem instance. |

### Budget Tracking

| Property | Type | Description |
|----------|------|-------------|
| `eval_count` | `int` | Total evaluations so far. |
| `evals_left` | `int \| None` | Remaining evaluation budget. `None` if unlimited. |
| `evals_exceeded` | `bool` | Whether the evaluation cap has been reached. |
| `evals_progress_fraction` | `float` | Fraction of eval budget consumed (0–1). |
| `time_elapsed` | `float` | Seconds since `start_logging()`. |
| `time_left` | `float \| None` | Seconds remaining. `None` if unlimited. |
| `time_exceeded` | `bool` | Whether the time cap has been reached. |
| `time_progress_fraction` | `float` | Fraction of time budget consumed (0–1). |
| `budget_exceeded` | `bool` | `True` when **any** budget (time **or** evals) is exhausted. This is the main loop-termination check. |

### Best Results

| Property | Type | Description |
|----------|------|-------------|
| `best_loss` | `float \| None` | Lowest loss observed. `None` before the first evaluation. |
| `best_params` | `Array \| None` | Raw parameters at `best_loss` (may be in unbounded space). |
| `best_params_bounded` | `Array \| None` | Best parameters mapped to bounded space via the active mapping (custom mapping if configured, otherwise sigmoid). **Use this for final output.** |

### Current State

| Property | Type | Description |
|----------|------|-------------|
| `current_loss` | `float \| Array \| None` | Loss from the most recent evaluation. |
| `current_params` | `Array \| None` | Parameters from the most recent evaluation. |

### Raw History

These properties return **copies** to prevent external mutation.

| Property | Type | Description |
|----------|------|-------------|
| `loss_history` | `list` | All recorded losses (may contain batched arrays). |
| `grad_history` | `list` | All recorded gradients (if saving was enabled). |
| `hessian_history` | `list` | All recorded Hessians (if saving was enabled). |
| `params_history` | `list` | All recorded parameter vectors (raw space, i.e. as it was given to the `Objective`). |
| `params_history_bounded` | `list` | Params history mapped to bounded space. |
| `time_steps` | `list[float]` | Elapsed time at each recorded evaluation. |

### Reduced History

**Rationale:** Batched evaluations produce `(batch, ...)` shaped entries. Downstream analysis (benchmarking, plotting) expects flat lists of scalars/vectors. The `*_reduced` properties collapse each batch to a single representative value:

1. Select the entry (for loss, grad, Hessian and param) with the minimum loss if available for that step.
2. Else select the entry with the minimum gradient norm.
3. Else select the entry with the minimum Hessian norm.
4. Else take the first element.

| Property | Type | Description |
|----------|------|-------------|
| `loss_history_reduced` | `list[float]` | Losses with batches reduced to `nanmin`. |
| `params_history_reduced` | `list[Array \| None]` | Params with batches reduced per the rule above. |
| `params_history_reduced_bounded` | `list[Array \| None]` | Reduced params in bounded space. |
| `grad_history_reduced` | `list[Array \| None]` | Grads with batches reduced. |
| `hessian_history_reduced` | `list[Array \| None]` | Hessians with batches reduced. |

### Progress Counters

| Property | Type | Description |
|----------|------|-------------|
| `improvement_count` | `int` | How many times `best_loss` was improved. |
| `evals_since_improvement` | `int` | Evaluations since the last improvement — useful for patience-based early stopping. |

---

## I/O Methods

### `save_run_data(algorithm_name, filepath=None, hyper_param_str=None) → Path`

Saves the full optimization state to a compressed NPZ file. Writes atomically (to `.tmp.npz` first, then `os.replace`) to prevent corruption from interrupted HPC jobs.

Default path: `data/objective_run_data/{budget_dir}/{hyper_param_str}/{problem}_{algo}_{timestamp}.npz`

### `load_run_data(filepath)`

Restores all tracking state from a previously saved NPZ file. Adjusts `start_time` so that `time_elapsed` continues seamlessly from where the checkpoint left off.

### `output_to_files(hyper_param_str="", …) → Path`

Writes human-readable outputs:
- JSON with best parameters
- JSON with loss history
- PNG plot of the loss curve
- (For optical problems) PNG plot of the sensitivity curve vs. target

Output directory: `data/problem_output/{problem_name}/{algorithm_str}/{hyper_param_str}/`

### `get_summary() → dict`

Returns a snapshot dictionary:

```python
{
    "eval_count": int,
    "time_elapsed": float,
    "best_loss": float | None,
    "current_loss": float | None,
    "improvement_count": int,
    "evals_since_improvement": int,
    "budget_exceeded": bool,
    "time_exceeded": bool,
    "evals_exceeded": bool,
}
```

---

## Internal Logging Details

Every evaluation method follows the same pipeline internally:

1. **Execute** the JAX function (`_func`, `_value_and_grad_func`, `_vmap_func`, etc.)
2. **`_log_time()`** — record a `time_steps` entry; check time budget.
3. **`_log_evals(params, loss, grad, hessian)`** — record histories; update `best_loss` / `best_params`; update `improvement_count` / `evals_since_improvement`; check eval budget.
4. **`_log_to_file()`** — if `save_to_file_every` is set, trigger a periodic checkpoint.

> **Important:** These are private methods — do not call `_log_time()`, `_log_evals()`, or `_log_to_file()` directly from algorithm code. If you want manual logging, use the public `log_evaluation(params, loss, grad, hessian=None)` method instead, which wraps all three. See the [JIT-compiled loop guide](Implementing-a-New-Algorithm.md#custom-jit-compiled-loops-with-log_evaluation) for details.

Budget enforcement happens *after* the evaluation returns. This means the algorithm always receives a valid result, but once any budget is exceeded the history stops growing and `budget_exceeded` becomes `True`.

When a batch evaluation (`vmap_*`) would push `eval_count` past `max_evals`, the evaluations are counted but *not logged*, preserving history alignment and setting the `budget_exceeded` flag to `True`. The `time_steps` entry added by `_log_time()` is also removed to keep all lists in sync. This may be subject to change but in the current setting, this is the most straight-forward way and irrelevant if budged is planned well (reducing population as `evals_left` nears zero).
