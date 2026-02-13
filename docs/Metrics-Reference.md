# Metrics Reference

All metric functions live in `dfbench.benchmark.metrics`. They are organised into three tiers:

1. **Per-run** — operate on a single run's loss/time arrays
2. **Aggregation** — combine per-run scalars into statistics
3. **Multi-run** — inherently need data from all runs (diversity, top-k)

Plus **time-slicing utilities** used to evaluate metrics at arbitrary wall-clock cutoffs.

---

## Per-Run Metrics

These take a single run's arrays and return a scalar.

### `run_min_loss(losses) → float`

Minimum loss achieved in the run. Returns `inf` if the loss array is empty.

### `run_has_success(losses, threshold) → bool`

Whether **any** loss value falls below `threshold`. This is a binary indicator, not a count.

### `run_first_success_idx(losses, threshold) → int | None`

The iteration index at which the loss first drops below `threshold`. Returns `None` if no success occurred. (Internal helper, typically not called directly.)

### `run_first_success_time(losses, time_steps, threshold) → float | None`

Wall-clock time of first success. Looks up the `time_steps` entry at the index returned by `run_first_success_idx`.

### `run_auc(losses, time_steps, floor, baseline_loss, max_time) → float`

Area under the loss curve, computed via trapezoidal integration over wall-clock time.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `losses` | required | Loss values per evaluation |
| `time_steps` | required | Elapsed time per evaluation |
| `floor` | `None` | Shift losses by subtracting floor, then clamp to 0. Typically set to `success_loss` so the AUC measures "excess loss above success" |
| `baseline_loss` | `None` | Expected loss of a random guess. If provided with `max_time`, normalizes the result |
| `max_time` | `None` | Required if `baseline_loss` is set |

**Normalization:**

When both `baseline_loss` and `max_time` are given:

$$\text{AUC}_\text{norm} = -\log_2\!\left(\frac{\text{AUC}_\text{algo}}{\text{AUC}_\text{baseline}}\right)$$

where $\text{AUC}_\text{baseline} = \text{baseline\_loss} \times \text{max\_time}$ (constant loss over time).

Interpretation:
- Positive → better than random
- 0 → equal to random
- Negative → worse than random
- `inf` → perfect (zero AUC)

---

## Aggregation Functions

These combine lists of per-run scalars into summary statistics.

### `agg_mean_std(values: list[float]) → (mean, std)`

Mean and standard deviation using `jnp.nanmean` / `jnp.nanstd` to handle NaN values gracefully.

### `agg_min(values: list[float]) → float`

Global minimum across all runs. Returns `inf` for empty input.

### `agg_fraction_true(values: list[bool]) → float`

Fraction of `True` entries. E.g., fraction of runs that achieved success.

### `agg_mean_std_filtered(values: list[float | None], fallback=nan) → (mean, std)`

Mean and std of non-`None` values. Returns `(fallback, 0.0)` if all values are `None`. Used for metrics like "time to success" where failed runs contribute `None`.

---

## Multi-Run Metrics

These inherently require data from all runs.

### `multi_solution_diversity_overall(params, bounds) → (mean, std)`

Mean pairwise Euclidean distance between all successful solutions, normalized to $[0, 1]$.

**Normalization procedure:**
1. Each parameter dimension is scaled to $[0, 1]$ by bounds: $(x - \text{lb}) / (\text{ub} - \text{lb})$
2. Euclidean distance is divided by $\sqrt{n_\text{params}}$ so the maximum possible distance is 1

Returns `(0.0, 0.0)` if fewer than 2 successful solutions exist.

**Why this metric?** It measures whether independent runs converge to the same optimum or spread across multiple local minima. High diversity suggests the loss landscape has many viable solutions.

### `multi_solution_diversity_nn(params, bounds) → (mean, std)`

Mean **nearest-neighbor** distance instead of mean pairwise distance. Same normalization as above.

**Why both?** Overall diversity is dominated by far-apart clusters; NN diversity captures local packing density. A landscape with two tight clusters far apart will have high overall diversity but low NN diversity.

### `multi_auc_top_k(run_min_losses, run_aucs, k_fraction=0.1) → (mean, std)`

AUC statistics for the **top k%** runs by final minimum loss.

1. Sort runs by `min_loss`
2. Select the best `max(1, n_runs × k_fraction)` runs
3. Return mean and std of their AUC values

**Why?** Average AUC is dragged down by failed runs. Top-k AUC measures how well the algorithm performs *when it works*.

### `compute_performance_profile(run_min_losses, loss_thresholds) → (thresholds, success_rates, normalized_auc)`

Empirical CDF of final losses: for each threshold $\tau$, compute the fraction of runs with `min_loss < τ`.

**Default thresholds:** `jnp.linspace(-1.0, 5.0, 601)` — loss values on log scale for typical Differometor problems.

**Returns:**
| Value | Shape | Description |
|-------|-------|-------------|
| `thresholds` | `(n_thresholds,)` | Loss threshold values |
| `success_rates` | `(n_thresholds,)` | Fraction of runs below each threshold |
| `normalized_auc` | scalar | Area under the curve divided by threshold range |

**Interpretation:** A higher `normalized_auc` means the algorithm consistently achieves low losses. Similar to an ROC curve but for optimization performance.

---

## Time-Slicing Utilities

These functions enable evaluating metrics at arbitrary wall-clock cutoffs.

### `get_index_at_time(time_steps, t) → int`

Returns the last index $i$ where `time_steps[i] ≤ t`. Returns `-1` if `t` is before the first evaluation.

### `slice_history_at_time(history, time_steps, t) → list`

Returns `history[:idx+1]` where `idx = get_index_at_time(time_steps, t)`. Returns `[]` if no evaluations occurred by time `t`.

### `get_value_at_time(history, time_steps, t, default=None)`

Returns the single value `history[idx]` at time `t`, or `default` if no data exists.

**Why three functions?** Different callers need different things:
- `slice_history_at_time` → computing metrics on the prefix (AUC, min loss)
- `get_value_at_time` → reading best params at a given time
- `get_index_at_time` → low-level, used by the other two

---

## How the Benchmark Applies Metrics

For each time sample $t$ and each run $r$:

```
losses_at_t = slice_history_at_time(run.loss_history, run.time_steps, t)
time_at_t   = slice_history_at_time(run.time_steps,   run.time_steps, t)

per_run_min_loss[r]     = run_min_loss(losses_at_t)
per_run_has_success[r]  = run_has_success(losses_at_t, success_loss)
per_run_first_success[r] = run_first_success_time(losses_at_t, time_at_t, success_loss)
per_run_auc[r]          = run_auc(losses_at_t, time_at_t, ...)
```

Then aggregated:

```
fraction_of_success[t]  = agg_fraction_true(per_run_has_success)
min_loss[t]             = agg_min(per_run_min_loss)
avg_loss[t]             = agg_mean_std(per_run_min_loss)
time_to_success[t]      = agg_mean_std_filtered(per_run_first_success)
auc_top_1[t]            = per_run_auc[best_run]
auc_top_10[t]           = multi_auc_top_k(per_run_min_loss, per_run_auc)
performance_profile_auc[t] = compute_performance_profile(per_run_min_loss)[2]
```

Additionally, for diversity, only the parameters of **successful** runs at time $t$ are used:

```
successful_params = [get_value_at_time(run.params, ..., t) for run if has_success]
diversity_overall[t] = multi_solution_diversity_overall(successful_params, bounds)
diversity_nn[t]      = multi_solution_diversity_nn(successful_params, bounds)
```
