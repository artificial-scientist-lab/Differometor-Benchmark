# Benchmarking

The benchmark module orchestrates multi-algorithm comparison on a single problem. It runs each algorithm multiple times with independent seeds, then evaluates a suite of metrics at regularly-sampled time points.

---

## Quick Start

```python
from dfbench.benchmark import Benchmark, AlgorithmConfig
from dfbench.algorithms import AdamGD, EvoxPSO, BotorchBO
from dfbench.problems import VoyagerProblem

problem = VoyagerProblem()
configs = [
    AlgorithmConfig(AdamGD(), {"learning_rate": 0.1}, name="Adam_lr0.1"),
    AlgorithmConfig(EvoxPSO(variant="PSO"), {"pop_size": 100}, name="PSO_100"),
    AlgorithmConfig(BotorchBO(), {"n_initial": 50}, name="BO"),
]

benchmark = Benchmark(
    problem=problem,
    success_loss=0.1,
    configs=configs,
    n_runs=100,
    max_time=300,
)
results = benchmark.run()
benchmark.print_summary(results)
```

---

## Key Concepts

### Time-Sampled Evaluation

Rather than reporting a single final number, the benchmark evaluates **every metric at multiple time slices**. This produces curves showing how each metric evolves over wall-clock time, enabling fair comparison between algorithms with very different per-iteration costs.

Given `max_time=300` and `n_time_samples=100`, metrics are computed at `[3, 6, 9, ..., 300]` seconds. At each time point $t$, a run's history is sliced to only include evaluations with `elapsed_time ≤ t`, and metrics are computed on that prefix.

### Why wall-clock time?

Iteration count is a poor comparison axis: one Adam iteration (1 eval) takes ~12 ms, while one BO iteration (1 acquisition + 1 eval) takes ~500 ms after a GP fit. Wall-clock time is the only fair common axis.

### Success threshold

A run is "successful" when it achieves a loss below `success_loss`. This is a problem-specific value chosen by the user; it defines what counts as a "good enough" solution for a given physics problem.

---

## API Reference

### `AlgorithmConfig`

Wraps an algorithm instance with the hyperparameters that should be passed to `optimize()`.

```python
AlgorithmConfig(
    algorithm: OptimizationAlgorithm,   # algorithm instance
    hyperparameters: dict | None = None, # kwargs for optimize()
    name: str | None = None,             # display name (default: algorithm_str)
)
```

**Why a separate config instead of passing kwargs directly?**
So the same algorithm instance can appear twice with different hyperparameters (e.g., `Adam_lr0.01` vs `Adam_lr0.1`).

### `Benchmark`

```python
Benchmark(
    problem: ContinuousProblem,         # the problem to benchmark on
    success_loss: float,                # loss threshold for "success"
    configs: list[AlgorithmConfig],     # algorithms to compare
    n_runs: int = 100,                  # independent runs per algorithm
    max_time: float = 300.0,            # wall-clock budget per run (seconds)
    n_time_samples: int = 100,          # metric evaluation points
    random_baseline_loss: float | None = None,  # for normalized AUC
    random_seed: int | None = None,     # master RNG seed
    storage_backend: StorageBackend | None = None,  # where run data is stored
)
```

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_runs` | 100 | More runs -> more reliable statistics, but linear time cost |
| `max_time` | 300 | Seconds. All algorithms get the same wall-clock budget |
| `n_time_samples` | 100 | Points to sample in `[max_time/n, max_time]` |
| `random_baseline_loss` | `None` | Expected loss of random guess. If set, AUC metrics are normalized |
| `random_seed` | `None` | If set, generates deterministic per-run seeds from this master seed |
| `storage_backend` | `None` | Where benchmark run data (NPZ/JSON/CSV) is physically stored. Defaults to a `LocalFilesystemBackend` (cwd-relative). Swapping this redirects all benchmark artifacts (e.g. to a scratch disk or S3-backed prefix) without code changes. See [Storage & Checkpointing](Storage-and-Checkpointing). |

### `Benchmark.run()`

```python
results = benchmark.run(
    verbose: int = 1,
    save_csv: bool = True,
    save_run_data: bool = False,
    load_from: str | Path | None = None,
    output_dir: str | Path = "./data/benchmark_run_data",
) -> list[BenchmarkResult]
```

| Parameter | Description |
|-----------|-------------|
| `save_csv` | Write all time-sampled metrics to a CSV file in `./data/benchmark_results/` |
| `save_run_data` | Save raw `RunData` (per-evaluation losses, times, params) to NPZ files |
| `load_from` | Path to a previously saved run data directory (*skips running*, re-evaluates metrics only) |
| `output_dir` | Base directory for run data NPZ files |

---

## Data Flow

```
AlgorithmConfig[]                                  <- user defines
       │
       ▼
   Benchmark.run()
       │
       ├── _collect_all_run_data()                 <- runs algorithms
       │       │
       │       └── _collect_algorithm_runs()       <- n_runs × optimize()
       │               │
       │               └── RunData.from_objective() <- extracts arrays
       │
       ├── _evaluate_algorithm()                   <- computes metrics at each time slice
       │       │
       │       ├── slice_history_at_time()
       │       ├── run_min_loss(), run_has_success(), run_auc(), ...
       │       ├── agg_mean_std(), agg_fraction_true(), ...
       │       └── multi_solution_diversity_*(), compute_performance_profile()
       │
       └── BenchmarkResult[]                       <- returned
               │
               ├── print_summary()                 <- console table
               └── _save_results_to_csv()          <- CSV file
```

---

## Data Classes

### `RunData`

Serializable data extracted from one `Objective` after optimization:

| Field | Shape | Description |
|-------|-------|-------------|
| `loss_history` | `(n_evals,)` | Loss at each evaluation |
| `time_steps` | `(n_evals,)` | Elapsed wall-clock time at each evaluation |
| `params_history` | `(n_evals, n_params)` | Bounded parameters at each evaluation |
| `best_loss` | scalar | Global best loss |
| `best_params` | `(n_params,)` | Parameters corresponding to `best_loss` |
| `eval_count` | `int` | Total evaluation count |

Created via `RunData.from_objective(obj)` which reads the `Objective`'s reduced (non-batched) properties.

### `AlgorithmRunData`

Groups all runs for one algorithm configuration:

| Field | Type | Description |
|-------|------|-------------|
| `algorithm_name` | `str` | Display name from `AlgorithmConfig` |
| `runs` | `list[RunData]` | One per independent run |
| `hyperparameters` | `dict` | Kwargs passed to `optimize()` |

### `BenchmarkResult`

Time-sampled metrics for one algorithm. Every metric has shape `(n_time_samples,)`.

**Single-value metrics (`SingleMetric`):**

| Metric | Meaning |
|--------|---------|
| `fraction_of_success` | Fraction of runs with loss < `success_loss` |
| `min_loss` | Global minimum loss across all runs |
| `performance_profile_auc` | Normalized AUC of the empirical CDF of final losses |
| `auc_top_1` | AUC of the single best run |

**Aggregate metrics (`AggregateMetric`, has `.mean` and `.std`):**

| Metric | Meaning |
|--------|---------|
| `avg_loss` | Per-run minimum loss, averaged |
| `time_to_success` | Wall-clock time to first success (successful runs only) |
| `evals_to_success` | Evaluation count to first success (successful runs only) |
| `solution_diversity_overall` | Mean pairwise distance of successful solutions |
| `solution_diversity_nn` | Mean nearest-neighbor distance of successful solutions |
| `auc_top_10` | AUC statistics of top 10% runs by final loss |

---

## Saving and Loading Run Data

### Saving

```python
results = benchmark.run(save_run_data=True)
```

This creates a timestamped directory under `./data/benchmark_run_data/`:

```
data/benchmark_run_data/
└── VoyagerProblem_2024-01-15_14-30-00/
    ├── metadata.json
    ├── Adam_lr0.1.npz
    ├── PSO_100.npz
    └── BO.npz
```

**`metadata.json`** stores the benchmark configuration (problem name, success threshold, seeds, algorithm list). Each **`.npz`** file contains all runs for one algorithm in NumPy's compressed format. All files are written **atomically** through the configured `StorageBackend` (temp-in-same-dir + `os.replace`), so an interrupted benchmark never leaves half-written files. See [Storage & Checkpointing](Storage-and-Checkpointing).

### Reloading

```python
results = benchmark.run(load_from="./data/benchmark_run_data/VoyagerProblem_2024-01-15_14-30-00")
```

This **skips running algorithms entirely** and re-evaluates metrics from saved data. Useful for:

- Adjusting `success_loss` or `n_time_samples` without re-running
- Computing new metrics on old data
- Moving data between machines

### Legacy Format Support

Older data used `all_wall_time_indices` instead of per-evaluation `time_steps`. The loader auto-detects the format and converts legacy data via linear interpolation between wall-time checkpoints.

---

## CSV Output

`save_csv=True` (the default) writes a CSV file to `./data/benchmark_results/`:

```
benchmark_VoyagerProblem_2024-01-15_14-30-00.csv
```

Each row is one `(algorithm, time_sample)` pair. Columns include `_mean` and `_std` for every metric. This is the primary format for downstream plotting and analysis.

---

## Reproducibility

If `random_seed` is provided, the `Benchmark` generates deterministic per-run seeds:

```python
rng = np.random.RandomState(self._random_seed)
run_seeds = [int(rng.randint(0, 2**31)) for _ in range(self._n_runs)]
```

Each run gets the same seed regardless of which algorithm is being evaluated. This means run `i` always starts from the same random state across all algorithms, reducing variance in comparisons.

---

## Printing Results

```python
benchmark.print_summary(results)
```

```
==========================================================================================
BENCHMARK SUMMARY (at final time)
==========================================================================================
Algorithm                  Success%     Min Loss          Avg Loss        Time(s)
------------------------------------------------------------------------------------------
Adam_lr0.1                    85.0%     0.012345     0.0567±0.0234     45.2±12.3
PSO_100                       72.0%     0.023456     0.0890±0.0456     78.9±25.1
BO                            68.0%     0.034567     0.1123±0.0678    120.5±42.7
==========================================================================================
```

Values shown are from the **final** time sample (i.e., at `max_time`).

---

## Design Notes

### Why not use iteration count as the x-axis?

Different algorithms have vastly different per-iteration costs. A BO iteration includes fitting a GP and optimizing an acquisition function; an Adam iteration is a single forward+backward pass. Comparing at "iteration 1000" is meaningless, but comparing at "t = 60 s" is fair.

### Why save raw run data?

Metrics evolve as the project matures. Saving raw `(loss, time, params)` tuples means new metrics can be computed on old data without re-running expensive experiments. The `load_from` parameter enables this workflow.

### Why NPZ instead of HDF5 / Parquet?

NumPy's NPZ format is:
- Zero-dependency (no `h5py` or `pyarrow` needed)
- Handles ragged arrays via `dtype=object`
- Fast for small-to-medium datasets (typical benchmark: ~100 runs × ~10k evals ≈ 1 MB)
