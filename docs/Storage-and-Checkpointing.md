# Storage & Checkpointing

dfbench persists optimization runs through the modular `dfbench.core.storage` package. The package separates three concerns: **what** is saved (the `RunState` data contract), **how** it is encoded (serializers), and **where** it ends up on disk (storage backends). A facade class, `CheckpointManager`, glues these together and is the only storage object an `Objective` ever talks to.

Because the concerns are split, every storage choice is injectable: you can swap NPZ for JSON, local disk for S3, or redirect all artifacts to a scratch disk by changing one argument at construction time. No library code needs to be touched.

**Import:**

```python
from dfbench.core.storage import (
    CheckpointManager,
    NpzCheckpointSerializer,
    JsonCheckpointSerializer,
    LocalFilesystemBackend,
    RunPathResolver,
    RunDataExporter,
    RunState,
    RunMetadata,
)
```

---

## Architecture

```
Objective
    │
    ▼
CheckpointManager          ← the only facade Objective talks to
    │
    ├── CheckpointSerializer   (how: NPZ or JSON)
    ├── StorageBackend         (where: local FS, memory, S3, ...)
    ├── RunPathResolver        (path layout from components)
    └── RunDataExporter        (human-readable JSON + PNG view)
            │
            ▼
        RunState               (shared data contract)
        RunMetadata            (problem/algo/budget identity + problem_spec)
```

### Separation of concerns

| Layer | Responsibility | Protocol / Class |
|-------|----------------|------------------|
| **Data contract** | What a run looks like in memory | `RunState`, `RunMetadata` |
| **Serializer** | Encode/decode `RunState` ↔ bytes | `CheckpointSerializer` |
| **Backend** | Where bytes physically go | `StorageBackend` |
| **Resolver** | Build structured paths | `RunPathResolver` |
| **Exporter** | Human-readable JSON + PNG view | `RunDataExporter` |
| **Manager** | Orchestrate save/load/periodic checkpoint | `CheckpointManager` |

---

## `RunState`: the shared data contract

`RunState` is a plain dataclass that holds everything needed to checkpoint or export a run. It is kept independent of the `Objective` class on purpose, so serializers, exporters, and tests can operate on a run without importing `Objective`.

| Field | Type | Description |
|-------|------|-------------|
| `loss_history` | `np.ndarray` | Aligned loss history (object dtype for ragged/batched entries) |
| `grad_history` | `np.ndarray` | Aligned gradient history |
| `hessian_history` | `np.ndarray` | Aligned Hessian history |
| `params_history` | `np.ndarray` | Aligned parameter history (raw space) |
| `eval_type_history` | `np.ndarray` | Per-eval bitmask call type |
| `time_steps` | `np.ndarray` | Elapsed-time stamps aligned with histories |
| `eval_count` | `int` | Total evaluations |
| `best_loss` | `float` | Lowest loss observed |
| `best_params` | `np.ndarray` | Parameters at `best_loss` (float64; empty if none) |
| `improvement_count` | `int` | Times `best_loss` was improved |
| `evals_since_improvement` | `int` | Evaluations since last improvement |
| `log_call_count` | `int` | Number of internal `_log_evals` invocations |
| `eval_type_counts` | `dict[int, int]` | Distribution of eval call types |
| `metadata` | `RunMetadata` | Record with run identity + problem spec |

`Objective._build_run_state()` is the single place that converts the Objective's internal histories and counters into a `RunState`. `Objective._apply_run_state(state)` reverses it on load.

---

## `RunMetadata`: run identity and problem reconstruction

`RunMetadata` is a small, human-readable dataclass that sits alongside the numeric histories. It is stored as a JSON string *inside* the checkpoint file, so one file is enough to fully describe the run that produced it.

| Field | Type | Description |
|-------|------|-------------|
| `problem_name` | `str` | Problem label (e.g. `"voyager"`) |
| `algorithm_name` | `str` | Algorithm label |
| `hyper_param_str` | `str` | Hyperparameter string for path organisation |
| `timestamp` | `str` | Run timestamp (`YYYY-MM-DD_HH-MM-SS`) |
| `max_time` | `float \| None` | Time budget |
| `max_evals` | `int \| None` | Eval budget |
| `unbounded` | `bool` | Whether the Objective ran in unbounded mode |
| `extra` | `dict[str, Any]` | Extension point; holds `problem_spec` (see below) |

Every checkpoint carries a `format_version` scalar (`RunState`/`RunMetadata` write the current `FORMAT_VERSION`). Loaders refuse files written with a newer version than they understand, and fill missing optional fields with defaults when loading files that predate those fields.

### Embedded problem spec

If the wrapped problem implements the reconstructive `to_spec()` contract (see [Problems](Problems)), `Objective._build_metadata` records it in `metadata.extra["problem_spec"]`. This makes a checkpoint self-describing: the problem identity is recoverable from the file alone, not just from the caller's memory.

```python
state = manager.load(path)
problem = CheckpointManager.reconstruct_problem(state)  # or None
```

---

## Serializers

A `CheckpointSerializer` is a protocol with `serialize(state) -> bytes` and `deserialize(bytes) -> RunState`. dfbench ships two implementations.

### `NpzCheckpointSerializer` (default)

Compressed NumPy `.npz`. The NPZ is self-describing: it contains a `format_version` scalar and a `metadata` JSON string alongside the numeric arrays. Ragged/batched histories use `dtype=object` arrays, but `best_params` is kept as `float64` (never object dtype) so JAX can consume it directly on load. Missing keys fall back to empty defaults, so checkpoints that omit some optional histories still load.

### `JsonCheckpointSerializer`

A fully pickle-free JSON format, with histories encoded as nested lists. It is slower and larger than NPZ, but trivially inspectable and safe to load from untrusted sources.

```python
from dfbench.core.storage import JsonCheckpointSerializer

manager = CheckpointManager(serializer=JsonCheckpointSerializer())
```

**Why two formats:** NPZ is the default because it is small and fast. JSON exists for the cases where portability and zero-pickle safety matter more than size and speed, e.g. loading a checkpoint produced by someone else's machine.

---

## Storage Backends

A `StorageBackend` is a small protocol (`save_bytes` / `load_bytes` / `exists` / `delete`). Keeping it this narrow means the local filesystem can be swapped for memory, S3, or any other target without touching the serializers or the `CheckpointManager`.

### `LocalFilesystemBackend` (default)

Writes are **atomic**. Data is first written to a temporary file in the *same directory* as the target (so `os.replace` stays on one filesystem) and then renamed into place with `os.replace`. If `os.replace` fails, the temp file is cleaned up and the exception propagates. The previous good file is never destroyed.

```python
from dfbench.core.storage import LocalFilesystemBackend

backend = LocalFilesystemBackend(root="./data/objective_run_data")
```

| Argument | Default | Description |
|----------|---------|-------------|
| `root` | `None` | Base directory. Relative keys resolve against it; absolute keys are used verbatim. `None` means keys are used as given (cwd-relative). |

**Why atomic writes:** HPC jobs get killed without warning, and a half-written checkpoint is worse than no checkpoint. The temp-then-replace pattern guarantees a reader always sees either the previous complete file or the new complete file, never a partial one.

---

## `RunPathResolver`: structured path construction

`RunPathResolver` builds filesystem paths from semantic components, so no `./data/...` string is hardcoded inside `Objective`. The root directory is configurable, letting users redirect all artifacts without editing library code.

Saving layout:

```
{root}/{budget_dir}/{algo}_{hyper_param_str}/{problem}_{algo}_{timestamp}.{ext}
```

where `budget_dir` is e.g. `time100s_evals1000` or `unlimited`. When
`hyper_param_str` is empty/None the directory segment collapses to just
`.../{algo}/...`.

```python
from dfbench.core.storage import RunPathResolver

resolver = RunPathResolver(root="./data/objective_run_data", extension="npz")
path = resolver.checkpoint_path(
    problem_name="voyager",
    algorithm_name="adam_gd",
    timestamp="2026-01-01_00-00-00",
    hyper_param_str="lr0.1",
    max_time=100.0,
    max_evals=1000,
)
# → ./data/objective_run_data/time100s_evals1000/adam_gd_lr0.1/voyager_adam_gd_2026-01-01_00-00-00.npz
```

---

## `RunDataExporter`: human-readable JSON + PNG

`RunDataExporter` treats the human-readable artifacts as a *derived view* over the shared `RunState` instead of a second write path inside `Objective`. Plotting is split into pure functions (`plot_loss_curve`, `plot_sensitivity`) that return matplotlib figures; writing those figures and the JSON to disk is a separate step.

For optical problems that expose `calculate_sensitivity` / `_frequencies` / `_target_sensitivities`, a sensitivity plot is produced in addition to the loss curve.

```python
from dfbench.core.storage import RunDataExporter

exporter = RunDataExporter(root="./data/problem_output")
out_dir = exporter.export(state, problem=problem, hyper_param_str="lr0.1")
```

Files written to `{root}/{problem_name}/{algorithm_name}/{hyper_param_str}/`:

| File | Content |
|------|---------|
| `{prefix}_parameters{suffix}.json` | Best parameters (bounded space) |
| `{prefix}_losses{suffix}.json` | Full loss history |
| `{prefix}_losses{suffix}.png` | Loss curve plot |
| `{prefix}_sensitivity{suffix}.png` | Sensitivity curve vs. target (optical problems only) |

**Why an exporter-as-a-view:** keeping a parallel write path on the problem would mean two places to maintain formats, paths, and atomicity, with drift risk between them. Deriving everything from `RunState` leaves one source of truth, and the problem's responsibility stays limited to defining the objective and describing how to rebuild itself.

---

## `CheckpointManager`: the facade

`CheckpointManager` is the only storage object an `Objective` holds. It wires a serializer, backend, and resolver together, owns the periodic-checkpoint cadence (`save_every`), and provides `save` / `load` / `tick` operations. It also owns the cached checkpoint path, so periodic saves overwrite the same file instead of creating timestamped duplicates, and exposes `last_checkpoint_eval` and `save_every` for the display layer.

```python
from dfbench.core.storage import CheckpointManager, LocalFilesystemBackend, NpzCheckpointSerializer, RunPathResolver

manager = CheckpointManager(
    backend=LocalFilesystemBackend(root="./data/objective_run_data"),
    serializer=NpzCheckpointSerializer(),
    resolver=RunPathResolver(root="./data/objective_run_data"),
    save_every=1000,
)

# Save
path = manager.save(state)

# Load
state = manager.load(path)

# Periodic checkpoint (lazy: state_factory only called when due)
# Returns wall-clock duration of the save (0.0 if no checkpoint was taken)
dt = manager.tick(eval_count=obj.eval_count,
                  state_factory=lambda: obj._build_run_state())

# Reconstruct the problem from a loaded checkpoint
problem = CheckpointManager.reconstruct_problem(state)
```

### Cached path behaviour

The first `save()` without explicit overrides caches the computed path. Subsequent saves without overrides overwrite the same file. Passing `explicit_path` or `hyper_param_str` bypasses the cache. `load()` caches the loaded path, so a resume-then-save cycle overwrites the same file.

### `tick`: periodic checkpointing

`tick` is called by `Objective._log_to_file` after each evaluation. The manager checks the cadence (`save_every`); if a checkpoint is due, it lazily calls `state_factory` to build a `RunState`, saves it, and returns the wall-clock duration of the save. The `Objective` then advances `_start_time` by that duration, so the checkpoint write does not consume wall-clock budget. If no checkpoint is due, `tick` returns `0.0` and `state_factory` is never called.

---

## Relationship to `Objective`

The `Objective` assembles a `CheckpointManager` and `RunDataExporter` internally with sensible defaults. These are not user-facing constructor parameters; the only storage knob exposed to the user is `save_to_file_every`, which sets the `save_every` cadence on the internal manager:

```python
obj = Objective(problem, save_to_file_every=1000)
```

The storage components remain modular and individually testable (see the sections above). Advanced users who need to swap a serializer, backend, or resolver can subclass `Objective` and override the internal assembly, or use the storage classes directly outside the `Objective`.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Single shared `RunState`** | Every serializer reads from and writes to the same dataclass, so formats don't drift. |
| **Schema versioning (`FORMAT_VERSION`)** | Loaders refuse newer-than-supported files and fall back gracefully for older ones. |
| **Metadata separated from numeric data** | A small JSON record inside the NPZ identifies the run without parsing large arrays. |
| **Decoupled I/O from `Objective`** | `Objective` only builds/applies `RunState`; storage backends and formats are pluggable and testable in isolation. |
| **True atomic writes** | Temp-in-same-dir + `os.replace` only. The previous good file always survives a failed write. |
| **No `allow_pickle=True` for untrusted data** | Object arrays hold only numeric arrays we constructed ourselves; `best_params` is `float64`. The JSON serializer is fully pickle-free. |
| **Exporter is a derived view** | JSON/PNG outputs come from `RunState`, not a second write path, so there is one source of truth. |
| **Problem spec embedded in checkpoint** | A saved run is self-describing: problem identity + histories + algo/budget. This enables provenance auditing and cross-process resume. |