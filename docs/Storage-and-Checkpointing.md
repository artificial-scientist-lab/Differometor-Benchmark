# Storage & Checkpointing

dfbench persists optimization runs through a modular `dfbench.core.storage` package that decouples **what** is saved (the `RunState` data contract) from **how** it is encoded (serializers) and **where** it physically goes (storage backends). A single facade — the `CheckpointManager` — wires these together and is the only storage object the `Objective` holds.

This layering makes every storage concern injectable: swap NPZ for JSON, local disk for S3, or redirect all artifacts to a scratch disk with a one-line change at construction time — no library code edits required.

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
        RunState               (canonical data contract)
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

## `RunState` — the canonical data contract

A plain dataclass holding everything needed to checkpoint or export a run. It is deliberately independent of the `Objective` class so serializers, exporters, and tests can operate on it without importing `Objective`.

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
| `metadata` | `RunMetadata` | Sidecar with run identity + problem spec |

`Objective._build_run_state()` is the single place that converts the Objective's internal histories/counters into a `RunState`. `Objective._apply_run_state(state)` reverses it on load.

---

## `RunMetadata` — run identity and problem reconstruction

A small, human-readable dataclass that travels alongside the numeric histories. It is stored as a JSON string *inside* the checkpoint file, so a single file is fully self-describing.

| Field | Type | Description |
|-------|------|-------------|
| `problem_name` | `str` | Problem label (e.g. `"voyager"`) |
| `algorithm_name` | `str` | Algorithm label |
| `hyper_param_str` | `str` | Hyperparameter string for path organisation |
| `timestamp` | `str` | Run timestamp (`YYYY-MM-DD_HH-MM-SS`) |
| `max_time` | `float \| None` | Time budget |
| `max_evals` | `int \| None` | Eval budget |
| `unbounded` | `bool` | Whether the Objective ran in unbounded mode |
| `extra` | `dict[str, Any]` | Extension point — holds `problem_spec` (see below) |

Every checkpoint includes a `format_version` field (`RunState`/`RunMetadata` write the current `FORMAT_VERSION`). Loaders refuse files written with a newer version than they understand, and fall back gracefully for older files missing keys.

### Embedded problem spec

If the wrapped problem implements the reconstructive `to_spec()` contract (see [Problems](Problems)), `Objective._build_metadata` records it in `metadata.extra["problem_spec"]`. This makes a checkpoint fully self-describing: the problem identity is recoverable from the file alone, not just from the caller's memory.

```python
state = manager.load(path)
problem = CheckpointManager.reconstruct_problem(state)  # or None
```

---

## Serializers

A `CheckpointSerializer` is a protocol with `serialize(state) -> bytes` and `deserialize(bytes) -> RunState`. Two implementations ship with dfbench:

### `NpzCheckpointSerializer` (default)

Compressed NumPy `.npz`. Matches the historical dfbench format but is now self-describing: the NPZ contains a `format_version` scalar and a `metadata` JSON string alongside the numeric arrays. Uses `dtype=object` arrays for ragged/batched histories but keeps `best_params` as `float64` (never object dtype) so JAX can consume it directly on load. Backwards-compatible with files written by older dfbench versions that lack `metadata` / `format_version` / some optional histories — missing keys fall back to empty defaults.

### `JsonCheckpointSerializer`

A fully pickle-free JSON format (histories encoded as nested lists). Slower and larger than NPZ, but trivially inspectable and safe to load from untrusted sources.

```python
from dfbench.core.storage import JsonCheckpointSerializer

manager = CheckpointManager(serializer=JsonCheckpointSerializer())
```

**Rationale — two formats:** NPZ is the default for efficiency. JSON exists for cases where maximum portability and zero-pickle safety matter more than size/speed.

---

## Storage Backends

A `StorageBackend` is a tiny protocol (`save_bytes` / `load_bytes` / `exists` / `delete`) so the local filesystem can be swapped for memory, S3, etc. without touching serializers or the `CheckpointManager`.

### `LocalFilesystemBackend` (default)

Writes are **atomic**: data is first written to a temporary file in the *same directory* as the target (so `os.replace` stays on one filesystem) and then renamed into place with `os.replace`. If `os.replace` fails, the temp file is cleaned up and the exception propagates — the previous good file is never destroyed, unlike a naive remove-then-rename fallback.

```python
from dfbench.core.storage import LocalFilesystemBackend

backend = LocalFilesystemBackend(root="./data/objective_run_data")
```

| Argument | Default | Description |
|----------|---------|-------------|
| `root` | `None` | Base directory. Relative keys resolve against it; absolute keys are used verbatim. `None` means keys are used as given (cwd-relative). |

**Rationale — atomic writes:** Long HPC jobs are killed without warning. A half-written checkpoint would be worse than none. The temp-then-replace pattern guarantees a reader always sees either the previous complete file or the new complete file, never a partial one.

---

## `RunPathResolver` — structured path construction

Builds filesystem paths from semantic components so no `./data/...` string is hardcoded in `Objective`. The root directory is configurable, letting users redirect all artifacts without editing library code.

Default layout (matches the historical convention):

```
{root}/{budget_dir}/{hyper_param_str}/{problem}_{algo}_{timestamp}.{ext}
```

where `budget_dir` is e.g. `time100s_evals1000` or `unlimited`.

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
# → ./data/objective_run_data/time100s_evals1000/lr0.1/voyager_adam_gd_2026-01-01_00-00-00.npz
```

---

## `RunDataExporter` — human-readable JSON + PNG

Replaces the old `Objective.output_to_files` by treating the human-readable artifacts as a *derived view* over the canonical `RunState`, not as a second write path inside the Objective. Plotting is split into pure functions (`plot_loss_curve`, `plot_sensitivity`) that return matplotlib figures; writing those figures/JSON to disk is a separate step.

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

**Rationale — exporter as a view:** Keeping a parallel write path on the problem (the old `OpticalSetupProblem.output_to_files`) would mean two places to maintain formats, paths, and atomicity, with drift risk. The exporter derives everything from `RunState`, so there is a single source of truth and the problem's responsibility stays limited to "define the objective + describe how to rebuild itself".

---

## `CheckpointManager` — the facade

The only storage object the `Objective` holds. It wires a serializer, backend, and resolver together, owns the periodic-checkpoint cadence (`save_every`), and provides `save` / `load` / `tick` operations. It also owns the cached checkpoint path so periodic saves overwrite the same file rather than creating timestamped duplicates, and exposes `last_checkpoint_eval` and `save_every` for the display layer.

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

The first `save()` without explicit overrides caches the computed path. Subsequent saves without overrides overwrite the same file. Passing `explicit_path` or `hyper_param_str` bypasses the cache. `load()` caches the loaded path so a resume-then-save cycle overwrites the same file.

### `tick` — periodic checkpointing

Called by `Objective._log_to_file` after each evaluation. The manager checks the cadence (`save_every`), and if a checkpoint is due, lazily calls `state_factory` to build a `RunState`, saves it, and returns the wall-clock duration of the save. The Objective advances `_start_time` by that duration so the checkpoint write does not consume wall-clock budget. If no checkpoint is due, `tick` returns `0.0` and `state_factory` is never called.

---

## Relationship to `Objective`

The `Objective` assembles a `CheckpointManager` and `RunDataExporter` internally with sensible defaults — these are **not** user-facing constructor parameters. The only storage-related knob exposed to the user is `save_to_file_every`, which sets the `save_every` cadence on the internal manager:

```python
obj = Objective(problem, save_to_file_every=1000)
```

The storage components remain modular and individually testable (see the sections above). Advanced users who need to swap a serializer, backend, or resolver can subclass `Objective` and override the internal assembly, or use the storage classes directly outside the Objective.

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Single canonical `RunState`** | Every serializer reads from and writes to the same dataclass, so formats don't drift. |
| **Schema versioning (`FORMAT_VERSION`)** | Loaders refuse newer-than-supported files and fall back gracefully for older ones. |
| **Metadata separated from numeric data** | Small JSON sidecar inside the NPZ identifies the run without parsing large arrays. |
| **Decoupled I/O from `Objective`** | `Objective` only builds/applies `RunState`; storage backends and formats are pluggable and testable in isolation. |
| **True atomic writes** | Temp-in-same-dir + `os.replace` only; the old unsafe remove-then-rename fallback is gone so a previous good file always survives a failed write. |
| **No `allow_pickle=True` for untrusted data** | Object arrays hold only numeric arrays we constructed ourselves; `best_params` is `float64`. JSON serializer is fully pickle-free. |
| **Exporter is a derived view** | JSON/PNG outputs come from `RunState`, not a second write path, so there is one source of truth. |
| **Problem spec embedded in checkpoint** | A saved run is fully self-describing: problem identity + histories + algo/budget. Enables provenance auditing and cross-process resume. |