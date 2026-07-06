"""Canonical in-memory representation of an optimization run.

A :class:`RunState` is a plain dataclass holding everything needed to
checkpoint or export a run. It is deliberately independent of the
:class:`~dfbench.core.objective.Objective` class so that serializers,
exporters, and tests can operate on it without importing the Objective.

The companion :class:`RunMetadata` carries small, human-readable
descriptors (problem name, algorithm name, hyperparameter string, budget
limits, timestamp) that travel alongside the numeric histories.

This module also defines the **invariant contract** every :class:`RunState`
must satisfy before it crosses the disk trust boundary (checkpoint save or
load). :func:`validate_run_state` checks that contract and produces a
machine-readable :class:`ValidationReport` so a malformed or tampered run
becomes a deterministic, reportable rejection instead of a crash deep
inside a serializer or the scoring layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

FORMAT_VERSION: int = 1
"""On-disk format version written by serializers.

Increment this when the serialized schema changes in a
backwards-incompatible way.  Loaders should refuse (or warn about) files
written with a newer version than they understand.
"""


def _empty_object_array() -> np.ndarray:
    """Return an empty object-dtype array, the empty-history default."""
    return np.array([], dtype=object)


@dataclass
class RunMetadata:
    """Small, human-readable descriptors for a run.

    Stored as a JSON sidecar next to the binary checkpoint so that a run
    can be identified without parsing the (potentially large) numeric
    arrays.
    """

    problem_name: str = "problem"
    algorithm_name: str = "unknown"
    hyper_param_str: str = ""
    timestamp: str = ""
    max_time: float | None = None
    max_evals: int | None = None
    unbounded: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "problem_name": self.problem_name,
            "algorithm_name": self.algorithm_name,
            "hyper_param_str": self.hyper_param_str,
            "timestamp": self.timestamp,
            "max_time": self.max_time,
            "max_evals": self.max_evals,
            "unbounded": self.unbounded,
            "format_version": FORMAT_VERSION,
        }
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunMetadata":
        version = d.get("format_version")
        if version is not None and int(version) > FORMAT_VERSION:
            raise ValueError(
                f"Run data format version {version} is newer than the "
                f"supported version {FORMAT_VERSION}. Please update dfbench."
            )
        return cls(
            problem_name=str(d.get("problem_name", "problem")),
            algorithm_name=str(d.get("algorithm_name", "unknown")),
            hyper_param_str=str(d.get("hyper_param_str", "")),
            timestamp=str(d.get("timestamp", "")),
            max_time=d.get("max_time"),
            max_evals=d.get("max_evals"),
            unbounded=bool(d.get("unbounded", False)),
            extra=d.get("extra", {}) or {},
        )


@dataclass
class RunState:
    """Full serializable snapshot of one optimization run.

    All numeric histories are stored as plain ``numpy.ndarray`` (object
    dtype for ragged/batched entries). This is the single data contract
    every serializer reads from and writes to.

    The aux histories (``sensitivity_loss_history``, ``penalty_history``,
    ``is_feasible_history``, ``violations_history``, and the three
    ``power_*_history`` fields) are populated only when the matching
    :class:`~dfbench.core.storage.saveconfig.SaveConfig` token is enabled
    and the run used the ``value_aux`` / ``vmap_value_aux`` methods. They
    are kept as separate leaf arrays rather than a dict so both the NPZ
    and JSON serializers can encode them without pickling.
    """

    # Aligned histories (all length == log_call_count, modulo placeholders)
    loss_history: np.ndarray
    grad_history: np.ndarray
    hessian_history: np.ndarray
    params_history: np.ndarray
    eval_type_history: np.ndarray
    time_steps: np.ndarray

    # Scalar / aggregate state
    eval_count: int
    best_loss: float
    best_params: np.ndarray  # empty array if None
    improvement_count: int
    evals_since_improvement: int

    # Lightweight call-type tracking
    log_call_count: int
    eval_type_counts: dict[int, int]

    # Metadata sidecar
    metadata: RunMetadata = field(default_factory=RunMetadata)

    # Best-loss location (defaults: unknown for legacy checkpoints)
    best_eval_index: int | None = None
    best_batch_index: int | None = None

    # Aux diagnostics histories (aligned with the histories above; empty
    # when the corresponding save token was not enabled or no aux eval ran).
    # Kept as separate leaf arrays rather than a dict so both NPZ and JSON
    # serializers can encode them without pickling.
    sensitivity_loss_history: np.ndarray = field(default_factory=_empty_object_array)
    penalty_history: np.ndarray = field(default_factory=_empty_object_array)
    is_feasible_history: np.ndarray = field(default_factory=_empty_object_array)
    violations_history: np.ndarray = field(default_factory=_empty_object_array)
    power_hard_history: np.ndarray = field(default_factory=_empty_object_array)
    power_soft_history: np.ndarray = field(default_factory=_empty_object_array)
    power_detector_history: np.ndarray = field(default_factory=_empty_object_array)


# ------------------------------------------------------------------
# Invariant contract
# ------------------------------------------------------------------
#
# The validator is a pure function — no I/O, no exceptions raised. It
# *collects* every violation so a competition rejection report lists all
# problems in one pass rather than one-at-a-time (raise→catch→raise→catch).
#
# Two tiers are checked:
#
# * **Structural (A)** — cheap type/shape/range checks that always hold
#   for a well-formed ``RunState``. Checked on every call.
# * **Semantic (B)** — cross-field consistency checks (e.g.
#   ``best_loss == nanmin(loss_history)``). Skipped when ``strict=False``
#   because they can transiently fail for a partial mid-run snapshot (for
#   example, when ``best_params`` was intentionally left empty by a
#   ``log_evaluation(None, loss)`` call).
#
# The on-disk format version is *not* re-checked here: it is already
# enforced at deserialization (``RunMetadata.from_dict`` and each
# serializer's ``deserialize``), and ``RunMetadata.to_dict`` always
# hardcodes ``FORMAT_VERSION``, so a constructed ``RunMetadata`` cannot
# carry a future version by construction.
#
# Invariants deliberately *not* checked (see plan #2):
#   - NaN presence inside histories (legal placeholder for missing
#     grad/hessian/params when a SaveConfig flag is off).
#   - Batched-array internal shapes (expensive, fragile).
#   - ``metadata.extra["problem_spec"]`` schema (belongs to the typed
#     ProblemSpec work, plan #4).
#   - SaveConfig internal coherence (belongs to SaveConfig, separate PR).
_BEST_LOSS_ATOL: float = 1e-9


@dataclass
class RunStateValidationError:
    """A single invariant violation found on a :class:`RunState`.

    Attributes:
        field: Name of the ``RunState`` field (or ``"metadata.<sub>"``)
            that failed.
        invariant: Identifier of the invariant (e.g. ``"A3"``, ``"B2"``).
        detail: Human-readable explanation of the mismatch, including the
            offending values where useful.
    """

    field: str
    invariant: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.invariant}] {self.field}: {self.detail}"


@dataclass
class ValidationReport:
    """Aggregate result of validating one :class:`RunState`.

    Attributes:
        ok: ``True`` iff ``errors`` is empty.
        errors: Every violation found, in check order. Empty iff ``ok``.
    """

    ok: bool
    errors: list[RunStateValidationError]

    @classmethod
    def passing(cls) -> "ValidationReport":
        return cls(ok=True, errors=[])

    def raise_if_invalid(self) -> None:
        """Raise a :class:`RunStateValidationException` if not ``ok``.

        Convenience for call sites that want the raise-on-invalid behaviour
        without losing the multi-error report: the exception carries every
        violation in its ``errors`` attribute and its message lists them.
        """
        if self.ok:
            return
        raise RunStateValidationException(self.errors)


class RunStateValidationException(ValueError):
    """Raised when a :class:`RunState` fails invariant validation.

    Carries the full :class:`ValidationReport` error list on
    :attr:`errors` so callers can produce a machine-readable rejection
    report rather than parsing the message string.
    """

    def __init__(self, errors: list[RunStateValidationError]) -> None:
        self.errors: list[RunStateValidationError] = list(errors)
        lines = "\n".join(f"  - {e}" for e in self.errors)
        super().__init__(
            f"RunState failed {len(self.errors)} invariant check(s):\n{lines}"
        )


def _is_nonneg_int(x: Any) -> bool:
    return bool(isinstance(x, int | np.integer) and not isinstance(x, bool) and x >= 0)


def _is_int(x: Any) -> bool:
    return bool(isinstance(x, int | np.integer) and not isinstance(x, bool))


def _history_len(a: np.ndarray) -> int:
    """Length of a history array along the time (first) axis.

    Histories are stored either as 1-D object arrays (ragged/batched
    entries, the canonical serializer contract) or as N-D numeric arrays
    (when NumPy collapses uniform-shape object arrays, e.g. five (2,)
    gradients become shape (5, 2)). In both cases the first axis is the
    time axis. A 0-d array is treated as length 0 — it should never occur
    for a well-formed history but ``np.array([])`` is 1-d so this branch
    is defensive only.
    """
    arr = np.asarray(a)
    if arr.ndim == 0:
        return 0
    return int(arr.shape[0])


def _history_alignment_target(state: RunState) -> int:
    """Return the expected history row count for A3 alignment checks."""
    if _is_nonneg_int(state.log_call_count) and int(state.log_call_count) > 0:
        return int(state.log_call_count)
    return int(state.eval_count) if _is_nonneg_int(state.eval_count) else 0


def _check_structural(state: RunState) -> list[RunStateValidationError]:
    errs: list[RunStateValidationError] = []

    # A1 — eval_count is a non-negative int
    if not _is_nonneg_int(state.eval_count):
        errs.append(
            RunStateValidationError(
                field="eval_count",
                invariant="A1",
                detail=f"expected non-negative int, got {state.eval_count!r}",
            )
        )

    # A2 — the six aligned histories are numpy arrays. NumPy collapses
    # uniform-shape object arrays to N-D numeric arrays (e.g. a grad
    # history of five (2,)-vectors becomes shape (5, 2)), so we only
    # require an ndarray; A3 checks the *first* axis is the time axis.
    history_fields = (
        "loss_history",
        "grad_history",
        "hessian_history",
        "params_history",
        "eval_type_history",
        "time_steps",
        "sensitivity_loss_history",
        "penalty_history",
        "is_feasible_history",
        "violations_history",
        "power_hard_history",
        "power_soft_history",
        "power_detector_history",
    )
    for name in history_fields:
        arr = getattr(state, name)
        if not isinstance(arr, np.ndarray):
            errs.append(
                RunStateValidationError(
                    field=name,
                    invariant="A2",
                    detail=f"expected np.ndarray, got {type(arr).__name__}",
                )
            )

    # A4 — best_params is 1-D float64, or empty
    bp = state.best_params
    if not isinstance(bp, np.ndarray):
        errs.append(
            RunStateValidationError(
                field="best_params",
                invariant="A4",
                detail=f"expected np.ndarray, got {type(bp).__name__}",
            )
        )
    elif bp.ndim != 1:
        errs.append(
            RunStateValidationError(
                field="best_params",
                invariant="A4",
                detail=f"expected 1-D array, got {bp.ndim}-D shape {bp.shape}",
            )
        )

    # A5 — eval_type_counts keys are ints, values non-negative ints
    counts = state.eval_type_counts
    if not isinstance(counts, dict):
        errs.append(
            RunStateValidationError(
                field="eval_type_counts",
                invariant="A5",
                detail=f"expected dict, got {type(counts).__name__}",
            )
        )
    else:
        for k, v in counts.items():
            if not _is_int(k):
                errs.append(
                    RunStateValidationError(
                        field="eval_type_counts",
                        invariant="A5",
                        detail=f"key {k!r} is not an int",
                    )
                )
                break
            if not _is_nonneg_int(v):
                errs.append(
                    RunStateValidationError(
                        field="eval_type_counts",
                        invariant="A5",
                        detail=f"value for key {k} is not a non-negative int: {v!r}",
                    )
                )
                break

    # A6 — metadata is a RunMetadata
    if not isinstance(state.metadata, RunMetadata):
        errs.append(
            RunStateValidationError(
                field="metadata",
                invariant="A6",
                detail=f"expected RunMetadata, got {type(state.metadata).__name__}",
            )
        )

    # A7 — scalar counters are non-negative ints
    for name in ("improvement_count", "evals_since_improvement", "log_call_count"):
        v = getattr(state, name)
        if not _is_nonneg_int(v):
            errs.append(
                RunStateValidationError(
                    field=name,
                    invariant="A7",
                    detail=f"expected non-negative int, got {v!r}",
                )
            )

    return errs


def _check_aligned(
    state: RunState, errs: list[RunStateValidationError]
) -> dict[str, int]:
    """A3 — aligned lengths for non-empty histories.

    Histories may legitimately be empty (a SaveConfig flag is off, or the
    run was reduced by ``RunData.to_run_state`` which drops grad/hessian/
    eval_type). The rule is therefore: every *non-empty* history has
    length ``log_call_count``; empty is always allowed. ``eval_count`` counts
    individual parameter evaluations, while histories store one row per logged
    call and may contain batched entries. Legacy reduced states may have
    ``log_call_count == 0`` with non-empty histories; those fall back to
    ``eval_count``.

    Returns the lengths so semantic checks can reuse them without
    recomputing. Assumes the A2 ndarray check already ran; callers should
    skip A3 if A2 failed for a history (its length is unreliable then).
    """
    history_fields = (
        "loss_history",
        "grad_history",
        "hessian_history",
        "params_history",
        "eval_type_history",
        "time_steps",
        "sensitivity_loss_history",
        "penalty_history",
        "is_feasible_history",
        "violations_history",
        "power_hard_history",
        "power_soft_history",
        "power_detector_history",
    )
    lengths: dict[str, int] = {}
    bad_shapes = {e.field for e in errs if e.invariant == "A2"}
    target = _history_alignment_target(state)
    for name in history_fields:
        if name in bad_shapes:
            lengths[name] = -1
            continue
        n = _history_len(getattr(state, name))
        lengths[name] = n
        if n > 0 and n != target:
            errs.append(
                RunStateValidationError(
                    field=name,
                    invariant="A3",
                    detail=(
                        f"length {n} != log_call_count {target} "
                        "(empty is allowed; non-empty must match)"
                    ),
                )
            )
    return lengths


def _has_recorded_loss(loss_history: np.ndarray, loss_n: int) -> bool:
    """Return whether ``loss_history`` contains at least one non-NaN entry.

    A hessian-only or grad-only call increments ``eval_count`` and appends
    a NaN placeholder to ``loss_history`` without updating ``best_loss``;
    such a run legitimately has ``best_loss == inf`` with ``eval_count > 0``.
    The B-tier invariants therefore key on "was a real loss recorded",
    not on "did any eval fire".
    """
    if loss_n <= 0:
        return False
    arr = np.asarray(loss_history, dtype=object)
    for step in arr.tolist():
        a = np.asarray(step, dtype=np.float64)
        if a.size == 0:
            continue
        if not np.all(np.isnan(a)):
            return True
    return False


def _reduce_loss_min(loss_history: np.ndarray, loss_n: int) -> float | None:
    """Return ``nanmin`` over all per-step loss minima, or ``None`` if no
    non-NaN entry exists."""
    if loss_n <= 0:
        return None
    per_step = []
    for step in np.asarray(loss_history, dtype=object).tolist():
        a = np.asarray(step, dtype=np.float64)
        if a.size == 0:
            continue
        m = float(np.nanmin(a))
        if not np.isnan(m):
            per_step.append(m)
    return min(per_step) if per_step else None


def _check_semantic(
    state: RunState, lengths: dict[str, int]
) -> list[RunStateValidationError]:
    errs: list[RunStateValidationError] = []
    loss_n = lengths.get("loss_history", 0)
    has_loss = _has_recorded_loss(state.loss_history, loss_n)
    bl = float(state.best_loss)

    # B1 — best_loss is finite iff a non-NaN loss was recorded.
    # grad-only / hessian-only evals increment eval_count and append NaN
    # to loss_history without updating best_loss, so best_loss==inf with
    # eval_count>0 is legal as long as no real loss was recorded.
    if has_loss and np.isinf(bl):
        errs.append(
            RunStateValidationError(
                field="best_loss",
                invariant="B1",
                detail=(
                    "loss_history has non-NaN entries but best_loss is inf "
                    "(best was never updated from a loss-bearing eval)"
                ),
            )
        )
    if not has_loss and not np.isinf(bl):
        errs.append(
            RunStateValidationError(
                field="best_loss",
                invariant="B1",
                detail=(
                    f"loss_history has no non-NaN entry but best_loss={bl} "
                    "(best set without a recorded loss)"
                ),
            )
        )

    # B2 — best_loss == nanmin(loss_history) when a loss was recorded.
    # nanmin handles batched entries and NaN placeholders. Skipped when
    # no loss was recorded (B1 covers that case).
    if has_loss and not np.isinf(bl):
        try:
            expected = _reduce_loss_min(state.loss_history, loss_n)
            if expected is not None and abs(bl - expected) > _BEST_LOSS_ATOL:
                errs.append(
                    RunStateValidationError(
                        field="best_loss",
                        invariant="B2",
                        detail=(
                            f"best_loss={bl} != nanmin(loss_history)={expected} "
                            f"(atol={_BEST_LOSS_ATOL})"
                        ),
                    )
                )
        except (TypeError, ValueError) as exc:
            errs.append(
                RunStateValidationError(
                    field="loss_history",
                    invariant="B2",
                    detail=f"could not reduce loss_history to per-step minima: {exc}",
                )
            )

    # B4 — sum(eval_type_counts.values()) == log_call_count
    if isinstance(state.eval_type_counts, dict):
        total = sum(int(v) for v in state.eval_type_counts.values())
        if total != int(state.log_call_count):
            errs.append(
                RunStateValidationError(
                    field="eval_type_counts",
                    invariant="B4",
                    detail=(
                        f"sum(eval_type_counts.values())={total} != "
                        f"log_call_count={int(state.log_call_count)}"
                    ),
                )
            )

    # B5 — improvement_count <= log_call_count
    if int(state.improvement_count) > int(state.log_call_count):
        errs.append(
            RunStateValidationError(
                field="improvement_count",
                invariant="B5",
                detail=(
                    f"improvement_count={int(state.improvement_count)} > "
                    f"log_call_count={int(state.log_call_count)}"
                ),
            )
        )

    # B6 — evals_since_improvement <= eval_count
    if int(state.evals_since_improvement) > int(state.eval_count):
        errs.append(
            RunStateValidationError(
                field="evals_since_improvement",
                invariant="B6",
                detail=(
                    f"evals_since_improvement={int(state.evals_since_improvement)} > "
                    f"eval_count={int(state.eval_count)}"
                ),
            )
        )

    return errs


def _check_format_version(state: RunState, errs: list[RunStateValidationError]) -> None:
    # No-op: format_version is enforced at deserialization time (see
    # RunMetadata.from_dict and each serializer's deserialize). A constructed
    # RunMetadata always reports FORMAT_VERSION via to_dict, so this check
    # would be dead code. Kept as a stub for future per-field version gates.
    return None


def validate_run_state(state: RunState, *, strict: bool = True) -> ValidationReport:
    """Check the :class:`RunState` invariant contract.

    Args:
        state: The snapshot to validate.
        strict: When ``True`` (default), run both the structural and
            semantic tiers. When ``False``, skip the semantic tier — use
            this for partial mid-run snapshots where a B-tier invariant
            can transiently fail (e.g. ``best_params`` not yet set, or a
            ``RunData.to_run_state`` reduction that zeroes the counters).

    Returns:
        A :class:`ValidationReport` collecting every violation. The report
        is :class:`RunStateValidationException`-raiseable via
        :meth:`ValidationReport.raise_if_invalid`.

    This function never raises on a malformed state; it only raises if
    *itself* has a bug (in which case the exception comes from Python, not
    the validator).
    """
    errs = _check_structural(state)
    lengths = _check_aligned(state, errs)
    _check_format_version(state, errs)
    if strict:
        errs.extend(_check_semantic(state, lengths))
    return ValidationReport(ok=len(errs) == 0, errors=errs)
