"""Checkpoint manager: the facade :class:`Objective` talks to.

A :class:`CheckpointManager` wires together a serializer, a storage
backend, and a path resolver, and provides the high-level
``save``/``load``/``tick`` operations the Objective needs. This is the
*only* storage object the Objective holds, so swapping formats
(NPZ <-> JSON), locations (local disk <-> S3), or naming conventions is
a one-line change at construction time.

The manager also owns the periodic-checkpoint cadence (``save_every``)
and the wall-clock-exclusion timing, so the Objective's
``_log_to_file`` is reduced to a single ``tick()`` call. It owns the
cached checkpoint path so that periodic saves overwrite the same file
rather than creating timestamped duplicates, and it exposes
``last_checkpoint_eval`` and ``save_every`` for the display layer.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from dfbench.core.storage.backends import (
    LocalFilesystemBackend,
    StorageBackend,
)
from dfbench.core.storage.resolver import RunPathResolver
from dfbench.core.storage.serializers import (
    CheckpointSerializer,
    NpzCheckpointSerializer,
)
from dfbench.core.storage.state import (
    RunMetadata,
    RunState,
    validate_run_state,
)


class CheckpointManager:
    """Orchestrate checkpoint save/load for a run.

    Args:
        backend: Where bytes go. Defaults to a local filesystem backend
            rooted at ``./data/objective_run_data`` (the historical
            default). The backend has the storage root.
        serializer: How a :class:`RunState` is encoded. Defaults to the
            NPZ serializer.
        resolver: How artifact paths are built from components. Defaults
            to a :class:`RunPathResolver`, which returns relative paths
            that the backend combines with its root.
        save_every: Periodic checkpoint cadence in evaluations. ``None``
            disables auto-saving. The manager owns this so the Objective
            does not need to pass it on every call.
        validate_on_save: When ``True`` (default), run
            :func:`validate_run_state` (strict) before serializing a state
            and refuse to write a malformed artifact. Set to ``False`` for
            trusted batch replay where a known-legacy state may drift.
        validate_on_load: When ``True`` (default), run
            :func:`validate_run_state` (strict) after deserializing and
            refuse to surface a corrupted or tampered artifact to the
            scorer. Set to ``False`` to load legacy files that predate the
            invariant contract.
    """

    def __init__(
        self,
        backend: StorageBackend | None = None,
        serializer: CheckpointSerializer | None = None,
        resolver: RunPathResolver | None = None,
        save_every: int | None = None,
        validate_on_save: bool = True,
        validate_on_load: bool = True,
    ) -> None:
        self.resolver = resolver or RunPathResolver()
        self.backend: StorageBackend = backend or LocalFilesystemBackend(
            root="./data/objective_run_data"
        )
        self.serializer: CheckpointSerializer = serializer or NpzCheckpointSerializer()

        # Keep the resolver's extension in sync with the serializer's format
        # so checkpoint paths match the on-disk artifact (e.g. .json vs .npz).
        serializer_ext = getattr(self.serializer, "extension", None)
        if serializer_ext is not None:
            self.resolver.extension = serializer_ext

        self.save_every: int | None = save_every
        self.validate_on_save: bool = validate_on_save
        self.validate_on_load: bool = validate_on_load
        self._cached_path: Path | None = None
        self.last_checkpoint_eval: int | None = None

    # ------------------------------------------------------------------
    # path handling
    # ------------------------------------------------------------------

    def resolve_path(
        self,
        metadata: RunMetadata,
        timestamp: str,
        explicit_path: str | Path | None = None,
        hyper_param_str: str | None = None,
    ) -> Path:
        """Return the checkpoint path for ``metadata``.

        If ``explicit_path`` is given it wins (used by tests and custom
        layouts) and is passed straight to the backend. Otherwise the
        resolver builds a relative path, which the backend then anchors
        to its root. The first computed path is cached so subsequent
        periodic saves without overrides overwrite the same file.
        """
        if explicit_path is not None:
            return Path(explicit_path)

        hp = (
            hyper_param_str if hyper_param_str is not None else metadata.hyper_param_str
        )
        algo = metadata.algorithm_name or "unknown"
        problem = metadata.problem_name or "problem"

        return self.resolver.checkpoint_path(
            problem_name=problem,
            algorithm_name=algo,
            timestamp=timestamp,
            hyper_param_str=hp,
            max_time=metadata.max_time,
            max_evals=metadata.max_evals,
        )

    def _effective_path(
        self,
        metadata: RunMetadata,
        timestamp: str,
        explicit_path: str | Path | None,
        hyper_param_str: str | None,
    ) -> Path:
        """Compute the path, caching it when no override is provided."""
        if (
            explicit_path is None
            and hyper_param_str is None
            and self._cached_path is not None
        ):
            return self._cached_path
        path = self.resolve_path(metadata, timestamp, explicit_path, hyper_param_str)
        if explicit_path is None and hyper_param_str is None:
            self._cached_path = path
        return path

    # ------------------------------------------------------------------
    # save / load
    # ------------------------------------------------------------------

    def save(
        self,
        state: RunState,
        *,
        explicit_path: str | Path | None = None,
        hyper_param_str: str | None = None,
    ) -> Path:
        """Serialize and persist ``state``; return the on-disk path.

        The returned :class:`~pathlib.Path` is the absolute path the
        backend actually wrote to, so callers can ``exists()`` / ``open()``
        it directly. If neither ``explicit_path`` nor ``hyper_param_str``
        is given, the path computed from ``state.metadata`` is cached so
        later saves overwrite the same file.

        Raises:
            RunStateValidationException: If ``validate_on_save`` is enabled
                and ``state`` fails the strict invariant check.
        """
        if self.validate_on_save:
            validate_run_state(state, strict=True).raise_if_invalid()
        timestamp = state.metadata.timestamp
        key = self._effective_path(
            state.metadata, timestamp, explicit_path, hyper_param_str
        )
        data = self.serializer.serialize(state)
        self.backend.save_bytes(key, data)
        self.last_checkpoint_eval = state.eval_count
        return Path(self.backend.resolve(key))

    def load(self, path: str | Path) -> RunState:
        """Load and return a :class:`RunState` from ``path``.

        The path is cached so subsequent saves without overrides
        overwrite the same file (matches the historical resume-then-save
        behaviour). ``path`` is usually the absolute path returned by
        :meth:`save`.

        Raises:
            RunStateValidationException: If ``validate_on_load`` is enabled
                and the deserialized state fails the strict invariant
                check (corrupted or tampered artifact).
        """
        p = Path(path)
        data = self.backend.load_bytes(p)
        state = self.serializer.deserialize(data)
        if self.validate_on_load:
            validate_run_state(state, strict=True).raise_if_invalid()
        self._cached_path = p
        self.last_checkpoint_eval = state.eval_count
        return state

    # ------------------------------------------------------------------
    # periodic checkpointing
    # ------------------------------------------------------------------

    def should_checkpoint(self, eval_count: int) -> bool:
        """Return whether a periodic checkpoint is due at ``eval_count``."""
        if self.save_every is None or self.save_every <= 0:
            return False
        return eval_count % self.save_every == 0

    def tick(
        self,
        eval_count: int,
        state_factory: Callable[[], RunState],
    ) -> float:
        """Periodic checkpoint hook called by the Objective after each eval.

        If a checkpoint is due (per ``save_every``), builds a
        :class:`RunState` via ``state_factory`` and saves it. Returns the
        wall-clock time spent saving (0.0 if no checkpoint was taken) so
        the caller can exclude that duration from its elapsed-time clock.

        ``state_factory`` is called lazily so building the (potentially
        large) snapshot is skipped when no checkpoint is due.
        """
        if not self.should_checkpoint(eval_count):
            return 0.0
        t0 = time.time()
        self.save(state_factory())
        return time.time() - t0

    # ------------------------------------------------------------------
    # problem reconstruction
    # ------------------------------------------------------------------

    @staticmethod
    def extract_problem_spec(state: RunState) -> dict | None:
        """Return the embedded ``problem_spec`` dict from a loaded state, if any.

        The returned dict is the raw form stored in
        ``metadata.extra["problem_spec"]`` — either the typed container
        (``{"type", "version", "params"}``) or the legacy flat form
        (``{"type", <kwargs>}``). Callers that want the typed
        :class:`~dfbench.core.problem.ProblemSpec` container should
        normalize it via ``ProblemSpec.from_dict(spec)``; callers that
        want a rebuilt problem should call
        :func:`dfbench.core.problem.build_problem_from_spec(spec)`.

        Problem *reconstruction* lives in :mod:`dfbench.core.problem`,
        not here. Storage extracts the spec; the problem layer owns
        rebuilding it.
        """
        return state.metadata.extra.get("problem_spec")
