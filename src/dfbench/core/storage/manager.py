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
from dfbench.core.storage.state import RunMetadata, RunState


class CheckpointManager:
    """Orchestrate checkpoint save/load for a run.

    Args:
        backend: Where bytes go. Defaults to a local filesystem backend
            rooted at the resolver's ``root``.
        serializer: How a :class:`RunState` is encoded. Defaults to the
            NPZ serializer.
        resolver: How paths are built from components. Defaults to the
            historical ``./data/objective_run_data`` layout.
        save_every: Periodic checkpoint cadence in evaluations. ``None``
            disables auto-saving. The manager owns this so the Objective
            does not need to pass it on every call.
    """

    def __init__(
        self,
        backend: StorageBackend | None = None,
        serializer: CheckpointSerializer | None = None,
        resolver: RunPathResolver | None = None,
        save_every: int | None = None,
    ) -> None:
        self.resolver = resolver or RunPathResolver()
        self.backend: StorageBackend = backend or LocalFilesystemBackend(
            root=self.resolver.root
        )
        self.serializer: CheckpointSerializer = serializer or NpzCheckpointSerializer()

        # Keep the resolver's extension in sync with the serializer's format
        # so checkpoint paths match the on-disk artifact (e.g. .json vs .npz).
        serializer_ext = getattr(self.serializer, "extension", None)
        if serializer_ext is not None:
            self.resolver.extension = serializer_ext

        self.save_every: int | None = save_every
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
        layouts). Otherwise the resolver builds the structured path. The
        first computed structured path is cached so subsequent periodic
        saves without overrides overwrite the same file.
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
        """Serialize and persist ``state``; return the written path.

        If neither ``explicit_path`` nor ``hyper_param_str`` is given, the
        path computed from ``state.metadata`` is cached so later saves
        overwrite the same file.
        """
        timestamp = state.metadata.timestamp
        path = self._effective_path(
            state.metadata, timestamp, explicit_path, hyper_param_str
        )
        data = self.serializer.serialize(state)
        self.backend.save_bytes(path, data)
        self.last_checkpoint_eval = state.eval_count
        return path

    def load(self, path: str | Path) -> RunState:
        """Load and return a :class:`RunState` from ``path``.

        The path is cached so subsequent saves without overrides
        overwrite the same file (matches the historical resume-then-save
        behaviour).
        """
        p = Path(path)
        data = self.backend.load_bytes(p)
        state = self.serializer.deserialize(data)
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
        """Return the embedded ``problem_spec`` from a loaded state, if any."""
        return state.metadata.extra.get("problem_spec")

    @staticmethod
    def reconstruct_problem(state: RunState):
        """Rebuild the :class:`ContinuousProblem` recorded in ``state``.

        Returns ``None`` if the run did not record a problem spec (e.g. the
        problem did not implement ``to_spec``). Requires the relevant
        problem module to be imported so its class is registered.
        """
        spec = CheckpointManager.extract_problem_spec(state)
        if spec is None:
            return None
        from dfbench.core.problem import build_problem_from_spec

        return build_problem_from_spec(spec)
