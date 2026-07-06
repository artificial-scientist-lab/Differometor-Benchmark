"""Structured key construction for run artifacts.

A :class:`RunPathResolver` builds relative paths from semantic components
(problem name, algorithm name, hyperparameter string, budget limits,
timestamp) so that :class:`Objective` and
:class:`~dfbench.core.storage.manager.CheckpointManager` never hardcode
``./data/...`` strings. The actual storage root (a directory on disk, an
S3 prefix, etc.) is in the :class:`StorageBackend`, not the resolver;
the resolver just emits relative paths that the backend combines with its
root.

    The layout mirrors the historical dfbench convention but is now
    expressed in one place::

    {budget_dir}/{algo}_{hyper_param_str}/{problem}_{algo}_{timestamp}.npz

    where ``budget_dir`` is e.g. ``time100s_evals1000`` or ``unlimited``.
    When ``hyper_param_str`` is empty/None the segment collapses to just
    ``{algo}`` (no trailing underscore).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _safe(s: str) -> str:
    """Make a string safe for use as a filesystem path component."""
    return s.replace("/", "_").replace(" ", "_").strip("_")


@dataclass
class RunPathResolver:
    """Build structured artifact keys (relative paths).

    The path returned is relative, the storage backend gives a concrete location 
    on disk or in some other storage.

    Attributes:
        extension: File extension (without dot) for checkpoint files,
            e.g. ``"npz"`` or ``"json"``. Used by the manager to build the
            default filename; the serializer itself is format-agnostic to
            the extension.
    """

    extension: str = "npz"

    def checkpoint_dir(
        self,
        problem_name: str,
        algorithm_name: str,
        hyper_param_str: str | None = None,
        max_time: float | None = None,
        max_evals: int | None = None,
    ) -> Path:
        """Return the relative path for a checkpoint directory.

        Nothing is created here; the storage backend makes the parent
        directories when it writes. The returned path is rooted by the
        backend, not by the resolver.
        """
        parts = []
        if max_time is not None:
            parts.append(f"time{int(max_time)}s")
        if max_evals is not None:
            parts.append(f"evals{max_evals}")
        budget_dir = "_".join(parts) if parts else "unlimited"

        d = Path(budget_dir)
        if hyper_param_str:
            d = d / f"{_safe(algorithm_name)}_{_safe(hyper_param_str)}"
        elif algorithm_name:
            d = d / _safe(algorithm_name)
        return d

    def checkpoint_path(
        self,
        problem_name: str,
        algorithm_name: str,
        timestamp: str,
        hyper_param_str: str | None = None,
        max_time: float | None = None,
        max_evals: int | None = None,
    ) -> Path:
        """Return the relative path for a checkpoint file."""
        d = self.checkpoint_dir(
            problem_name,
            algorithm_name,
            hyper_param_str,
            max_time,
            max_evals,
        )
        filename = (
            f"{_safe(problem_name)}_{_safe(algorithm_name)}_{timestamp}"
            f".{self.extension}"
        )
        return d / filename

    @staticmethod
    def safe_component(s: str) -> str:
        """Public helper exposing the path-safety transform."""
        return _safe(s)
