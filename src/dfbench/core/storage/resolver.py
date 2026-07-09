"""Structured key construction for run artifacts.

A :class:`RunPathResolver` builds relative paths from semantic components
(problem name, algorithm name, hyperparameter string, budget limits,
timestamp) so that :class:`Objective` and
:class:`~dfbench.core.storage.manager.CheckpointManager` never hardcode
``./data/...`` strings. The actual storage root (a directory on disk, an
S3 prefix, etc.) is in the :class:`StorageBackend`, not the resolver;
the resolver just emits relative paths that the backend combines with its
root.

    The default layout is flat inside the budget directory::

    {budget_dir}/{problem}_{algo}_{hyper_param_str}_{timestamp}.npz

    Set ``algo_directory=True`` to insert an ``{algo}_{hyper_param_str}``
    segment after ``{budget_dir}`` (collapsing to just ``{algo}`` when
    ``hyper_param_str`` is empty/None). ``budget_dir`` is e.g.
    ``time100s_evals1000`` or ``unlimited``. The algorithm and
    hyperparameter string are always part of the filename, so a file
    stays self-describing even when copied out of its directory.
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

    The path returned is relative; the storage backend is what anchors it
    to a concrete location on disk or in some other store.

    Attributes:
        extension: File extension (without dot) for checkpoint files,
            e.g. ``"npz"`` or ``"json"``. Used by the manager to build the
            default filename; the serializer itself is format-agnostic to
            the extension.
        algo_directory: When ``True``, insert an ``{algo}_{hyper_param_str}``
            directory segment between the budget directory and the file
            (collapsing to just ``{algo}`` when ``hyper_param_str`` is
            empty/None). When ``False`` (default), the layout is flat
            inside the budget directory. The algorithm and hyperparameter
            string are always part of the filename either way, so a file
            is self-describing even when copied out of its directory.
    """

    extension: str = "npz"
    algo_directory: bool = False

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
        if self.algo_directory:
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
        hp_fmt = f"_{_safe(hyper_param_str)}" if hyper_param_str else ""
        filename = (
            f"{_safe(problem_name)}_{_safe(algorithm_name)}{hp_fmt}_{timestamp}"
            f".{self.extension}"
        )
        return d / filename

    @staticmethod
    def safe_component(s: str) -> str:
        """Public helper exposing the path-safety transform."""
        return _safe(s)
