"""Structured path construction for run artifacts.

A :class:`RunPathResolver` builds filesystem paths from semantic
components (problem name, algorithm name, hyperparameter string, budget
limits, timestamp) so that :class:`Objective` and
:class:`~dfbench.core.storage.manager.CheckpointManager` never hardcode
``./data/...`` strings. The root directory is configurable, letting users
redirect all artifacts to a scratch disk or an S3-prefix-backed backend
without editing library code.

The default layout mirrors the historical dfbench convention but is now
expressed in one place::

    {root}/{budget_dir}/{hyper_param_str}/{problem}_{algo}_{timestamp}.npz

where ``budget_dir`` is e.g. ``time100s_evals1000`` or ``unlimited``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _safe(s: str) -> str:
    """Make a string safe for use as a filesystem path component."""
    return s.replace("/", "_").replace(" ", "_").strip("_")


@dataclass
class RunPathResolver:
    """Build structured artifact paths under a configurable root.

    Attributes:
        root: Base directory for all artifacts. Defaults to
            ``./data/objective_run_data`` to match the historical layout.
        extension: File extension (without dot) for checkpoint files,
            e.g. ``"npz"`` or ``"json"``. Used by the manager to build the
            default filename; the serializer itself is format-agnostic to
            the extension.
    """

    root: str | Path = "./data/objective_run_data"
    extension: str = "npz"

    def checkpoint_dir(
        self,
        problem_name: str,
        algorithm_name: str,
        hyper_param_str: str | None = None,
        max_time: float | None = None,
        max_evals: int | None = None,
    ) -> Path:
        """Return the directory a checkpoint should live in.

        The directory is *not* created here; the storage backend creates
        parents as needed when it writes.
        """
        parts = []
        if max_time is not None:
            parts.append(f"time{int(max_time)}s")
        if max_evals is not None:
            parts.append(f"evals{max_evals}")
        budget_dir = "_".join(parts) if parts else "unlimited"

        d = Path(self.root) / budget_dir
        if hyper_param_str:
            d = d / _safe(hyper_param_str)
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
        """Return the full path for a checkpoint file."""
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
