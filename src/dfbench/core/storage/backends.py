"""Storage backends: where serialized bytes physically go.

A :class:`StorageBackend` is a tiny protocol (``save_bytes`` /
``load_bytes`` / ``exists`` / ``delete`` / ``resolve``) so that the local
filesystem can be swapped for memory, S3, etc. without touching serializers
or the :class:`~dfbench.core.storage.manager.CheckpointManager`. The
backend owns the storage root (a directory on disk, an S3 prefix, etc.).
Callers just hand it opaque keys and use :meth:`resolve` to get the actual
location of a stored artifact back.

The default :class:`LocalFilesystemBackend` performs an **atomic write**
(write to a sibling temp file then :func:`os.replace`) and, unlike the
previous fallback, never destroys an existing good file before the new
one is in place. If ``os.replace`` fails the temp file is left in place
and the exception propagates, so no data is silently lost.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class StorageBackend(Protocol):
    """Minimal byte-level storage abstraction."""

    def save_bytes(self, key: str | Path, data: bytes) -> None:
        """Persist ``data`` under ``key`` atomically (best effort)."""
        ...

    def load_bytes(self, key: str | Path) -> bytes:
        """Read bytes previously stored under ``key``."""
        ...

    def exists(self, key: str | Path) -> bool:
        """Return whether ``key`` is present in the backend."""
        ...

    def delete(self, key: str | Path) -> None:
        """Remove ``key`` from the backend (no error if missing)."""
        ...

    def resolve(self, key: str | Path) -> Path | str:
        """Return where ``key`` is physically stored.

        A filesystem backend returns an absolute :class:`~pathlib.Path`;
        a non-filesystem backend returns whatever logical identifier makes
        sense for it (a URI, a memory id, etc.).
        :class:`~dfbench.core.storage.manager.CheckpointManager.save`
        uses this to hand callers back a path they can ``exists()`` /
        ``open()`` directly.
        """
        ...


class LocalFilesystemBackend:
    """Filesystem-backed :class:`StorageBackend`.

    Writes are atomic: data is first written to a temporary file in the
    *same directory* as the target (so :func:`os.replace` stays on one
    filesystem) and then renamed into place with :func:`os.replace`. The
    temp file uses ``.tmp`` + a random suffix so it never collides with a
    real checkpoint and is easy to spot/clean up after a crash.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        """Initialize the backend.

        Args:
            root: Optional base directory. If given, relative ``key``
                arguments are resolved against it. Absolute keys are
                used verbatim. ``None`` (default) means keys are used as
                given (cwd-relative).
        """
        self._root = Path(root) if root is not None else None

    @property
    def root(self) -> Path | None:
        return self._root

    def _resolve(self, key: str | Path) -> Path:
        p = Path(key)
        if self._root is not None and not p.is_absolute():
            p = self._root / p
        return p

    def save_bytes(self, key: str | Path, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, path)
        except Exception:
            # Best-effort cleanup of the temp file on failure; do not
            # touch the destination, so a previous good file survives.
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            raise

    def load_bytes(self, key: str | Path) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Storage key not found: {path}")
        return path.read_bytes()

    def exists(self, key: str | Path) -> bool:
        return self._resolve(key).exists()

    def delete(self, key: str | Path) -> None:
        self._resolve(key).unlink(missing_ok=True)

    def resolve(self, key: str | Path) -> Path:
        """Return the absolute on-disk path where ``key`` is.

        Calls :func:`pathlib.Path.resolve` so a relative backend root
        still produces an absolute path. That absolute path then
        goes through :meth:`_resolve` thanks to ``is_absolute()``.
        """
        return self._resolve(key).resolve()
