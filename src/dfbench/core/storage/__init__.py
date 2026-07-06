"""Modular storage, checkpointing, and export for optimization runs.

This package decouples *what* is saved (the :class:`RunState` data contract)
from *how* and *where* it is saved:

* :class:`RunState`: plain dataclass snapshot of an optimization run,
  independent of the :class:`~dfbench.core.objective.Objective` class.
* :class:`CheckpointSerializer` / :class:`RunDataExporter`: format
  strategies (NPZ, JSON, PNG) behind small protocols.
* :class:`StorageBackend`: where bytes go (local filesystem by default,
  easily swapped for S3 / memory / etc.).
* :class:`RunPathResolver`: builds structured paths from components so no
  ``./data/...`` string is hardcoded in :class:`Objective`.
* :class:`CheckpointManager`: the single facade :class:`Objective` calls;
  it wires a serializer, backend, and resolver together and drives the
  periodic-save / load lifecycle.

Typical usage from inside an Objective::

    from dfbench.core.storage import CheckpointManager, LocalFilesystemBackend

    manager = CheckpointManager(
        backend=LocalFilesystemBackend(root="./data/objective_run_data"),
        serializer=NpzCheckpointSerializer(),
        resolver=RunPathResolver(),
    )
    path = manager.save(state)   # absolute Path on disk
    state = manager.load(path)
"""

from dfbench.core.storage.state import (
    FORMAT_VERSION,
    RunMetadata,
    RunState,
    RunStateValidationError,
    RunStateValidationException,
    ValidationReport,
    validate_run_state,
)
from dfbench.core.storage.saveconfig import SaveConfig
from dfbench.core.storage.backends import (
    StorageBackend,
    LocalFilesystemBackend,
)
from dfbench.core.storage.serializers import (
    CheckpointSerializer,
    NpzCheckpointSerializer,
    JsonCheckpointSerializer,
    RunCollectionSerializer,
    NpzRunCollectionSerializer,
)
from dfbench.core.storage.resolver import RunPathResolver
from dfbench.core.storage.exporter import RunDataExporter
from dfbench.core.storage.manager import CheckpointManager

__all__ = [
    "RunState",
    "RunMetadata",
    "SaveConfig",
    "StorageBackend",
    "LocalFilesystemBackend",
    "CheckpointSerializer",
    "NpzCheckpointSerializer",
    "JsonCheckpointSerializer",
    "RunCollectionSerializer",
    "NpzRunCollectionSerializer",
    "RunPathResolver",
    "RunDataExporter",
    "CheckpointManager",
    # Invariant contract
    "FORMAT_VERSION",
    "RunStateValidationError",
    "RunStateValidationException",
    "ValidationReport",
    "validate_run_state",
]
