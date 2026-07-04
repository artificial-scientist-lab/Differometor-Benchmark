"""Checkpoint serializers: how a :class:`RunState` is encoded on disk.

Two implementations are provided:

* :class:`NpzCheckpointSerializer` - compressed NumPy ``.npz``. This is the
  default and matches the historical dfbench format, but now writes a
  ``format_version`` key and a JSON-encoded metadata sidecar *inside* the
  same NPZ so a single file is fully self-describing. It still uses
  ``dtype=object`` arrays for ragged/batched histories, but **never** uses
  ``allow_pickle=True`` for untrusted external data: object arrays hold
  only numeric ``numpy.ndarray``/``jnp.ndarray`` converted to ``ndarray``
  before saving, so the pickle path is only exercised on arrays we
  constructed ourselves.
* :class:`JsonCheckpointSerializer` - a fully pickle-free JSON format
  (histories encoded as nested lists). Slower and larger, but trivially
  inspectable and safe to load from untrusted sources.

Both produce ``bytes`` consumed by a :class:`StorageBackend` and accept
``bytes`` when decoding, so they are fully decoupled from the filesystem.
"""

from __future__ import annotations

import io
import json
from typing import Protocol, runtime_checkable

import numpy as np

from dfbench.core.storage.state import (
    FORMAT_VERSION,
    RunMetadata,
    RunState,
)


@runtime_checkable
class CheckpointSerializer(Protocol):
    """Encode/decode a :class:`RunState` to/from bytes."""

    #: File extension (without dot) implied by this serializer's format,
    #: used by :class:`CheckpointManager` to keep the path resolver in sync.
    extension: str

    def serialize(self, state: RunState) -> bytes:
        """Encode ``state`` to a byte string."""
        ...

    def deserialize(self, data: bytes) -> RunState:
        """Decode a byte string produced by :meth:`serialize`."""
        ...


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    """Convert a JAX array, scalar, or list to a numpy array."""
    if isinstance(x, np.ndarray):
        return x
    if x is None:
        return np.array([])
    return np.asarray(x)


def _empty_object_array() -> np.ndarray:
    return np.array([], dtype=object)


class NpzCheckpointSerializer:
    """Compressed-NPZ :class:`CheckpointSerializer`.

    The on-disk artifact is a single ``.npz`` containing the numeric
    histories plus a ``metadata`` JSON string and a ``format_version``
    scalar. Backwards-compatible with files written by older dfbench
    versions that lack ``metadata`` / ``format_version`` / some optional
    histories: missing keys fall back to empty defaults.
    """

    extension: str = "npz"

    def serialize(self, state: RunState) -> bytes:
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            format_version=np.array(FORMAT_VERSION, dtype=np.int64),
            metadata=np.array(json.dumps(state.metadata.to_dict())),
            loss_history=np.asarray(state.loss_history, dtype=object),
            grad_history=np.asarray(state.grad_history, dtype=object),
            hessian_history=np.asarray(state.hessian_history, dtype=object),
            params_history=np.asarray(state.params_history, dtype=object),
            eval_type_history=np.asarray(state.eval_type_history, dtype=object),
            time_steps=np.asarray(state.time_steps, dtype=object),
            sensitivity_loss_history=np.asarray(
                state.sensitivity_loss_history, dtype=object
            ),
            penalty_history=np.asarray(state.penalty_history, dtype=object),
            is_feasible_history=np.asarray(state.is_feasible_history, dtype=object),
            violations_history=np.asarray(state.violations_history, dtype=object),
            power_hard_history=np.asarray(state.power_hard_history, dtype=object),
            power_soft_history=np.asarray(state.power_soft_history, dtype=object),
            power_detector_history=np.asarray(
                state.power_detector_history, dtype=object
            ),
            eval_count=np.array(state.eval_count, dtype=np.int64),
            best_loss=np.asarray(state.best_loss, dtype=np.float64),
            best_params=(
                np.asarray(state.best_params, dtype=np.float64)
                if state.best_params.size > 0
                else np.array([], dtype=np.float64)
            ),
            improvement_count=np.array(state.improvement_count, dtype=np.int64),
            evals_since_improvement=np.array(
                state.evals_since_improvement, dtype=np.int64
            ),
            best_eval_index=(
                np.array(state.best_eval_index, dtype=np.int64)
                if state.best_eval_index is not None
                else np.array([], dtype=np.int64)
            ),
            best_batch_index=(
                np.array(state.best_batch_index, dtype=np.int64)
                if state.best_batch_index is not None
                else np.array([], dtype=np.int64)
            ),
            log_call_count=np.array(state.log_call_count, dtype=np.int64),
            eval_type_counts=np.array(
                json.dumps({str(k): v for k, v in state.eval_type_counts.items()})
            ),
        )
        return buffer.getvalue()

    def deserialize(self, data: bytes) -> RunState:
        buffer = io.BytesIO(data)
        with np.load(buffer, allow_pickle=True) as d:
            files = set(d.files)

            # Version check
            if "format_version" in files:
                version = int(d["format_version"])
                if version > FORMAT_VERSION:
                    raise ValueError(
                        f"Run data format version {version} is newer than "
                        f"supported {FORMAT_VERSION}. Please update dfbench."
                    )
            # else: legacy file with no version -> load best-effort.

            if "metadata" in files:
                metadata = RunMetadata.from_dict(json.loads(str(d["metadata"])))
            else:
                metadata = RunMetadata()

            def _obj(key: str) -> np.ndarray:
                if key in files:
                    return d[key]
                return _empty_object_array()

            # Rebuild eval_type_counts dict — single source of truth is the
            # JSON-encoded "eval_type_counts" field.  For legacy files that
            # predate this field, fall back to rebuilding from
            # eval_type_history.
            counts: dict[int, int]
            if "eval_type_counts" in files:
                counts = {
                    int(k): int(v)
                    for k, v in json.loads(str(d["eval_type_counts"])).items()
                }
            elif "eval_type_history" in files:
                et = d["eval_type_history"].tolist()
                counts = {}
                for k in et:
                    counts[k] = counts.get(k, 0) + 1
            else:
                counts = {}

            # best_eval_index / best_batch_index: stored as empty int arrays
            # when None; reconstruct as None when the loaded array is empty.
            bei = (
                int(d["best_eval_index"].item())
                if "best_eval_index" in files and d["best_eval_index"].size > 0
                else None
            )
            bbi = (
                int(d["best_batch_index"].item())
                if "best_batch_index" in files and d["best_batch_index"].size > 0
                else None
            )

            return RunState(
                loss_history=_obj("loss_history"),
                grad_history=_obj("grad_history"),
                hessian_history=_obj("hessian_history"),
                params_history=_obj("params_history"),
                eval_type_history=_obj("eval_type_history"),
                time_steps=_obj("time_steps"),
                sensitivity_loss_history=_obj("sensitivity_loss_history"),
                penalty_history=_obj("penalty_history"),
                is_feasible_history=_obj("is_feasible_history"),
                violations_history=_obj("violations_history"),
                power_hard_history=_obj("power_hard_history"),
                power_soft_history=_obj("power_soft_history"),
                power_detector_history=_obj("power_detector_history"),
                eval_count=int(d["eval_count"]) if "eval_count" in files else 0,
                best_loss=float(d["best_loss"])
                if "best_loss" in files
                else float("inf"),
                best_params=(
                    np.asarray(d["best_params"], dtype=np.float64)
                    if "best_params" in files and d["best_params"].size > 0
                    else np.array([], dtype=np.float64)
                ),
                improvement_count=(
                    int(d["improvement_count"]) if "improvement_count" in files else 0
                ),
                evals_since_improvement=(
                    int(d["evals_since_improvement"])
                    if "evals_since_improvement" in files
                    else 0
                ),
                best_eval_index=bei,
                best_batch_index=bbi,
                log_call_count=(
                    int(d["log_call_count"]) if "log_call_count" in files else 0
                ),
                eval_type_counts=counts,
                metadata=metadata,
            )


class JsonCheckpointSerializer:
    """Fully pickle-free JSON :class:`CheckpointSerializer`.

    Histories are stored as nested lists. Slower and larger than NPZ but
    safe to load from untrusted sources and trivially inspectable.
    """

    extension: str = "json"

    def serialize(self, state: RunState) -> bytes:
        def _tolist(a: np.ndarray):
            a = np.asarray(a)
            if a.dtype == object:
                return [np.asarray(x).tolist() for x in a.tolist()]
            return a.tolist()

        payload = {
            "format_version": FORMAT_VERSION,
            "metadata": state.metadata.to_dict(),
            "loss_history": _tolist(state.loss_history),
            "grad_history": _tolist(state.grad_history),
            "hessian_history": _tolist(state.hessian_history),
            "params_history": _tolist(state.params_history),
            "eval_type_history": _tolist(state.eval_type_history),
            "time_steps": _tolist(state.time_steps),
            "sensitivity_loss_history": _tolist(state.sensitivity_loss_history),
            "penalty_history": _tolist(state.penalty_history),
            "is_feasible_history": _tolist(state.is_feasible_history),
            "violations_history": _tolist(state.violations_history),
            "power_hard_history": _tolist(state.power_hard_history),
            "power_soft_history": _tolist(state.power_soft_history),
            "power_detector_history": _tolist(state.power_detector_history),
            "eval_count": state.eval_count,
            "best_loss": state.best_loss,
            "best_params": (
                np.asarray(state.best_params, dtype=np.float64).tolist()
                if state.best_params.size > 0
                else []
            ),
            "improvement_count": state.improvement_count,
            "evals_since_improvement": state.evals_since_improvement,
            "best_eval_index": state.best_eval_index,
            "best_batch_index": state.best_batch_index,
            "log_call_count": state.log_call_count,
            "eval_type_counts": {str(k): v for k, v in state.eval_type_counts.items()},
        }
        return json.dumps(payload, indent=2).encode("utf-8")

    def deserialize(self, data: bytes) -> RunState:
        payload = json.loads(data.decode("utf-8"))
        version = payload.get("format_version", 1)
        if int(version) > FORMAT_VERSION:
            raise ValueError(
                f"Run data format version {version} is newer than "
                f"supported {FORMAT_VERSION}. Please update dfbench."
            )
        metadata = RunMetadata.from_dict(payload.get("metadata", {}))

        def _to_obj_array(val):
            arr = np.array(val, dtype=object)
            return arr

        return RunState(
            loss_history=_to_obj_array(payload.get("loss_history", [])),
            grad_history=_to_obj_array(payload.get("grad_history", [])),
            hessian_history=_to_obj_array(payload.get("hessian_history", [])),
            params_history=_to_obj_array(payload.get("params_history", [])),
            eval_type_history=_to_obj_array(payload.get("eval_type_history", [])),
            time_steps=np.asarray(payload.get("time_steps", []), dtype=object),
            sensitivity_loss_history=_to_obj_array(
                payload.get("sensitivity_loss_history", [])
            ),
            penalty_history=_to_obj_array(payload.get("penalty_history", [])),
            is_feasible_history=_to_obj_array(payload.get("is_feasible_history", [])),
            violations_history=_to_obj_array(payload.get("violations_history", [])),
            power_hard_history=_to_obj_array(payload.get("power_hard_history", [])),
            power_soft_history=_to_obj_array(payload.get("power_soft_history", [])),
            power_detector_history=_to_obj_array(
                payload.get("power_detector_history", [])
            ),
            eval_count=int(payload.get("eval_count", 0)),
            best_loss=float(payload.get("best_loss", float("inf"))),
            best_params=np.asarray(payload.get("best_params", []), dtype=np.float64),
            improvement_count=int(payload.get("improvement_count", 0)),
            evals_since_improvement=int(payload.get("evals_since_improvement", 0)),
            best_eval_index=payload.get("best_eval_index"),
            best_batch_index=payload.get("best_batch_index"),
            log_call_count=int(payload.get("log_call_count", 0)),
            eval_type_counts={
                int(k): int(v) for k, v in payload.get("eval_type_counts", {}).items()
            },
            metadata=metadata,
        )


# ------------------------------------------------------------------
# Multi-run collection serializer (used by Benchmark)
# ------------------------------------------------------------------


@runtime_checkable
class RunCollectionSerializer(Protocol):
    """Encode/decode a collection of :class:`RunState` objects.

    Used by :class:`~dfbench.benchmark.benchmark.Benchmark` to persist all
    runs for one algorithm configuration as a single self-describing
    artifact. The collection carries the algorithm name, hyperparameters,
    and one :class:`RunState` per run — each encoded by a per-run
    :class:`CheckpointSerializer`.
    """

    extension: str

    def serialize_collection(
        self,
        algorithm_name: str,
        hyperparameters: dict,
        runs: list[RunState],
    ) -> bytes: ...

    def deserialize_collection(self, data: bytes) -> tuple[str, dict, list[RunState]]:
        """Return ``(algorithm_name, hyperparameters, runs)``."""
        ...


class NpzRunCollectionSerializer:
    """NPZ-backed :class:`RunCollectionSerializer`.

    Packs all runs into a single compressed ``.npz``. Each run's full
    :class:`RunState` is serialized by the per-run ``run_serializer``
    (default :class:`NpzCheckpointSerializer`) and stored as a base64
    string, so the per-run schema (including ``format_version``,
    ``metadata``, and embedded ``problem_spec``) is preserved exactly.
    A ``format_version`` and a ``collection_metadata`` JSON sidecar make
    the collection self-describing.
    """

    extension: str = "npz"

    def __init__(self, run_serializer: CheckpointSerializer | None = None) -> None:
        self._run_serializer = run_serializer or NpzCheckpointSerializer()

    def serialize_collection(
        self,
        algorithm_name: str,
        hyperparameters: dict,
        runs: list[RunState],
    ) -> bytes:
        import base64

        run_blobs = [
            base64.b64encode(self._run_serializer.serialize(r)).decode("ascii")
            for r in runs
        ]
        collection_meta = {
            "algorithm_name": algorithm_name,
            "hyperparameters": hyperparameters,
            "n_runs": len(runs),
        }
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            collection_format_version=np.array(FORMAT_VERSION, dtype=np.int64),
            collection_metadata=np.array(json.dumps(collection_meta)),
            run_blobs=np.array(run_blobs, dtype=object),
        )
        return buffer.getvalue()

    def deserialize_collection(self, data: bytes) -> tuple[str, dict, list[RunState]]:
        import base64

        buffer = io.BytesIO(data)
        with np.load(buffer, allow_pickle=True) as d:
            files = set(d.files)
            if "collection_format_version" in files:
                version = int(d["collection_format_version"])
                if version > FORMAT_VERSION:
                    raise ValueError(
                        f"Collection format version {version} is newer than "
                        f"supported {FORMAT_VERSION}. Please update dfbench."
                    )
            meta = json.loads(str(d["collection_metadata"]))
            algorithm_name = meta["algorithm_name"]
            hyperparameters = meta.get("hyperparameters", {})
            blobs = d["run_blobs"].tolist()
            runs = [
                self._run_serializer.deserialize(base64.b64decode(b)) for b in blobs
            ]
        return algorithm_name, hyperparameters, runs
