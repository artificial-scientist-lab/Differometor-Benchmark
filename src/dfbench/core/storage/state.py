"""Canonical in-memory representation of an optimization run.

A :class:`RunState` is a plain dataclass holding everything needed to
checkpoint or export a run. It is deliberately independent of the
:class:`~dfbench.core.objective.Objective` class so that serializers,
exporters, and tests can operate on it without importing the Objective.

The companion :class:`RunMetadata` carries small, human-readable
descriptors (problem name, algorithm name, hyperparameter string, budget
limits, timestamp) that travel alongside the numeric histories.
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
    """

    # Aligned histories (all length == eval_count, modulo placeholders)
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
