"""Optimization run state for ask-tell optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dfbench.core.parameter_config import ParameterConfig


@dataclass
class OptimizationState:
    """Serializable optimizer-side state attached to an Objective.

    Objective remains responsible for evaluation histories and budget tracking.
    This object records the run context and ask/tell-side progress that is not
    inherently part of evaluating the objective function.
    """

    algorithm_name: str
    problem_name: str
    seed: int | None = None
    unbounded: bool = False
    algorithm_metadata: dict[str, Any] = field(default_factory=dict)
    problem_metadata: dict[str, Any] = field(default_factory=dict)
    search_space: dict[str, Any] | None = None
    current_config: ParameterConfig | None = None
    current_loss: Any | None = None
    best_config: ParameterConfig | None = None
    best_loss: float | None = None
    ask_count: int = 0
    tell_count: int = 0
    optimizer_state: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_objective(
        cls,
        objective: Any,
        *,
        algorithm_name: str,
        seed: int | None = None,
        unbounded: bool | None = None,
        algorithm_metadata: dict[str, Any] | None = None,
    ) -> "OptimizationState":
        """Build a state snapshot from an Objective-like object."""
        problem = objective.problem
        search_space_obj = getattr(problem, "search_space", None)
        search_space = (
            search_space_obj.to_dict()
            if search_space_obj is not None and hasattr(search_space_obj, "to_dict")
            else None
        )
        problem_metadata: dict[str, Any] = {}
        if search_space and isinstance(search_space.get("metadata"), dict):
            problem_metadata.update(search_space["metadata"])
        structure_info = getattr(problem, "structure_info", None)
        if structure_info is not None:
            problem_metadata.update(
                structure_info() if callable(structure_info) else structure_info
            )

        return cls(
            algorithm_name=algorithm_name,
            problem_name=getattr(problem, "name", problem.__class__.__name__),
            seed=seed,
            unbounded=objective.unbounded if unbounded is None else unbounded,
            algorithm_metadata=algorithm_metadata or {},
            problem_metadata=problem_metadata,
            search_space=search_space,
        )

    def record_ask(self, config: ParameterConfig | None = None) -> None:
        """Record a candidate proposal."""
        self.ask_count += 1
        if config is not None:
            self.current_config = config

    def record_tell(
        self,
        *,
        config: ParameterConfig | None = None,
        loss: Any | None = None,
    ) -> None:
        """Record feedback received by the optimizer."""
        self.tell_count += 1
        if config is not None:
            self.current_config = config
        self.current_loss = loss
        best_loss = _scalar_min(loss)
        if best_loss is None:
            return
        if self.best_loss is None or best_loss < self.best_loss:
            self.best_loss = best_loss
            self.best_config = config or self.current_config

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "algorithm_name": self.algorithm_name,
            "problem_name": self.problem_name,
            "seed": self.seed,
            "unbounded": self.unbounded,
            "algorithm_metadata": self.algorithm_metadata,
            "problem_metadata": self.problem_metadata,
            "search_space": self.search_space,
            "current_config": (
                self.current_config.to_dict() if self.current_config else None
            ),
            "current_loss": _jsonable(self.current_loss),
            "best_config": self.best_config.to_dict() if self.best_config else None,
            "best_loss": self.best_loss,
            "ask_count": self.ask_count,
            "tell_count": self.tell_count,
            "optimizer_state": _jsonable(self.optimizer_state),
        }


def _scalar_min(value: Any) -> float | None:
    if value is None:
        return None
    try:
        import numpy as np

        arr = np.asarray(value, dtype=float)
        if arr.size == 0 or np.all(np.isnan(arr)):
            return None
        return float(np.nanmin(arr))
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if hasattr(value, "item"):
        try:
            item = value.item()
            return item if item is value else _jsonable(item)
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return repr(value)
