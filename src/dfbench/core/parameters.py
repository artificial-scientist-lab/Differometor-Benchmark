"""Standalone parameter primitives for benchmark search-space definitions.

These classes describe the *type* of a search variable without knowing
anything about Differometor, dfbench objectives, or optimization algorithms.
They are intentionally small, serializable, and close to the vocabulary used
by benchmark suites such as HPOBench, Syne Tune, and Bayesmark.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray


ParameterKind: TypeAlias = Literal["float", "integer", "discrete"]
ParameterValue: TypeAlias = float | int | str | bool


@dataclass
class Parameter(ABC):
    """Base class for one search-space variable."""

    name: str
    """Stable parameter name.

    For Differometor dimensions this is often derived from
    ``"{component}.{property}"``.
    """

    default: ParameterValue | None = None
    """Optional default value in the parameter's native domain."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Optional lightweight annotations.
    Keep values JSON-friendly if the parameter will be serialized.
    """

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Parameter name must be a non-empty string.")
        if self.default is not None:
            self.validate(self.default)

    @property
    @abstractmethod
    def kind(self) -> ParameterKind:
        """Kind tag used by serializers and external benchmark adapters."""

    @abstractmethod
    def validate(self, value: Any) -> None:
        """Raise ``ValueError`` if ``value`` is outside the parameter domain."""

    @abstractmethod
    def sample(
        self,
        rng: np.random.Generator,
        size: int | None = None,
    ) -> Any:
        """Draw one or more samples in the parameter's native domain."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""


@dataclass
class FloatParameter(Parameter):
    """Continuous scalar parameter."""

    lower: float = 0.0
    """The lower bound of the parameter."""

    upper: float = 1.0
    """The upper bound of the parameter."""

    log: bool = False
    """Whether the parameter is log-scaled."""

    log_base: float | None = None
    """The base of the logarithm for log-scaled parameters.
    If None, defaults to e. Ignored if log is False."""

    def __post_init__(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError("FloatParameter bounds must be finite.")
        if self.lower >= self.upper:
            raise ValueError(
                f"FloatParameter lower must be < upper, got "
                f"{self.lower} >= {self.upper}."
            )
        if self.log and self.lower <= 0:
            raise ValueError("Log-scaled FloatParameter requires lower > 0.")
        if self.log_base is not None and self.log_base <= 0:
            raise ValueError(
                "Log-scaled FloatParameter requires log_base > 0."
                f" Got log_base={self.log_base}."
            )
        if self.log_base is not None and self.log_base == 1:
            raise ValueError(
                "Log-scaled FloatParameter's log_base cannot be 1."
                f" Got log_base={self.log_base}."
            )
        super().__post_init__()

    @property
    def kind(self) -> Literal["float"]:
        return "float"

    def validate(self, value: Any) -> None:
        if not isinstance(value, int | float | np.floating | np.integer):
            raise ValueError(f"{self.name} must be a float-like value.")
        numeric = float(value)
        if not np.isfinite(numeric) or not self.lower <= numeric <= self.upper:
            raise ValueError(
                f"{self.name}={value!r} is outside [{self.lower}, {self.upper}]."
            )

    def sample(
        self,
        rng: np.random.Generator,
        size: int | None = None,
    ) -> float | NDArray[np.float64]:
        if self.log:
            low = np.log(self.lower)
            high = np.log(self.upper)
            values = np.exp(rng.uniform(low, high, size=size))
        else:
            values = rng.uniform(self.lower, self.upper, size=size)
        return float(values) if size is None else values

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "lower": self.lower,
            "upper": self.upper,
            "log": self.log,
            "log_base": self.log_base,
            "default": self.default,
            "metadata": self.metadata,
        }


@dataclass
class IntegerParameter(Parameter):
    """Discrete ordered integer parameter."""

    lower: int = 0
    """The lower bound of the parameter."""

    upper: int = 1
    """The upper bound of the parameter."""

    log: bool = False
    """Whether the parameter is log-scaled."""

    log_base: float | None = None
    """The base of the logarithm for log-scaled parameters.
    If None, defaults to e. Ignored if log is False."""

    def __post_init__(self) -> None:
        if not isinstance(self.lower, int) or not isinstance(self.upper, int):
            raise ValueError("IntegerParameter bounds must be integers.")
        if self.lower > self.upper:
            raise ValueError(
                f"IntegerParameter lower must be <= upper, got "
                f"{self.lower} > {self.upper}."
            )
        if self.log and self.lower <= 0:
            raise ValueError("Log-scaled IntegerParameter requires lower > 0.")
        if self.log_base is not None and self.log_base <= 0:
            raise ValueError(
                "Log-scaled IntegerParameter requires log_base > 0."
                f" Got log_base={self.log_base}."
            )
        if self.log_base is not None and self.log_base == 1:
            raise ValueError(
                "Log-scaled IntegerParameter's log_base cannot be 1."
                f" Got log_base={self.log_base}."
            )

        # NOTE (Soham): Notations such as 1E5 or 1e5 are valid floats but not
        # valid integers. So, we need to check if the lower and upper bounds are
        # actually integers and not floats that can be converted to integers.
        lower_int = int(self.lower)
        upper_int = int(self.upper)
        if self.lower != lower_int or self.upper != upper_int:
            raise ValueError(
                f"IntegerParameter bounds must be integers, got "
                f"lower={self.lower}, upper={self.upper}."
            )
        super().__post_init__()

    @property
    def kind(self) -> Literal["integer"]:
        return "integer"

    def validate(self, value: Any) -> None:
        if not isinstance(value, int | np.integer) or isinstance(value, bool):
            raise ValueError(f"{self.name} must be an integer.")
        numeric = int(value)
        if not self.lower <= numeric <= self.upper:
            raise ValueError(
                f"{self.name}={value!r} is outside [{self.lower}, {self.upper}]."
            )

    def sample(
        self,
        rng: np.random.Generator,
        size: int | None = None,
    ) -> int | NDArray[np.int64]:
        """Draw one or more samples in the parameter's native domain."""
        if self.log:
            low = np.log(self.lower)
            high = np.log(self.upper + 1)
            values = np.floor(np.exp(rng.uniform(low, high, size=size))).astype(int)
            values = np.clip(values, self.lower, self.upper)
        else:
            values = rng.integers(self.lower, self.upper + 1, size=size)
        return int(values) if size is None else values

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "lower": self.lower,
            "upper": self.upper,
            "log": self.log,
            "log_base": self.log_base,
            "default": self.default,
            "metadata": self.metadata,
        }


@dataclass
class DiscreteParameter(Parameter):
    """Unordered discrete parameter."""

    choices: tuple[ParameterValue, ...] = ()

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("DiscreteParameter choices must be non-empty.")
        if len(set(self.choices)) != len(self.choices):
            raise ValueError("DiscreteParameter choices must be unique.")
        super().__post_init__()

    @property
    def kind(self) -> Literal["discrete"]:
        return "discrete"

    def validate(self, value: Any) -> None:
        if value not in self.choices:
            raise ValueError(
                f"{self.name}={value!r} is not one of {list(self.choices)!r}."
            )

    def sample(
        self,
        rng: np.random.Generator,
        size: int | None = None,
    ) -> ParameterValue | NDArray[Any]:
        values = rng.choice(np.asarray(self.choices, dtype=object), size=size)
        if size is None:
            return values.item() if hasattr(values, "item") else values
        return values

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "choices": list(self.choices),
            "default": self.default,
            "metadata": self.metadata,
        }
