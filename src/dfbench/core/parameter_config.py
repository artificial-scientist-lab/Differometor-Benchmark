"""Rich parameter-configuration containers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from dfbench.core.search_space import SearchSpace


@dataclass(frozen=True)
class ParameterConfig:
    """A numeric parameter vector together with its search-space meaning."""

    values: Mapping[str, Any]
    """The parameter values as dict of name -> value.    """

    search_space: SearchSpace
    """The Search Space object of the ParameterConfig."""

    unbounded: bool = False

    @property
    def vector(self) -> np.ndarray:
        """Return the parameter values as a 1D numpy array."""
        return np.array(list(self.values.values()), dtype=np.float64)

    @classmethod
    def from_values(
        cls,
        values: np.ndarray | Mapping[str, Any] | Sequence[Any],
        search_space: SearchSpace | dict[str, Any] | None = None,
        *,
        unbounded: bool = False,
    ) -> "ParameterConfig":
        """Create a config from array-like values and optional space metadata."""
        match values:
            case np.ndarray():
                if values.ndim == 1:
                    values = values.flatten()
                elif values.ndim > 2:
                    raise ValueError(
                        f"Values must be 1D or 2D, got {values.ndim}D array."
                    )
            case Mapping():
                values = dict(values)
            case Sequence():
                values = list(values)
            case _:
                raise TypeError(
                    f"Values must be array-like, mapping, or sequence, "
                    f"got {type(values)}."
                )

        match search_space:
            case SearchSpace():
                pass
            case dict():
                search_space = SearchSpace.from_dict(search_space)
            case _:
                raise TypeError(
                    f"Search space must be a SearchSpace or dict, "
                    f"got {type(search_space)}."
                )

        return cls(values=values, search_space=search_space, unbounded=unbounded)

    @property
    def is_batched(self) -> bool:
        return self.values.ndim == 2

    @property
    def names(self) -> tuple[str, ...]:
        if not self.search_space:
            return tuple(f"parameter_{i}" for i in range(self.n_params))
        dimensions = self.search_space.get("dimensions", [])
        return tuple(
            str(dimension.get("name", f"parameter_{i}"))
            for i, dimension in enumerate(dimensions)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": self.values,
            "names": self.names,
            "unbounded": self.unbounded,
            "search_space": self.search_space,
        }
