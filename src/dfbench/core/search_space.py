"""Standalone search-space containers for Differometor benchmark variables.

The classes here intentionally do not import or modify existing dfbench
problems, objectives, or algorithms.  They provide a benchmark-style schema
for naming parameters, attaching them to Differometor component properties,
serializing the space, and sampling native parameter values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

from dfbench.core.parameters import (
    DiscreteParameter,
    FloatParameter,
    IntegerParameter,
    Parameter,
    ParameterValue,
)


NativeSample: TypeAlias = dict[str, ParameterValue]
DifferometorTarget: TypeAlias = tuple[str, str]
DifferometorOptimizationPair: TypeAlias = (
    DifferometorTarget | tuple[DifferometorTarget, ...]
)
RawDifferometorOptimizationPair: TypeAlias = Sequence[str] | Sequence[Sequence[str]]


@dataclass(frozen=True, slots=True)
class TargetRef:
    """A destination in a Differometor setup."""

    component: str
    """Name of the Differometor component."""

    property: str
    """Name of the component property controlled by a search value."""

    def __post_init__(self) -> None:
        if not isinstance(self.component, str) or not self.component:
            raise ValueError("Target component must be a non-empty string.")
        if not isinstance(self.property, str) or not self.property:
            raise ValueError("Target property must be a non-empty string.")

    @property
    def label(self) -> str:
        return f"{self.component}.{self.property}"

    def as_pair(self) -> DifferometorTarget:
        return (self.component, self.property)

    def to_dict(self) -> dict[str, str]:
        return {"component": self.component, "property": self.property}

    @classmethod
    def from_pair(cls, pair: Sequence[str]) -> "TargetRef":
        """Create a target from Differometor's ``(component, property)`` form."""
        if len(pair) != 2:
            raise ValueError(
                "A Differometor target must have exactly two entries: "
                "(component, property)."
            )
        component, property_name = pair
        return cls(str(component), str(property_name))


@dataclass(frozen=True, slots=True)
class SearchDimension:
    """One search variable and the Differometor target(s) it controls.

    A dimension usually targets one ``(component, property)`` pair.  UIFO-style
    coupled variables can target multiple component properties with one scalar
    value, represented by multiple ``TargetRef`` entries.
    """

    name: str
    parameter: Parameter
    targets: tuple[TargetRef, ...]
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SearchDimension name must be a non-empty string.")
        if not isinstance(self.parameter, Parameter):
            raise TypeError("SearchDimension parameter must be a Parameter.")
        if not self.targets:
            raise ValueError("SearchDimension must have at least one target.")
        if self.parameter.name != self.name:
            raise ValueError(
                "SearchDimension name and parameter name should match for "
                f"stable lookup ({self.name!r} != {self.parameter.name!r})."
            )

    @property
    def is_coupled(self) -> bool:
        return len(self.targets) > 1

    @property
    def property_names(self) -> tuple[str, ...]:
        return tuple(target.property for target in self.targets)

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(target.component for target in self.targets)

    @property
    def label(self) -> str:
        if self.is_coupled:
            first = self.targets[0]
            return f"{first.label} (coupled x{len(self.targets)})"
        return self.targets[0].label

    def optimization_pair(self) -> DifferometorOptimizationPair:
        pairs = tuple(target.as_pair() for target in self.targets)
        return pairs[0] if len(pairs) == 1 else pairs

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "parameter": self.parameter.to_dict(),
            "targets": [target.to_dict() for target in self.targets],
            "tags": list(self.tags),
            "metadata": self.metadata,
        }

    @classmethod
    def from_differometor_pair(
        cls,
        pair: RawDifferometorOptimizationPair,
        lower: float,
        upper: float,
        index: int,
    ) -> "SearchDimension":
        """Create one dimension from an existing Differometor optimization pair."""
        targets = _targets_from_optimization_pair(pair)
        name = _dimension_name(targets, index)
        return cls(
            name=name,
            parameter=FloatParameter(name=name, lower=float(lower), upper=float(upper)),
            targets=targets,
            tags=("differometor", "continuous"),
        )


@dataclass(frozen=True, slots=True)
class SearchSpace:
    """Named collection of benchmark search dimensions.

    This is the object a benchmark can expose to users and optimizers.  It is
    deliberately independent of the current dfbench ``Objective`` flow, while
    still able to emit Differometor-compatible ``optimization_pairs`` and
    numeric bounds for continuous/integer spaces.
    """

    name: str
    dimensions: tuple[SearchDimension, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("SearchSpace name must be a non-empty string.")
        if not self.dimensions:
            raise ValueError("SearchSpace must contain at least one dimension.")
        names = self.names
        if len(set(names)) != len(names):
            raise ValueError(f"SearchSpace dimension names must be unique: {names}.")

    @classmethod
    def from_bounds(
        cls,
        bounds: Any,
        optimization_pairs: Iterable[RawDifferometorOptimizationPair] | None = None,
        name: str = "search_space",
        metadata: dict[str, Any] | None = None,
    ) -> "SearchSpace":
        """Materialize dfbench's implicit ``optimization_pairs + bounds`` schema.

        Existing dfbench problems expose a numeric ``bounds`` array with shape
        ``(2, n_params)`` and, for Differometor-backed problems, an
        ``optimization_pairs`` sequence.  This constructor turns that pair of
        values into explicit ``SearchDimension`` objects while preserving the
        old representation through ``bounds_array()`` and
        ``optimization_pairs()``.
        """
        bounds_array = np.asarray(bounds, dtype=np.float64)
        if bounds_array.ndim != 2 or bounds_array.shape[0] != 2:
            raise ValueError(
                f"bounds must have shape (2, n_params), got {bounds_array.shape}."
            )

        n_params = bounds_array.shape[1]
        pairs = (
            list(optimization_pairs)
            if optimization_pairs is not None
            else [(f"parameter_{index}", "value") for index in range(n_params)]
        )
        if len(pairs) != n_params:
            raise ValueError(
                "optimization_pairs and bounds must describe the same number "
                f"of dimensions ({len(pairs)} != {n_params})."
            )

        dimensions = tuple(
            SearchDimension.from_differometor_pair(
                pair=pair,
                lower=bounds_array[0, index],
                upper=bounds_array[1, index],
                index=index,
            )
            for index, pair in enumerate(pairs)
        )
        return cls(name=name, dimensions=dimensions, metadata=metadata or {})

    @classmethod
    def from_problem(cls, problem: Any) -> "SearchSpace":
        """Build a search space from a current dfbench ``ContinuousProblem``."""
        return cls.from_bounds(
            bounds=problem.bounds,
            optimization_pairs=getattr(problem, "optimization_pairs", None),
            name=getattr(problem, "name", problem.__class__.__name__),
            metadata={"source": problem.__class__.__name__},
        )

    @property
    def n_dims(self) -> int:
        return len(self.dimensions)

    @property
    def n_params(self) -> int:
        return self.n_dims

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(dimension.name for dimension in self.dimensions)

    @property
    def component_names(self) -> tuple[str, ...]:
        names = {
            component
            for dimension in self.dimensions
            for component in dimension.component_names
        }
        return tuple(sorted(names))

    @property
    def property_names(self) -> tuple[str, ...]:
        names = {
            property_name
            for dimension in self.dimensions
            for property_name in dimension.property_names
        }
        return tuple(sorted(names))

    def optimization_pairs(self) -> tuple[DifferometorOptimizationPair, ...]:
        """Return the Differometor-compatible target mapping."""
        return tuple(dimension.optimization_pair() for dimension in self.dimensions)

    def bounds_array(self) -> NDArray[np.float64]:
        """Return numeric bounds with shape ``(2, n_dims)``.

        Raises:
            TypeError: If any dimension is discrete.
        """
        lower: list[float] = []
        upper: list[float] = []
        for dimension in self.dimensions:
            parameter = dimension.parameter
            if isinstance(parameter, FloatParameter | IntegerParameter):
                lower.append(float(parameter.lower))
                upper.append(float(parameter.upper))
            elif isinstance(parameter, DiscreteParameter):
                raise TypeError(
                    "Discrete dimensions do not have numeric bounds: "
                    f"{dimension.name!r}."
                )
            else:
                raise TypeError(f"Unsupported parameter type: {type(parameter)!r}.")
        return np.asarray([lower, upper], dtype=np.float64)

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        """Lower numeric bounds in search-space order."""
        return self.bounds_array()[0]

    @property
    def upper_bounds(self) -> NDArray[np.float64]:
        """Upper numeric bounds in search-space order."""
        return self.bounds_array()[1]

    def get(self, name: str) -> SearchDimension:
        for dimension in self.dimensions:
            if dimension.name == name:
                return dimension
        raise KeyError(name)

    def validate(self, sample: NativeSample) -> None:
        missing = set(self.names) - set(sample)
        extra = set(sample) - set(self.names)
        if missing:
            raise ValueError(f"Sample is missing parameters: {sorted(missing)}.")
        if extra:
            raise ValueError(f"Sample has unknown parameters: {sorted(extra)}.")
        for dimension in self.dimensions:
            dimension.parameter.validate(sample[dimension.name])

    def sample(
        self,
        seed: int | np.random.Generator | None = None,
        n: int = 1,
    ) -> NativeSample | list[NativeSample]:
        """Sample one or more native dictionaries keyed by dimension name."""
        if n < 1:
            raise ValueError("n must be at least 1.")
        rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(seed)
        )
        samples = [
            {
                dimension.name: dimension.parameter.sample(rng)
                for dimension in self.dimensions
            }
            for _ in range(n)
        ]
        return samples[0] if n == 1 else samples

    def sample_array(
        self,
        seed: int | np.random.Generator | None = None,
        n: int = 1,
    ) -> NDArray[np.float64]:
        """Sample numeric dimensions as a dense array in search-space order.

        Raises:
            TypeError: If any dimension is discrete.
        """
        for dimension in self.dimensions:
            if isinstance(dimension.parameter, DiscreteParameter):
                raise TypeError(
                    "Cannot create a numeric sample array from discrete "
                    f"dimension {dimension.name!r}."
                )
        sample_or_samples = self.sample(seed=seed, n=n)
        samples = (
            sample_or_samples
            if isinstance(sample_or_samples, list)
            else [sample_or_samples]
        )
        rows = [
            [sample[dimension.name] for dimension in self.dimensions]
            for sample in samples
        ]
        array = np.asarray(rows, dtype=np.float64)
        return array[0] if n == 1 else array

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_dims": self.n_dims,
            "dimensions": [dimension.to_dict() for dimension in self.dimensions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(
        cls,
        space_dict: dict[str, Any],
    ) -> SearchSpace:
        """Create a SearchSpace from a dictionary representation."""
        name = space_dict.get("name", "search_space")
        dimensions_data = space_dict.get("dimensions", [])
        metadata = space_dict.get("metadata", {})

        dimensions = []
        for dim_data in dimensions_data:
            parameter_data = dim_data.get("parameter", {})
            parameter_type = parameter_data.get("type")
            if parameter_type == "float":
                parameter = FloatParameter(
                    name=parameter_data["name"],
                    lower=parameter_data["lower"],
                    upper=parameter_data["upper"],
                )
            elif parameter_type == "integer":
                parameter = IntegerParameter(
                    name=parameter_data["name"],
                    lower=parameter_data["lower"],
                    upper=parameter_data["upper"],
                )
            elif parameter_type == "discrete":
                parameter = DiscreteParameter(
                    name=parameter_data["name"],
                    values=parameter_data["values"],
                )
            else:
                raise ValueError(f"Unsupported parameter type: {parameter_type}")

            targets = tuple(
                TargetRef(
                    component=target_data["component"], property=target_data["property"]
                )
                for target_data in dim_data.get("targets", [])
            )

            dimension = SearchDimension(
                name=dim_data.get("name", parameter.name),
                parameter=parameter,
                targets=targets,
                tags=tuple(dim_data.get("tags", [])),
                metadata=dim_data.get("metadata", {}),
            )
            dimensions.append(dimension)

        return cls(name=name, dimensions=tuple(dimensions), metadata=metadata)


def _targets_from_optimization_pair(
    pair: RawDifferometorOptimizationPair,
) -> tuple[TargetRef, ...]:
    """Normalize Differometor's single/coupled target formats."""
    if (
        isinstance(pair, (tuple, list))
        and len(pair) == 2
        and isinstance(pair[0], str)
        and isinstance(pair[1], str)
    ):
        return (TargetRef.from_pair(pair),)

    try:
        targets = tuple(TargetRef.from_pair(target) for target in pair)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "optimization_pairs entries must be either "
            "(component, property) or a sequence of such pairs."
        ) from error

    if not targets:
        raise ValueError("Coupled optimization-pair groups cannot be empty.")
    return targets


def _dimension_name(targets: tuple[TargetRef, ...], index: int) -> str:
    """Create a stable, readable variable name."""
    first = targets[0]
    if len(targets) == 1:
        return first.label
    return f"{first.label}.coupled_{index}"
