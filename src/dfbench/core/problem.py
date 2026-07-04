"""Abstract base class for continuous optimization problems.

Defines the interface that all optimization problems must implement.
Problems provide objective functions and bounds for the parameter space.

A :class:`ContinuousProblem` also carries a reconstructive contract via
:meth:`to_spec` / :meth:`from_spec`: a small, serializable dict that
captures everything needed to rebuild an *equivalent* problem instance
in a separate process. This is recorded in checkpoint metadata so a
saved run is self-describing — the problem identity is recoverable from
the file alone, not just from the caller's memory.

The typed container :class:`ProblemSpec` wraps the raw ``to_spec()`` dict
with an explicit ``version`` and ``params`` field, so checkpoint
consumers (the competition scoring harness, the leaderboard, cross-round
provenance) get a stable, schema-validated identity instead of an
untyped dict. Legacy flat specs (``{"type": ..., <kwargs>}``) are
accepted on load via :meth:`ProblemSpec.from_dict` for backward
compatibility with checkpoints written before the typed container existed.

Attributes:
    name (str): Human-readable name for the problem.
    objective_function (Callable): Objective to minimize. Expects parameters
        in the BOUNDED space (within `bounds`). The Objective wrapper owns
        any mapping needed for algorithms that optimize in unbounded space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from jaxtyping import Array, Float

# ------------------------------------------------------------------
# Problem spec container
# ------------------------------------------------------------------

PROBLEM_SPEC_VERSION: int = 1
"""On-disk schema version for the :class:`ProblemSpec` container.

Increment when the container structure (the ``type``/``version``/``params``
layout) changes in a backwards-incompatible way. The per-problem
constructor arguments live inside ``params`` and are versioned implicitly
by the problem class's own ``to_spec`` implementation; this version only
governs the container shape.
"""


@dataclass
class ProblemSpec:
    """Typed, serializable identity for a :class:`ContinuousProblem`.

    The container carries the registry ``type`` (the class name or
    ``spec_type``), a schema ``version`` for the container itself, and
    ``params`` — the constructor arguments needed to rebuild an
    equivalent problem instance. The dict produced by
    :meth:`to_dict` is JSON-safe and is what gets embedded in
    :class:`~dfbench.core.storage.RunMetadata.extra["problem_spec"].

    Attributes:
        type: Registry key matching a ``@register_problem``-decorated
            class. Must be a non-empty string.
        params: Constructor keyword arguments forwarded to the problem
            class on reconstruction. Must be a JSON-serializable dict.
        version: Container schema version. Defaults to
            :data:`PROBLEM_SPEC_VERSION`.
    """

    type: str
    params: dict[str, Any] = field(default_factory=dict)
    version: int = PROBLEM_SPEC_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.type, str) or not self.type:
            raise ValueError(
                f"ProblemSpec.type must be a non-empty string, got {self.type!r}."
            )
        if not isinstance(self.params, dict):
            raise TypeError(
                f"ProblemSpec.params must be a dict, got {type(self.params).__name__}."
            )
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise TypeError(
                f"ProblemSpec.version must be an int, got {type(self.version).__name__}."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for embedding in checkpoint metadata."""
        return {
            "type": self.type,
            "version": self.version,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ProblemSpec":
        """Reconstruct a :class:`ProblemSpec` from a dict.

        Accepts both the typed container form
        (``{"type": ..., "version": ..., "params": {...}}``) and the
        legacy flat form produced by older checkpoints
        (``{"type": ..., <constructor kwargs>...}``). In the legacy form
        every key other than ``"type"`` is treated as a constructor
        argument and collected into ``params``.

        Raises:
            ValueError: If the dict has no ``"type"`` key or the type is
                empty.
        """
        if not isinstance(d, dict):
            raise TypeError(
                f"ProblemSpec.from_dict expected a dict, got {type(d).__name__}."
            )
        if "type" not in d:
            raise ValueError("Problem spec missing required 'type' key.")
        type_name = d["type"]
        if not isinstance(type_name, str) or not type_name:
            raise ValueError(
                f"Problem spec 'type' must be a non-empty string, got {type_name!r}."
            )

        # Typed container form: explicit "params" sub-dict.
        if "params" in d and isinstance(d["params"], dict):
            return cls(
                type=type_name,
                params=dict(d["params"]),
                version=int(d.get("version", PROBLEM_SPEC_VERSION)),
            )

        # Legacy flat form: every key other than "type" (and "version" if
        # present) is a constructor argument.
        version = d.get("version", PROBLEM_SPEC_VERSION)
        params = {k: v for k, v in d.items() if k not in ("type", "version")}
        return cls(type=type_name, params=params, version=int(version))

    @classmethod
    def from_problem(cls, problem: "ContinuousProblem") -> "ProblemSpec":
        """Build a :class:`ProblemSpec` from a live problem instance.

        Uses the problem's :meth:`ContinuousProblem.to_spec` dict and
        normalizes it into the typed container via :meth:`from_dict`.
        """
        return cls.from_dict(problem.to_spec())


# ------------------------------------------------------------------
# Problem registry
# ------------------------------------------------------------------

_PROBLEM_REGISTRY: dict[str, type["ContinuousProblem"]] = {}


def register_problem(cls: type["ContinuousProblem"]) -> type["ContinuousProblem"]:
    """Register a problem class for reconstruction by ``type`` name.

    Used as a class decorator. The registered name is stored under the
    class's ``__name__`` unless the class defines a ``spec_type`` class
    attribute.
    """
    key = getattr(cls, "spec_type", None) or cls.__name__
    if key in _PROBLEM_REGISTRY and _PROBLEM_REGISTRY[key] is not cls:
        raise ValueError(
            f"Problem type '{key}' is already registered to "
            f"{_PROBLEM_REGISTRY[key].__name__}."
        )
    _PROBLEM_REGISTRY[key] = cls
    return cls


def build_problem_from_spec(spec: ProblemSpec | dict[str, Any]) -> "ContinuousProblem":
    """Reconstruct a problem instance from a :class:`ProblemSpec` or dict.

    Accepts either a typed :class:`ProblemSpec` container or a raw dict
    (legacy flat form or typed container form — both are normalized via
    :meth:`ProblemSpec.from_dict`). The ``type`` key must match a
    registered problem class; remaining keys (or the ``params`` sub-dict)
    are passed as keyword arguments to that class's constructor.
    """
    if isinstance(spec, ProblemSpec):
        ps = spec
    elif isinstance(spec, dict):
        ps = ProblemSpec.from_dict(spec)
    else:
        raise TypeError(
            f"build_problem_from_spec expected ProblemSpec or dict, "
            f"got {type(spec).__name__}."
        )

    type_name = ps.type
    if type_name not in _PROBLEM_REGISTRY:
        raise ValueError(
            f"Problem type '{type_name}' is not registered. "
            f"Known types: {sorted(_PROBLEM_REGISTRY.keys())}. "
            "Make sure the problem module is imported."
        )
    cls = _PROBLEM_REGISTRY[type_name]
    return cls(**ps.params)


def validate_spec_round_trip(
    problem: "ContinuousProblem",
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> "ContinuousProblem":
    """Rebuild ``problem`` from its own spec and assert equivalence.

    Reconstructs the problem via :func:`build_problem_from_spec` and checks
    that the rebuilt instance has the same ``n_params`` and matching
    ``bounds``. Returns the rebuilt problem on success.

    Raises:
        AssertionError: If the rebuilt problem's ``n_params`` or ``bounds``
            differ from the original.
    """
    spec = problem.to_spec()
    rebuilt = build_problem_from_spec(spec)

    assert rebuilt.n_params == problem.n_params, (
        f"Spec round-trip mismatch: original n_params={problem.n_params} "
        f"but rebuilt n_params={rebuilt.n_params}."
    )

    import numpy as np

    np.testing.assert_allclose(
        np.asarray(rebuilt.bounds),
        np.asarray(problem.bounds),
        rtol=rtol,
        atol=atol,
        err_msg="Spec round-trip mismatch: rebuilt bounds differ from original.",
    )
    return rebuilt


class ContinuousProblem(ABC):
    """Abstract base class for continuous optimization problems.

    See module docstring for the reconstructive contract.
    """

    name: str = "unkown_problem"

    objective_function: Callable[[Float[Array, "{self.n_params}"]], Float]

    # Optional override for the registry key; defaults to the class name.
    spec_type: ClassVar[str | None] = None

    def __init__(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def bounds(
        self,
    ) -> Float[Array, "2 {self.n_params}"]:
        """Parameter bounds as [lower_bounds, upper_bounds].

        Returns:
            Array of shape (2, n_params) where bounds[0] are lower bounds
            and bounds[1] are upper bounds for each parameter.
        """
        pass

    @property
    def n_params(self) -> int:
        """Number of parameters to optimize."""
        return len(self.optimization_pairs)

    @abstractmethod
    def to_spec(self) -> dict[str, Any]:
        """Return a serializable dict sufficient to rebuild this problem.

        The dict must include a ``"type"`` key (the registry name) plus
        every constructor argument needed for :meth:`from_spec` to produce
        an equivalent instance. Callables should be encoded by name (e.g.
        a registered penalty function's ``__name__``).

        Implementations must be pure and cheap — no JAX arrays, no live
        objects — so the dict can be JSON-serialised and stored in
        checkpoint metadata.

        Prefer :meth:`to_problem_spec` for new code; this method is kept
        for backward compatibility and is the source of truth the typed
        container derives from.
        """
        pass

    def to_problem_spec(self) -> ProblemSpec:
        """Return a typed :class:`ProblemSpec` container for this problem.

        Default implementation wraps :meth:`to_spec` via
        :meth:`ProblemSpec.from_problem`. Subclasses may override to
        produce the container directly, but the default is sufficient as
        long as :meth:`to_spec` is implemented correctly.
        """
        return ProblemSpec.from_problem(self)

    @classmethod
    def from_spec(cls, spec: ProblemSpec | dict[str, Any]) -> "ContinuousProblem":
        """Reconstruct a problem from a :meth:`to_spec` dict or :class:`ProblemSpec`.

        Default implementation uses the module-level registry; subclasses
        may override for custom reconstruction logic.
        """
        return build_problem_from_spec(spec)
