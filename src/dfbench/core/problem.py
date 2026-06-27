"""Abstract base class for continuous optimization problems.

Defines the interface that all optimization problems must implement.
Problems provide objective functions and bounds for the parameter space.

A :class:`ContinuousProblem` also carries a reconstructive contract via
:meth:`to_spec` / :meth:`from_spec`: a small, serializable dict that
captures everything needed to rebuild an *equivalent* problem instance
in a separate process. This is recorded in checkpoint metadata so a
saved run is self-describing — the problem identity is recoverable from
the file alone, not just from the caller's memory.

Attributes:
    name (str): Human-readable name for the problem.
    objective_function (Callable): Objective to minimize. Expects parameters
        in the BOUNDED space (within `bounds`). The Objective wrapper owns
        any mapping needed for algorithms that optimize in unbounded space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, ClassVar

from jaxtyping import Array, Float

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


def build_problem_from_spec(spec: dict[str, Any]) -> "ContinuousProblem":
    """Reconstruct a problem instance from a ``to_spec()`` dict.

    The dict must contain a ``"type"`` key matching a registered problem
    class. Remaining keys are passed as keyword arguments to that class's
    constructor.
    """
    if "type" not in spec:
        raise ValueError("Problem spec missing required 'type' key.")
    type_name = spec["type"]
    if type_name not in _PROBLEM_REGISTRY:
        raise ValueError(
            f"Problem type '{type_name}' is not registered. "
            f"Known types: {sorted(_PROBLEM_REGISTRY.keys())}. "
            "Make sure the problem module is imported."
        )
    cls = _PROBLEM_REGISTRY[type_name]
    kwargs = {k: v for k, v in spec.items() if k != "type"}
    return cls(**kwargs)


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
        """
        pass

    @classmethod
    def from_spec(cls, spec: dict[str, Any]) -> "ContinuousProblem":
        """Reconstruct a problem from a :meth:`to_spec` dict.

        Default implementation uses the module-level registry; subclasses
        may override for custom reconstruction logic.
        """
        return build_problem_from_spec(spec)
