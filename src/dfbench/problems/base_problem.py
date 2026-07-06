"""Base class for gravitational wave detector optimization problems."""

from abc import abstractmethod
from typing import Any, Callable, Mapping

import jax.numpy as jnp
from jaxtyping import Array, Float

from differometor.components import (
    DETECTOR_POWER_THRESHOLD,
    HARD_SIDE_POWER_THRESHOLD,
    SOFT_SIDE_POWER_THRESHOLD,
)

from dfbench.core.problem import ContinuousProblem, register_problem


DEFAULT_SIGNAL_FLOOR = 1e-20


# ---------------------------------------------------------------------------
# Power-penalty presets
# ---------------------------------------------------------------------------


def squashed_relu_penalty(value, threshold):
    """Default penalty: per-element ReLU squashed into [0, 1).

    ``max(value/threshold - 1, 0)`` passed through ``p / (1 + p)``.
    """
    relu = jnp.maximum(value / threshold - 1, 0)
    return relu / (1.0 + relu)


def relu_penalty(value, threshold):
    """Raw ReLU penalty: ``max(value/threshold - 1, 0)``."""
    return jnp.maximum(value / threshold - 1, 0)


def zero_penalty(value, threshold):
    """No penalty: Ignores power constraints."""
    return jnp.zeros_like(value)


# Registry of named penalty functions so they can be (de)serialised by name.
_PENALTY_FUNCTIONS: dict[str, Callable] = {
    "squashed_relu_penalty": squashed_relu_penalty,
    "relu_penalty": relu_penalty,
    "zero_penalty": zero_penalty,
}


def penalty_fn_to_name(fn: Callable | None) -> str | None:
    """Return the registry name of a penalty function, or None for default."""
    if fn is None:
        return None
    name = getattr(fn, "__name__", None)
    if name is None or name not in _PENALTY_FUNCTIONS:
        raise ValueError(
            f"Cannot serialise power_penalty_fn {fn!r}: not a registered "
            f"preset. Known: {sorted(_PENALTY_FUNCTIONS.keys())}."
        )
    return name


def name_to_penalty_fn(name: str | None) -> Callable | None:
    """Resolve a penalty-function name back to the callable, or None."""
    if name is None:
        return None
    if name not in _PENALTY_FUNCTIONS:
        raise ValueError(
            f"Unknown power_penalty_fn name '{name}'. "
            f"Known: {sorted(_PENALTY_FUNCTIONS.keys())}."
        )
    return _PENALTY_FUNCTIONS[name]


def sensitivity_single_noise(noises, powers, frequencies):
    """Single-noise sensitivity model used by lightweight Voyager problems."""
    del frequencies
    return noises[0] / jnp.abs(powers[0])


@register_problem
class OpticalSetupProblem(ContinuousProblem):
    """Abstract base class for optical setup optimization problems.

    This class extends ContinuousProblem with optical-specific functionality
    for gravitational wave detector optimization, including frequency-dependent
    sensitivity calculations.

    Inherits from ContinuousProblem to ensure compatibility with all optimization
    algorithms in the dfbench framework.
    """

    #: Opt-in flag: subclasses that compute a power-constraint penalty set this
    #: to ``True`` so :meth:`set_penalty_fn` and the Objective aux path know the
    #: problem actually has a penalty contract. ``VoyagerProblem`` and
    #: ``VoyagerTuningProblem`` leave it ``False`` because they have no power
    #: constraints, even though they inherit the method.
    _supports_power_penalty: bool = False

    def __init__(
        self,
        name: str,
        n_frequencies: int = 100,
        signal_floor: float = DEFAULT_SIGNAL_FLOOR,
    ):
        """Initialize the optimization problem.

        Args:
            name (str): Name of the problem, used for output file naming.
            n_frequencies (int): Number of frequency points for sensitivity calculation.
                Defaults to 100.
            signal_floor: Optional lower bound for detector signal
                magnitudes before sensitivity normalization.
        """
        self._name = name.lstrip("_")
        self._frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), n_frequencies)
        self._target_sensitivities = None  # to be set by subclasses
        self._power_penalty_fn: Callable = squashed_relu_penalty
        self._signal_floor = float(max(signal_floor, 0.0))

    @property
    def power_penalty_fn(self) -> Callable:
        """The function used to compute per-element power-constraint penalties.

        Signature: ``fn(value, threshold) -> penalty_contribution``
        """
        return self._power_penalty_fn

    @property
    def signal_floor(self) -> float:
        """Lower floor applied to detector signal magnitudes."""
        return self._signal_floor

    @property
    def power_thresholds(self) -> dict[str, float] | None:
        """Per-group power thresholds, or ``None`` when the problem has no penalty path.

        Returns a dict with keys ``"hard"``, ``"soft"``, ``"detector"`` mapping
        to the constant threshold values used by :meth:`_compute_power_violations`.
        Only meaningful on subclasses where ``_supports_power_penalty`` is ``True``.
        """
        if not self._supports_power_penalty:
            return None
        return {
            "hard": float(HARD_SIDE_POWER_THRESHOLD),
            "soft": float(SOFT_SIDE_POWER_THRESHOLD),
            "detector": float(DETECTOR_POWER_THRESHOLD),
        }

    def _compute_power_violations(self, powers):
        """Apply ``power_penalty_fn`` to each component group and concatenate."""
        fn = self._power_penalty_fn
        hard = fn(powers[0].squeeze(1), HARD_SIDE_POWER_THRESHOLD)
        soft = fn(powers[1].squeeze(1), SOFT_SIDE_POWER_THRESHOLD)
        det = fn(powers[2].squeeze(1), DETECTOR_POWER_THRESHOLD)
        return jnp.concatenate([hard, det, soft], axis=0)

    def _build_aux(self, powers, sensitivity_loss, penalty, violations):
        """Build the per-eval aux diagnostics dict from a power-constraint eval.

        Args:
            powers: ``(hard, soft, detector)`` tuple of per-group power arrays
                as returned by ``calculate_powers``.
            sensitivity_loss: scalar sensitivity loss (the unpenalised objective).
            penalty: scalar summed penalty contribution.
            violations: ``(n_constraints, n_freq)`` array of per-constraint
                penalty values, as returned by :meth:`_calculate_loss`.

        Returns:
            A dict pytree with leaves that vmapp cleanly:

            - ``sensitivity_loss``: scalar float.
            - ``penalty``: scalar float.
            - ``is_feasible``: scalar bool, ``True`` iff every per-group power
              is at or below its threshold. This is a physical check, independent
              of the active ``power_penalty_fn`` preset.
            - ``violations``: ``(n_constraints, n_freq)`` array.
            - ``power_values``: dict with ``"hard"``, ``"soft"``, ``"detector"``
              leaves holding the raw per-group power arrays.
        """
        hard, soft, det = powers
        is_feasible = jnp.logical_and(
            jnp.all(hard.squeeze(1) <= HARD_SIDE_POWER_THRESHOLD),
            jnp.logical_and(
                jnp.all(soft.squeeze(1) <= SOFT_SIDE_POWER_THRESHOLD),
                jnp.all(det.squeeze(1) <= DETECTOR_POWER_THRESHOLD),
            ),
        )
        return {
            "sensitivity_loss": sensitivity_loss,
            "penalty": penalty,
            "is_feasible": is_feasible,
            "violations": violations,
            "power_values": {
                "hard": hard,
                "soft": soft,
                "detector": det,
            },
        }

    def set_penalty_fn(self, fn: Callable) -> None:
        """Set the penalty function and rebuild the objective function.

        Updates ``_power_penalty_fn`` and re-traces the JIT-compiled
        ``objective_function`` so the new penalty takes effect.

        Must be called before the problem is wrapped in a logging
        ``Objective`` (or before ``Objective.start_logging()``).

        Raises:
            RuntimeError: If this problem does not opt into the power-penalty
                contract (``_supports_power_penalty`` is ``False``). Subclasses
                with a power-constraint path set that flag to ``True``.
        """
        if not self._supports_power_penalty:
            raise RuntimeError(
                f"{type(self).__name__} has no power-constraint path "
                f"(_supports_power_penalty is False); set_penalty_fn is only "
                f"supported on problems that compute power violations."
            )
        self._power_penalty_fn = fn
        self._build_objective_function()

    @abstractmethod
    def _build_objective_function(self) -> None:
        """(Re)build and assign the JIT-compiled ``objective_function``.

        Subclasses define their closed-over objective here. Called both
        from ``__init__`` and from ``set_penalty_fn`` so the compiled
        function picks up the current ``_power_penalty_fn``.
        """

    def _calculate_loss(self, sensitivities, reference_sensitivities, powers):
        """Calculate loss and penalties from sensitivities and power constraints."""
        violations = self._compute_power_violations(powers)
        losses = jnp.mean(jnp.log10(sensitivities.T / reference_sensitivities), axis=-1)
        penalties = jnp.sum(violations.T, axis=-1)
        return losses, penalties, violations

    def _apply_property_bounds_overrides(
        self,
        property_bounds: dict[str, list[float]],
        bounds_overrides: Mapping[str, tuple[float, float]] | None = None,
        allow_widen: bool = False,
    ) -> dict[str, list[float]]:
        """Return property bounds with optional user overrides applied.

        Args:
            property_bounds: Default bounds by property name.
            bounds_overrides: Optional bounds overrides by property name.
            allow_widen: If False (default), overrides may only narrow
                existing bounds.
        """
        merged_bounds = {
            property_name: [float(bounds[0]), float(bounds[1])]
            for property_name, bounds in property_bounds.items()
        }

        if not bounds_overrides:
            return merged_bounds

        for property_name, override in bounds_overrides.items():
            if property_name not in merged_bounds:
                valid_properties = ", ".join(sorted(merged_bounds.keys()))
                raise ValueError(
                    f"Unknown property in bounds_overrides: '{property_name}'. "
                    f"Valid properties: {valid_properties}"
                )

            if len(override) != 2:
                raise ValueError(
                    f"Override for '{property_name}' must be a tuple/list of "
                    "(lower, upper)."
                )

            lower = float(override[0])
            upper = float(override[1])
            if lower >= upper:
                raise ValueError(
                    f"Invalid override for '{property_name}': lower ({lower}) "
                    f"must be < upper ({upper})."
                )

            default_lower, default_upper = merged_bounds[property_name]
            if not allow_widen and (lower < default_lower or upper > default_upper):
                raise ValueError(
                    f"Override for '{property_name}' must narrow within "
                    f"[{default_lower}, {default_upper}], got [{lower}, {upper}]."
                )

            merged_bounds[property_name] = [lower, upper]

        return merged_bounds

    @staticmethod
    def _property_name_from_optimization_pair(optimization_pair) -> str:
        """Extract property name from standard or coupled optimization-pair formats."""
        if isinstance(optimization_pair[0], list):
            return optimization_pair[0][1]
        return optimization_pair[1]

    @staticmethod
    def _optimization_pair_label(optimization_pair) -> str:
        """Create a readable label for an optimization pair for display."""
        if isinstance(optimization_pair[0], list):
            first_component = optimization_pair[0][0]
            property_name = optimization_pair[0][1]
            n_coupled = len(optimization_pair)
            return f"{first_component} (coupled x{n_coupled}) :: {property_name}"

        component_name, property_name = optimization_pair
        return f"{component_name} :: {property_name}"

    def print_bounds(self) -> None:
        """Print all active parameter bounds in a human-readable list."""
        print(f"\nBounds for problem '{self.name}':")
        for index, optimization_pair in enumerate(self.optimization_pairs):
            label = self._optimization_pair_label(optimization_pair)
            lower = float(self.bounds[0, index])
            upper = float(self.bounds[1, index])
            print(f"  [{index:03d}] {label:<45} [{lower:.6g}, {upper:.6g}]")

    @property
    def name(self) -> str:
        """Name of the problem."""
        return self._name

    @property
    def frequencies(self) -> Float[Array, "n_frequencies"]:
        """Frequencies at which the sensitivity is calculated."""
        return self._frequencies

    @property
    def n_params(self) -> int:
        """Number of parameters to be optimized."""
        return len(self.optimization_pairs)

    @property
    @abstractmethod
    def bounds(self) -> Float[Array, "2 {self.n_params}"]:
        """Bounds for each parameter to be optimized. Shape: (2, n_params)."""
        pass

    @property
    @abstractmethod
    def optimization_pairs(self) -> list[tuple]:
        """List of (component, property) pairs to be optimized."""
        pass

    # objective_function is set as an instance attribute by subclasses (as a
    # JIT-compiled callable), following the ContinuousProblem protocol.

    @abstractmethod
    def calculate_sensitivity(
        self,
        optimized_parameters: Float[Array, "{self.n_params}"],
    ) -> Float[Array, "n_frequencies"]:
        """Calculate the sensitivity curve for given parameters.

        This is an optical-specific method not in the base ContinuousProblem.

        Args:
            optimized_parameters: Parameters to evaluate.

        Returns:
            Sensitivity values at each frequency point.
        """
        pass

    # --------- reconstructive spec ---------

    def _base_spec(self) -> dict[str, Any]:
        """Common optical-problem spec fields (subclass adds ``type`` + extras).

        Encodes the penalty function by name so the dict is JSON-safe.
        Subclasses append their own constructor args (n_frequencies,
        bounds_overrides, topology, ...) and the ``"type"`` registry key.
        """
        spec: dict[str, Any] = {
            "n_frequencies": int(self._frequencies.shape[0]),
            "power_penalty_fn": penalty_fn_to_name(self._power_penalty_fn),
        }
        return spec

    @staticmethod
    def _spec_to_penalty_fn(spec: dict[str, Any]) -> Callable | None:
        """Resolve a ``power_penalty_fn`` name from a spec to the callable."""
        return name_to_penalty_fn(spec.get("power_penalty_fn"))
