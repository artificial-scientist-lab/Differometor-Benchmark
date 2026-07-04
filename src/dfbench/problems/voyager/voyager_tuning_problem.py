"""Voyager problem with tuning-only optimization."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from differometor.components import signal_detector
from differometor.setups import voyager
from differometor.simulate import run, run_build_step, simulate
from differometor.utils import update_setup

from ..base_problem import OpticalSetupProblem, register_problem


@register_problem
class VoyagerTuningProblem(OpticalSetupProblem):
    """Voyager optimization over mirror tuning parameters only.

    Uses a predefined Voyager setup with one noise detector and two signal
    detectors for balanced homodyne detection. Only ``tuning`` of selected
    Voyager components is optimized. The loss is relative to the target Voyager
    setup.
    """

    def __init__(
        self,
        n_frequencies: int = 100,
        bounds_overrides: dict[str, tuple[float, float]] | None = None,
    ):
        """Initialize the tuning-only Voyager optimization problem.

        Args:
            n_frequencies (int): Number of frequency points for sensitivity calculation.
                Defaults to 100.
            bounds_overrides: Optional property-level bound overrides.
                Example: {"tuning": (0, 45)}.
                Overrides must narrow default bounds.
        """
        super().__init__(name="voyager_tuning", n_frequencies=n_frequencies)

        # use a predefined Voyager setup with one noise detector and two signal detectors
        self._setup, component_property_pairs = voyager()
        self._bounds_overrides = bounds_overrides

        # run the simulation with the frequency as the changing parameter
        carrier, signal, noise, detector_ports, *_ = run(
            self._setup, [("f", "frequency")], self._frequencies
        )

        # calculate the signal power at the detector ports
        powers = signal_detector(carrier, signal)
        powers = powers[detector_ports]

        # calculate the signal power from the two signal detectors for balanced homodyne detection
        powers = powers[0] - powers[1]

        # calculate the sensitivity
        self._target_sensitivities = noise / jnp.abs(powers)

        ### Start from random parameters and optimize the sensitivity ###
        # ---------------------------------------------------------------#

        # specify the ranges for the properties to be optimized
        property_bounds = {
            "tuning": [-180, 180],
        }
        property_bounds = self._apply_property_bounds_overrides(
            property_bounds,
            bounds_overrides,
        )

        # select properties/components to be optimized
        optimized_properties = ["tuning"]
        optimized_components = {"prm", "itmy", "etmy", "itmx", "etmx", "srm"}
        self._optimization_pairs = []
        for pair in component_property_pairs:
            if pair[1] in optimized_properties and pair[0] in optimized_components:
                self._optimization_pairs.append(pair)

        # build the setup once and then reuse it during the optimization
        simulation_arrays, detector_ports, *_ = run_build_step(
            self._setup,
            [("f", "frequency")],
            self._frequencies,
            self._optimization_pairs,
        )

        # Store as instance attributes for use in objective functions
        self._simulation_arrays = simulation_arrays
        self._detector_ports = detector_ports

        # calculate the bounds for the properties to be optimized
        self._bounds = np.array(
            [
                [property_bounds[pair[1]][0], property_bounds[pair[1]][1]]
                for pair in self._optimization_pairs
            ]
        ).T

        target_sensitivities = self._target_sensitivities

        @jax.jit
        def objective_function(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> Float:
            carrier, signal, noise = simulate(
                **{
                    **self._simulation_arrays,
                    "optimized_parameters": optimized_parameters,
                }
            )
            powers = signal_detector(carrier, signal)
            powers = powers[self._detector_ports]
            powers = powers[0] - powers[1]
            sensitivities = noise / jnp.abs(powers)

            # relative objective as in voyager_tuning.py
            return jnp.mean(jnp.log10(sensitivities / target_sensitivities))

        self.objective_function = objective_function

    @property
    def optimization_pairs(self) -> list[tuple]:
        """List of (component, property) pairs to be optimized."""
        return self._optimization_pairs

    @property
    def bounds(self) -> Float[Array, "2 {self.n_params}"]:
        """Bounds for each parameter to be optimized. Shape: (2, n_params)."""
        return self._bounds

    def calculate_sensitivity(
        self,
        optimized_parameters: Float[Array, "{self.n_params}"],
    ) -> Float[Array, "n_frequencies"]:
        """Calculate the sensitivity curve for given parameters.

        Args:
            optimized_parameters: Parameters to evaluate.

        Returns:
            Sensitivity values at each frequency point.
        """
        # Use no bounding here since the parameters are expected to already be inside the bounds
        update_setup(
            optimized_parameters,
            self._optimization_pairs,
            self._bounds,
            self._setup,
            bounding_function=lambda x, b: x,
        )

        carrier, signal, noise, detector_ports, *_ = run(
            self._setup, [("f", "frequency")], self._frequencies
        )
        powers = signal_detector(carrier, signal)
        powers = powers[detector_ports]
        powers = powers[0] - powers[1]
        sensitivities = noise / jnp.abs(powers)

        return sensitivities

    def to_spec(self) -> dict:
        """Return a serializable spec sufficient to rebuild this problem."""
        spec = self._base_spec()
        spec["type"] = "VoyagerTuningProblem"
        if self._bounds_overrides:
            spec["bounds_overrides"] = {
                k: [float(v[0]), float(v[1])] for k, v in self._bounds_overrides.items()
            }
        return spec
