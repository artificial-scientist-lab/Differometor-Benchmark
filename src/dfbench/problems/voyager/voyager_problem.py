"""Voyager problem with single-noise sensitivity."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from differometor.components import demodulate_signal_power
from differometor.setups import voyager
from differometor.simulate import run, run_build_step, simulate_in_parallel
from differometor.utils import sigmoid_bounding, update_setup

from ..base_problem import OpticalSetupProblem


class VoyagerProblem(OpticalSetupProblem):
    """Voyager optimization with single-noise sensitivity.

    Uses a predefined Voyager setup with one noise detector and two signal
    detectors for balanced homodyne detection. The loss is relative to the
    target Voyager setup.
    """

    def __init__(
        self,
        n_frequencies: int = 100,
    ):
        """Initialize the Voyager optimization problem.

        Args:
            n_frequencies (int): Number of frequency points for sensitivity calculation.
                Defaults to 100.
        """
        super().__init__(name="voyager", n_frequencies=n_frequencies)

        # use a predefined Voyager setup with one noise detector and two signal detectors
        self._setup, component_property_pairs = voyager()

        # run the simulation with the frequency as the changing parameter
        carrier, signal, noise, detector_ports, *_ = run(
            self._setup, [("f", "frequency")], self._frequencies
        )

        # calculate the signal power at the detector ports
        powers = demodulate_signal_power(carrier, signal)
        powers = powers[detector_ports]

        # calculate the signal power from the two signal detectors for balanced homodyne detection
        powers = powers[0] - powers[1]

        # calculate the sensitivity
        self._target_sensitivities = noise / jnp.abs(powers)
        target_loss = jnp.sum(jnp.log10(self._target_sensitivities))

        ### Start from random parameters and optimize the sensitivity ###
        # ---------------------------------------------------------------#

        # specify the ranges for the properties to be optimized
        property_bounds = {
            "reflectivity": [0, 1],
            "tuning": [0, 90],
            "db": [0.01, 20],
            "angle": [-180, 180],
            "power": [0.01, 200],
            "mass": [0.01, 200],
            "length": [1, 4000],
            "phase": [-180, 180],
        }

        # select properties to be optimized
        optimized_properties = [
            "reflectivity",
            "tuning",
            "db",
            "angle",
            "power",
            "mass",
            "length",
            "phase",
        ]
        self._optimization_pairs = []
        for pair in component_property_pairs:
            if pair[1] in optimized_properties:
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

        # abstract for pure objective_function
        bounds = self._bounds

        @jax.jit
        def objective_function(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> Float:
            carrier, signal, noise = simulate_in_parallel(
                optimized_parameters, *self._simulation_arrays[1:]
            )
            powers = demodulate_signal_power(carrier, signal)
            powers = powers[self._detector_ports]
            powers = powers[0] - powers[1]
            sensitivities = noise / jnp.abs(powers)

            # loss relative to target loss => loss < 0 is better than voyager setup
            return jnp.sum(jnp.log10(sensitivities)) - target_loss

        self.objective_function = objective_function

        @jax.jit
        def sigmoid_objective_function(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> Float:
            optimized_parameters = sigmoid_bounding(optimized_parameters, bounds)
            carrier, signal, noise = simulate_in_parallel(
                optimized_parameters, *self._simulation_arrays[1:]
            )
            powers = demodulate_signal_power(carrier, signal)
            powers = powers[self._detector_ports]
            powers = powers[0] - powers[1]
            sensitivities = noise / jnp.abs(powers)

            # loss relative to target loss => loss < 0 is better than voyager setup
            return jnp.sum(jnp.log10(sensitivities)) - target_loss

        self.sigmoid_objective_function = sigmoid_objective_function

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
        powers = demodulate_signal_power(carrier, signal)
        powers = powers[detector_ports]
        powers = powers[0] - powers[1]
        sensitivities = noise / jnp.abs(powers)

        return sensitivities
