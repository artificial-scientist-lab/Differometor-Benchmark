"""Voyager problem with realistic 3-noise model and power constraints."""

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from differometor.components import (
    HARD_SIDE_POWER_THRESHOLD,
    SOFT_SIDE_POWER_THRESHOLD,
    DETECTOR_POWER_THRESHOLD,
)
from differometor.setups import voyager
from differometor.simulate import run_setups, simulate, run_build_step
from differometor.utils import (
    sigmoid_bounding,
    sensitivity_qamplfreq_noise,
    calculate_sensitivities,
    calculate_powers,
)

from ..base_problem import OpticalSetupProblem


class ConstrainedVoyagerProblem(OpticalSetupProblem):
    """Voyager optimization with realistic 3-noise model and power constraints.

    This problem uses three different modulation modes (quantum noise, amplitude
    noise, frequency noise) to calculate a more realistic sensitivity. It also
    enforces power constraints on different components of the optical setup.
    """

    def __init__(
        self,
        n_frequencies: int = 100,
    ):
        """Initialize the Constrained Voyager optimization problem.

        Args:
            n_frequencies (int): Number of frequency points for sensitivity calculation.
                Defaults to 100.
        """
        super().__init__(name="voyager_constrained", n_frequencies=n_frequencies)

        ### Calculate the target sensitivity ###
        # --------------------------------------#

        # use a predefined Voyager setup with three different modulations (i.e. quantum noise, amplitude noise, frequency noise)
        q_setup, component_property_pairs = voyager(mode="space_modulation")
        ampl_setup, _ = voyager(mode="amplitude_modulation")
        freq_setup, _ = voyager(mode="frequency_modulation")
        self._setup = [q_setup, ampl_setup, freq_setup]

        # choose a sensitivity function that calculates sensitivities taking into account the three noise sources
        self._sensitivity_function = sensitivity_qamplfreq_noise

        # simulate the setups
        simulation_results = run_setups(self._setup, self._frequencies)

        # calculate the sensitivity values taking into account the three noise sources
        self._target_sensitivities = calculate_sensitivities(
            simulation_results, self._sensitivity_function, self._frequencies
        )

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

        # calculate the bounds for the properties to be optimized
        self._bounds = np.array(
            [
                [property_bounds[pair[1]][0], property_bounds[pair[1]][1]]
                for pair in self._optimization_pairs
            ]
        ).T

        # abstract for pure objective_function
        bounds = self._bounds

        # build the three modulation setups and store as instance attributes
        # Use the setups already created above
        self._q_arrays, *self._q_metadata = run_build_step(
            self._setup[0],
            [("f", "frequency")],
            self._frequencies,
            self._optimization_pairs,
        )
        self._ampl_arrays, *self._ampl_metadata = run_build_step(
            self._setup[1],
            [("f", "frequency")],
            self._frequencies,
            self._optimization_pairs,
        )
        self._freq_arrays, *self._freq_metadata = run_build_step(
            self._setup[2],
            [("f", "frequency")],
            self._frequencies,
            self._optimization_pairs,
        )

        @jax.jit
        def sigmoid_objective_function(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> Float:
            optimized_parameters = sigmoid_bounding(optimized_parameters, bounds)

            # simulate the three modulation setups
            q_results = simulate(**{**self._q_arrays, 'optimized_parameters': optimized_parameters})
            ampl_results = simulate(
                **{**self._ampl_arrays, 'optimized_parameters': optimized_parameters}
            )
            freq_results = simulate(
                **{**self._freq_arrays, 'optimized_parameters': optimized_parameters}
            )
            results = [
                (*q_results, *self._q_metadata),
                (*ampl_results, *self._ampl_metadata),
                (*freq_results, *self._freq_metadata),
            ]

            # calculate the sensitivities taking into account the three noise sources
            sensitivities = calculate_sensitivities(
                results, self._sensitivity_function, self._frequencies, homodyne=True
            )

            # calculate the light power at all components within the setup
            powers = calculate_powers(q_results[0], *self._q_metadata)

            # calculate the loss taking into account power violations
            sensitivity_loss, penalty, _ = self._calculate_loss(
                sensitivities, self._target_sensitivities, powers
            )
            penalty = penalty / (1.0 + penalty)

            return sensitivity_loss + penalty

        @jax.jit
        def objective_function(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> Float:
            # simulate the three modulation setups
            q_results = simulate(**{**self._q_arrays, 'optimized_parameters': optimized_parameters})
            ampl_results = simulate(
                **{**self._ampl_arrays, 'optimized_parameters': optimized_parameters}
            )
            freq_results = simulate(
                **{**self._freq_arrays, 'optimized_parameters': optimized_parameters}
            )
            results = [
                (*q_results, *self._q_metadata),
                (*ampl_results, *self._ampl_metadata),
                (*freq_results, *self._freq_metadata),
            ]

            # calculate the sensitivities taking into account the three noise sources
            sensitivities = calculate_sensitivities(
                results, self._sensitivity_function, self._frequencies, homodyne=True
            )

            # calculate the light power at all components within the setup
            powers = calculate_powers(q_results[0], *self._q_metadata)

            # calculate the loss taking into account power violations
            sensitivity_loss, penalty, _ = self._calculate_loss(
                sensitivities, self._target_sensitivities, powers
            )
            penalty = penalty / (1.0 + penalty)

            return sensitivity_loss + penalty

        self.sigmoid_objective_function = sigmoid_objective_function
        self.objective_function = objective_function

    def _calculate_loss(
        self,
        sensitivities,
        reference_sensitivities,
        powers,
    ):
        """Calculate loss and penalties from sensitivities and power constraints."""
        # calculate power violations (i.e. penalty based on much the power at each component exceeds its threshold)
        hard_side_violations = jnp.maximum(
            powers[0] / HARD_SIDE_POWER_THRESHOLD - 1, 0
        ).squeeze(1)
        soft_side_violations = jnp.maximum(
            powers[1] / SOFT_SIDE_POWER_THRESHOLD - 1, 0
        ).squeeze(1)
        detector_violations = jnp.maximum(
            powers[2] / DETECTOR_POWER_THRESHOLD - 1, 0
        ).squeeze(1)

        violations = jnp.concatenate(
            [hard_side_violations, detector_violations, soft_side_violations], axis=0
        )

        losses = jnp.mean(jnp.log10(sensitivities.T / reference_sensitivities), axis=-1)
        penalties = jnp.sum(violations.T, axis=-1)

        return losses, penalties, violations

    @property
    def optimization_pairs(self) -> list[tuple]:
        """List of (component, property) pairs to be optimized."""
        return self._optimization_pairs

    @property
    def bounds(self) -> Float[Array, "2 {self.n_params}"]:
        """Bounds for each parameter to be optimized. Shape: (2, n_params)."""
        return self._bounds

    @property
    def n_params(self) -> int:
        """Number of parameters to be optimized. Is equal to len(self.optimization_pairs)."""
        return len(self._optimization_pairs)

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
        # simulate the three modulation setups
        q_results = simulate(**{**self._q_arrays, 'optimized_parameters': optimized_parameters})
        ampl_results = simulate(
            **{**self._ampl_arrays, 'optimized_parameters': optimized_parameters}
        )
        freq_results = simulate(
            **{**self._freq_arrays, 'optimized_parameters': optimized_parameters}
        )
        results = [
            (*q_results, *self._q_metadata),
            (*ampl_results, *self._ampl_metadata),
            (*freq_results, *self._freq_metadata),
        ]

        # calculate the sensitivities taking into account the three noise sources
        sensitivities = calculate_sensitivities(
            results, self._sensitivity_function, self._frequencies, homodyne=True
        )  # Voyager uses homodyne detection

        return sensitivities
