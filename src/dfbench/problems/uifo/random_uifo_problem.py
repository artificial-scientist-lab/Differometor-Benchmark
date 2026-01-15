"""UIFO (Uniform Interferometer Field Optimization) problems with power constraints."""

import copy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from differometor.components import (
    HARD_SIDE_POWER_THRESHOLD,
    SOFT_SIDE_POWER_THRESHOLD,
    DETECTOR_POWER_THRESHOLD,
)
from differometor.setups import uifo, constrain_inter_grid_cell_spaces
from differometor.simulate import run_setups, simulate_in_parallel, run_build_step
from differometor.utils import (
    sigmoid_bounding,
    sensitivity_qamplfreq_noise,
    calculate_sensitivities,
    calculate_powers,
)

from ..base_problem import OpticalSetupProblem


class RandomUIFOProblem(OpticalSetupProblem):
    """UIFO problem with random topology generated from a seed.

    Creates random interferometer configurations in a grid pattern. The topology
    is fixed at initialization and only continuous optical parameters are optimized.
    """

    def __init__(
        self,
        size: int = 3,
        n_frequencies: int = 100,
        topology_seed: int = 42,
    ):
        """Initialize the random UIFO optimization problem.

        Args:
            size: Grid size (e.g., 3 for 3x3, 5 for 5x5). Defaults to 3.
            n_frequencies: Number of frequency points. Defaults to 100.
            topology_seed: Seed for random topology generation. Defaults to 42.
        """
        super().__init__(
            name=f"uifo_{size}x{size}_seed{topology_seed}", n_frequencies=n_frequencies
        )
        self._size = size
        self._topology_seed = topology_seed

        ### Calculate the target sensitivity using Voyager reference ###
        # -------------------------------------------------------------#

        # Import voyager for reference sensitivity
        from differometor.setups import voyager

        # use a predefined Voyager setup with three different modulations for reference
        q_setup, _ = voyager(mode="space_modulation")
        ampl_setup, _ = voyager(mode="amplitude_modulation")
        freq_setup, _ = voyager(mode="frequency_modulation")
        reference_setups = [q_setup, ampl_setup, freq_setup]

        # choose a sensitivity function that calculates sensitivities taking into account the three noise sources
        self._sensitivity_function = partial(
            sensitivity_qamplfreq_noise, frequencies=self._frequencies
        )

        # simulate the reference setups
        simulation_results = run_setups(reference_setups, self._frequencies)

        # calculate the sensitivity values taking into account the three noise sources
        self._target_sensitivities = calculate_sensitivities(
            simulation_results, self._sensitivity_function
        )

        ### Create random UIFO setup ###
        # ------------------------------#

        # define a random uifo with three different modulations
        q_noise_setup, component_property_pairs, centers, boundaries = uifo(
            size=size,
            mode="space_modulation",
            random=True,
            verbose=True,
            random_seed=topology_seed,
        )
        ampl_noise_setup, _ = uifo(
            size=size,
            mode="amplitude_modulation",
            centers=centers,
            boundaries=boundaries,
        )
        freq_noise_setup, _ = uifo(
            size=size,
            mode="frequency_modulation",
            centers=centers,
            boundaries=boundaries,
        )

        self._setup = [q_noise_setup, ampl_noise_setup, freq_noise_setup]

        # check if the random uifo uses a balanced homodyne detection scheme
        self._homodyne = False
        for node in q_noise_setup.nodes:
            if node[1]["component"] == "qhd":
                self._homodyne = True
                break

        ### Setup optimization parameters ###
        # ----------------------------------#

        # select properties to be optimized
        optimized_properties = [
            "reflectivity",
            "tuning",
            "db",
            "angle",
            "power",
            "mass",
            "length",
        ]

        # specify the ranges for the properties to be optimized
        property_bounds = {
            "db": [0, 10],
            "angle": [-360, 360],
            "power": [0, 200],
            "tuning": [-360, 360],
            "mass": [0.01, 200],
            "length": [0.1, 4000],
            "reflectivity": [0, 1],
        }

        # couple vertical and horizontal spaces at same positions, so that the grid structure of the uifo is always preserved
        self._optimization_pairs = constrain_inter_grid_cell_spaces(
            component_property_pairs, optimized_properties
        )

        # calculate the bounds for the properties to be optimized
        lower_bounds = []
        upper_bounds = []
        for optimization_pair in self._optimization_pairs:
            if isinstance(optimization_pair[0], list):
                property_name = optimization_pair[0][1]
            else:
                property_name = optimization_pair[1]
            lower_bounds.append(property_bounds[property_name][0])
            upper_bounds.append(property_bounds[property_name][1])
        self._bounds = np.array([lower_bounds, upper_bounds])

        # abstract for pure objective_function
        bounds = self._bounds

        # build the three modulation setups and store as instance attributes
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
            q_results = simulate_in_parallel(optimized_parameters, *self._q_arrays[1:])
            ampl_results = simulate_in_parallel(
                optimized_parameters, *self._ampl_arrays[1:]
            )
            freq_results = simulate_in_parallel(
                optimized_parameters, *self._freq_arrays[1:]
            )
            results = [
                (*q_results, *self._q_metadata),
                (*ampl_results, *self._ampl_metadata),
                (*freq_results, *self._freq_metadata),
            ]

            # calculate the sensitivities taking into account the three noise sources
            sensitivities = calculate_sensitivities(
                results, self._sensitivity_function, homodyne=self._homodyne
            )

            # calculate the light power at all components within the setup
            powers = calculate_powers(q_results[0], *self._q_metadata)

            # calculate the loss taking into account power violations
            sensitivity_loss, penalty, _ = self._calculate_loss(
                sensitivities, self._target_sensitivities, powers
            )

            return sensitivity_loss + penalty

        @jax.jit
        def objective_function(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> Float:
            # simulate the three modulation setups
            q_results = simulate_in_parallel(optimized_parameters, *self._q_arrays[1:])
            ampl_results = simulate_in_parallel(
                optimized_parameters, *self._ampl_arrays[1:]
            )
            freq_results = simulate_in_parallel(
                optimized_parameters, *self._freq_arrays[1:]
            )
            results = [
                (*q_results, *self._q_metadata),
                (*ampl_results, *self._ampl_metadata),
                (*freq_results, *self._freq_metadata),
            ]

            # calculate the sensitivities taking into account the three noise sources
            sensitivities = calculate_sensitivities(
                results, self._sensitivity_function, homodyne=self._homodyne
            )

            # calculate the light power at all components within the setup
            powers = calculate_powers(q_results[0], *self._q_metadata)

            # calculate the loss taking into account power violations
            sensitivity_loss, penalty, _ = self._calculate_loss(
                sensitivities, self._target_sensitivities, powers
            )

            return sensitivity_loss + penalty

        self.sigmoid_objective_function = sigmoid_objective_function
        self.objective_function = objective_function

    def _calculate_loss(self, sensitivities, reference_sensitivities, powers):
        """Calculate loss and penalties from sensitivities and power constraints."""
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

    @property
    def topology_seed(self) -> int:
        """The seed used to generate this problem's topology."""
        return self._topology_seed

    @property
    def structure_info(self) -> dict:
        """Metadata about the problem's discrete structure."""
        return {
            "size": self._size,
            "topology_seed": self._topology_seed,
            "n_params": self.n_params,
            "homodyne": self._homodyne,
        }

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
        q_results = simulate_in_parallel(optimized_parameters, *self._q_arrays[1:])
        ampl_results = simulate_in_parallel(
            optimized_parameters, *self._ampl_arrays[1:]
        )
        freq_results = simulate_in_parallel(
            optimized_parameters, *self._freq_arrays[1:]
        )
        results = [
            (*q_results, *self._q_metadata),
            (*ampl_results, *self._ampl_metadata),
            (*freq_results, *self._freq_metadata),
        ]

        # calculate the sensitivities taking into account the three noise sources
        sensitivities = calculate_sensitivities(
            results, self._sensitivity_function, homodyne=self._homodyne
        )

        return sensitivities
