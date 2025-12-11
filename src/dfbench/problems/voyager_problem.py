import json
import os
from datetime import datetime

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Array, Float

import differometor as df
from differometor.components import demodulate_signal_power
from differometor.setups import voyager
from differometor.utils import sigmoid_bounding, update_setup


class VoyagerProblem:
    def __init__(
        self,
        name: str = "voyager",
        n_frequencies: int = 100,
    ):
        """Initialize the Voyager optimization problem.

        Args:
            name (str): Name of the problem, used for output file naming. Defaults to "voyager".
            n_frequencies (int): Number of frequency points for sensitivity calculation.
                Defaults to 100.
        """
        self._name = name.lstrip("_")
        ### Calculate the target sensitivity ###
        # --------------------------------------#
        # use a predefined Voyager setup with one noise detector and two signal detectors
        self._setup, component_property_pairs = voyager()

        # set the frequency range
        self._frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), n_frequencies)

        # run the simulation with the frequency as the changing parameter
        carrier, signal, noise, detector_ports, *_ = df.run(
            self._setup, [("f", "frequency")], self._frequencies
        )

        # calculate the signal power at the detector ports
        powers = demodulate_signal_power(carrier, signal)
        powers = powers[detector_ports]

        # calculate the signal power from the two signal detectors for balanced homodyne detection
        powers = powers[0] - powers[1]

        # calculate the sensitivity
        self._target_sensitivity = noise / jnp.abs(powers)
        target_loss = jnp.sum(jnp.log10(self._target_sensitivity))

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
        simulation_arrays, detector_ports, *_ = df.run_build_step(
            self._setup,
            [("f", "frequency")],
            self._frequencies,
            self._optimization_pairs,
        )

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
            carrier, signal, noise = df.simulate_in_parallel(
                optimized_parameters, *simulation_arrays[1:]
            )
            powers = demodulate_signal_power(carrier, signal)
            powers = powers[detector_ports]
            powers = powers[0] - powers[1]
            sensitivity = noise / jnp.abs(powers)

            # loss relative to target loss => loss < 0 is better than voyager setup
            return jnp.sum(jnp.log10(sensitivity)) - target_loss

        self.objective_function = objective_function

        @jax.jit
        def sigmoid_objective_function(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> Float:
            optimized_parameters = sigmoid_bounding(optimized_parameters, bounds)
            carrier, signal, noise = df.simulate_in_parallel(
                optimized_parameters, *simulation_arrays[1:]
            )
            powers = demodulate_signal_power(carrier, signal)
            powers = powers[detector_ports]
            powers = powers[0] - powers[1]
            sensitivity = noise / jnp.abs(powers)

            # loss relative to target loss => loss < 0 is better than voyager setup
            return jnp.sum(jnp.log10(sensitivity)) - target_loss

        self.sigmoid_objective_function = sigmoid_objective_function

    @property
    def optimization_pairs(
        self,
    ) -> list[tuple]:  # TODO: Find out what's inside that tuple
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
    def frequencies(self) -> Float[Array, "n_frequencies"]:
        """Frequencies at which the sensitivity is calculated."""
        return self._frequencies

    def calculate_sensitivity(
        self,
        optimized_parameters: Float[Array, "{self.n_params}"],
    ) -> Float[Array, "n_frequencies"]:
        update_setup(
            optimized_parameters, self._optimization_pairs, self._bounds, self._setup
        )

        carrier, signal, noise, detector_ports, *_ = df.run(
            self._setup, [("f", "frequency")], self._frequencies
        )
        powers = demodulate_signal_power(carrier, signal)
        powers = powers[detector_ports]
        powers = powers[0] - powers[1]
        sensitivity = noise / jnp.abs(powers)

        return sensitivity

    def output_to_files(
        self,
        best_params: Float[Array, "{self.n_params}"] = None,
        losses: Float[Array, "iterations"] = None,
        population_losses: Float[Array, "iterations pop"] = None,
        algorithm_str: str = "",
        hyper_param_str: str = "",
        hyper_param_str_in_filename: bool = True,
    ) -> None:
        # Print best params and loss first
        print(f"Parameters of the best solution : {best_params}")
        print(
            f"Fitness value of the best solution = {self.objective_function(best_params)}"
        )

        # Prepare strings and timestamp
        algorithm_str = f"_{algorithm_str.strip('_')}" if algorithm_str != "" else ""
        hyper_param_str = (
            f"_{hyper_param_str.strip('_')}" if hyper_param_str != "" else ""
        )
        timestamp = datetime.now().strftime("_%Y-%m-%d_%H-%M")

        # Create output directory
        output_path = os.path.join(
            f"./algorithm_output/{self._name}/{algorithm_str.strip('_')}",
            hyper_param_str.strip("_"),  # directory should not have leading underscore
        )
        os.makedirs(output_path, exist_ok=True)

        # Send info to user
        print(f"Output directory: {output_path}")

        # Determine file name prefix and suffix
        file_prefix = f"{self._name}{algorithm_str}{timestamp}"
        file_suffix = hyper_param_str if hyper_param_str_in_filename else ""

        # Output best parameters to JSON
        with open(
            os.path.join(output_path, f"{file_prefix}_parameters{file_suffix}.json"),
            "w",
        ) as f:
            json.dump(best_params.tolist(), f, indent=4)

        # Output historical losses to JSON
        with open(
            os.path.join(output_path, f"{file_prefix}_losses{file_suffix}.json"),
            "w",
        ) as f:
            json.dump(losses.tolist(), f, indent=4)

        is_genetic = population_losses is not None

        plt.figure()
        plt.plot(losses)
        plt.xlabel("Generation" if is_genetic else "Iteration")
        plt.ylabel("Best losses" if is_genetic else "Loss")
        plt.axhline(0, color="red", linestyle="--")
        plt.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"{file_prefix}_losses{file_suffix}.png"))

        if population_losses is not None:
            plt.figure()
            plt.plot(population_losses)
            plt.xlabel("Generation")
            plt.ylabel("All losses")
            plt.axhline(0, color="red", linestyle="--")
            plt.grid()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    output_path, f"{file_prefix}_population_losses{file_suffix}.png"
                )
            )

        ### Calculate the sensitivity of the best found setup ###
        # -------------------------------------------------------#

        sensitivity = self.calculate_sensitivity(best_params)

        plt.figure()
        plt.plot(self._frequencies, sensitivity, label="Optimized Sensitivity")
        plt.plot(
            self._frequencies, self._target_sensitivity, label="Target Sensitivity"
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Sensitivity [/sqrt(Hz)]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_path, f"{file_prefix}_sensitivity{file_suffix}.png")
        )
