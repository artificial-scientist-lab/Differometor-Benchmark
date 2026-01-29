"""Base class for gravitational wave detector optimization problems."""

import json
import os
from abc import abstractmethod
from datetime import datetime

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float

from dfbench.core.protocols import ContinuousProblem


class OpticalSetupProblem(ContinuousProblem):
    """Abstract base class for optical setup optimization problems.

    This class extends ContinuousProblem with optical-specific functionality
    for gravitational wave detector optimization, including frequency-dependent
    sensitivity calculations.

    Inherits from ContinuousProblem to ensure compatibility with all optimization
    algorithms in the dfbench framework.
    """

    def __init__(self, name: str, n_frequencies: int = 100):
        """Initialize the optimization problem.

        Args:
            name (str): Name of the problem, used for output file naming.
            n_frequencies (int): Number of frequency points for sensitivity calculation.
                Defaults to 100.
        """
        self._name = name.lstrip("_")
        self._frequencies = jnp.logspace(jnp.log10(20), jnp.log10(5000), n_frequencies)
        self._target_sensitivities = None  # to be set by subclasses

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

    # objective_function and sigmoid_objective_function are set as instance
    # attributes by subclasses (as JIT-compiled callables), following the
    # pattern defined in ContinuousProblem protocol.

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

    def output_to_files(
        self,
        best_params: Float[Array, "{self.n_params}"] = None,
        losses: Float[Array, "iterations"] = None,
        population_losses: Float[Array, "iterations pop"] = None,
        algorithm_str: str = "",
        hyper_param_str: str = "",
        hyper_param_str_in_filename: bool = True,
    ) -> None:
        """Output optimization results to files.

        Creates JSON files with parameters and losses, and PNG plots of
        the optimization progress and final sensitivity curve.

        Args:
            best_params: Best parameters found during optimization.
            losses: Loss values over iterations/generations.
            population_losses: All population losses (for genetic algorithms).
            algorithm_str: Algorithm name for file naming.
            hyper_param_str: Hyperparameter string for file naming.
            hyper_param_str_in_filename: Whether to include hyperparameters in filename.
        """
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
            f"./data/problem_output/{self._name}/{algorithm_str.strip('_')}",
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

        sensitivities = self.calculate_sensitivity(best_params)

        plt.figure()
        plt.plot(self._frequencies, sensitivities, label="Optimized Sensitivity")

        plt.plot(
            self._frequencies, self._target_sensitivities, label="Target Sensitivity"
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
