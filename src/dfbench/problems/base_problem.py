"""Base class for gravitational wave detector optimization problems."""

import json
import os
from abc import abstractmethod
from datetime import datetime
from typing import Callable, Mapping

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import Array, Float

from differometor.components import (
    DETECTOR_POWER_THRESHOLD,
    HARD_SIDE_POWER_THRESHOLD,
    SOFT_SIDE_POWER_THRESHOLD,
)

from dfbench.core.problem import ContinuousProblem


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
    """No penalty — disables power constraints."""
    return jnp.zeros_like(value)


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
        self._power_penalty_fn: Callable = squashed_relu_penalty

    @property
    def power_penalty_fn(self) -> Callable:
        """The function used to compute per-element power-constraint penalties.

        Signature: ``fn(value, threshold) -> penalty_contribution``
        """
        return self._power_penalty_fn

    def _compute_power_violations(self, powers):
        """Apply ``power_penalty_fn`` to each component group and concatenate."""
        fn = self._power_penalty_fn
        hard = fn(powers[0].squeeze(1), HARD_SIDE_POWER_THRESHOLD)
        soft = fn(powers[1].squeeze(1), SOFT_SIDE_POWER_THRESHOLD)
        det = fn(powers[2].squeeze(1), DETECTOR_POWER_THRESHOLD)
        return jnp.concatenate([hard, det, soft], axis=0)

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
            if not allow_widen and (
                lower < default_lower or upper > default_upper
            ):
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
