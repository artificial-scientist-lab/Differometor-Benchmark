"""Voyager problem with realistic 3-noise model and power constraints."""

import jax
import numpy as np
from jaxtyping import Array, Float
from differometor.setups import voyager
from differometor.simulate import run_setups, simulate, run_build_step
from differometor.utils import (
    sensitivity_qamplfreq_noise,
    calculate_sensitivities,
    calculate_powers,
)

from ..base_problem import DEFAULT_SIGNAL_FLOOR, OpticalSetupProblem, register_problem


@register_problem
class ConstrainedVoyagerProblem(OpticalSetupProblem):
    """Voyager optimization with realistic 3-noise model and power constraints.

    This problem uses three different modulation modes (quantum noise, amplitude
    noise, frequency noise) to calculate a more realistic sensitivity. It also
    enforces power constraints on different components of the optical setup.
    """

    _supports_power_penalty = True

    def __init__(
        self,
        n_frequencies: int = 50,
        power_penalty_fn=None,
        bounds_overrides: dict[str, tuple[float, float]] | None = None,
        signal_floor: float = DEFAULT_SIGNAL_FLOOR,
    ):
        """Initialize the Constrained Voyager optimization problem.

        Args:
            n_frequencies (int): Number of frequency points for sensitivity calculation.
                Defaults to 100.
            power_penalty_fn: A callable ``fn(value, threshold) -> penalty`` applied
                per-element to compute power-constraint violations.  Built-in
                options are ``squashed_relu_penalty`` (default),
                ``relu_penalty``, and ``zero_penalty`` from
                ``dfbench.problems.base_problem``.
            bounds_overrides: Optional property-level bound overrides.
                Example: {"tuning": (0, 45)}.
                Overrides must narrow default bounds.
            signal_floor: Optional lower bound for detector signal
                magnitudes before sensitivity normalization.
        """
        super().__init__(
            name="voyager_constrained",
            n_frequencies=n_frequencies,
            signal_floor=signal_floor,
        )
        signal_floor = self._signal_floor
        if power_penalty_fn is not None:
            self._power_penalty_fn = power_penalty_fn
        self._bounds_overrides = bounds_overrides

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
            simulation_results,
            self._sensitivity_function,
            self._frequencies,
            True,
            signal_floor,
        )

        # specify the ranges for the properties to be optimized
        property_bounds = {
            "reflectivity": [0, 1],
            "tuning": [-180, 180],
            "db": [0.01, 10],
            "angle": [-180, 180],
            "power": [0.01, 200],
            "mass": [0.01, 200],
            "length": [1, 4000],
            "phase": [-180, 180],
        }
        property_bounds = self._apply_property_bounds_overrides(
            property_bounds,
            bounds_overrides,
        )

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

        self._build_objective_function()

    def _eval_core(self, optimized_parameters):
        """Shared evaluation body for the constrained Voyager objective.

        Runs the three modulation simulations, computes sensitivities and
        per-group powers, then the loss tuple. Used by both
        ``objective_function`` and ``objective_function_aux`` so the two
        stay in sync after ``_build_objective_function`` re-traces them.
        """
        q_results = simulate(
            **{**self._q_arrays, "optimized_parameters": optimized_parameters}
        )
        ampl_results = simulate(
            **{**self._ampl_arrays, "optimized_parameters": optimized_parameters}
        )
        freq_results = simulate(
            **{**self._freq_arrays, "optimized_parameters": optimized_parameters}
        )
        results = [
            (*q_results, *self._q_metadata),
            (*ampl_results, *self._ampl_metadata),
            (*freq_results, *self._freq_metadata),
        ]

        sensitivities = calculate_sensitivities(
            results,
            self._sensitivity_function,
            self._frequencies,
            True,
            self._signal_floor,
        )
        powers = calculate_powers(q_results[0], *self._q_metadata)
        sensitivity_loss, penalty, violations = self._calculate_loss(
            sensitivities, self._target_sensitivities, powers
        )
        return powers, sensitivity_loss, penalty, violations

    def _build_objective_function(self) -> None:
        """(Re)build the JIT-compiled objective and aux objective.

        Re-tracing picks up the current ``_power_penalty_fn`` so that
        ``set_penalty_fn`` takes effect on subsequent evaluations. Both
        the plain and the aux variant close over the same ``_eval_core``
        call so their results agree up to the returned aux dict.
        """

        @jax.jit
        def objective_function(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> Float:
            _, sensitivity_loss, penalty, _ = self._eval_core(optimized_parameters)
            return sensitivity_loss + penalty

        @jax.jit
        def objective_function_aux(
            optimized_parameters: Float[Array, "{self.n_params}"],
        ) -> tuple[Float, dict]:
            powers, sensitivity_loss, penalty, violations = self._eval_core(
                optimized_parameters
            )
            aux = self._build_aux(powers, sensitivity_loss, penalty, violations)
            return sensitivity_loss + penalty, aux

        self.objective_function = objective_function
        self.objective_function_aux = objective_function_aux

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
        q_results = simulate(
            **{**self._q_arrays, "optimized_parameters": optimized_parameters}
        )
        ampl_results = simulate(
            **{**self._ampl_arrays, "optimized_parameters": optimized_parameters}
        )
        freq_results = simulate(
            **{**self._freq_arrays, "optimized_parameters": optimized_parameters}
        )
        results = [
            (*q_results, *self._q_metadata),
            (*ampl_results, *self._ampl_metadata),
            (*freq_results, *self._freq_metadata),
        ]

        # calculate the sensitivities taking into account the three noise sources
        sensitivities = calculate_sensitivities(
            results,
            self._sensitivity_function,
            self._frequencies,
            True,
            self._signal_floor,
        )  # Voyager uses homodyne detection

        return sensitivities

    def to_spec(self) -> dict:
        """Return a serializable spec sufficient to rebuild this problem."""
        spec = self._base_spec()
        spec["type"] = "ConstrainedVoyagerProblem"
        if self._bounds_overrides:
            spec["bounds_overrides"] = {
                k: [float(v[0]), float(v[1])] for k, v in self._bounds_overrides.items()
            }
        return spec
