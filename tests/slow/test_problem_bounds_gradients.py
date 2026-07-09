"""Boundary value and gradient checks for real Differometor problems."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


pytestmark = pytest.mark.slow


def _boundary_points(problem):
    bounds = jnp.asarray(problem.bounds)
    lower = bounds[0]
    upper = bounds[1]
    midpoint = 0.5 * (lower + upper)

    names = ["all_lower", "all_upper", "midpoint"]
    points = [lower, upper, midpoint]

    for index in range(problem.n_params):
        names.append(f"lower_{index}")
        points.append(midpoint.at[index].set(lower[index]))
        names.append(f"upper_{index}")
        points.append(midpoint.at[index].set(upper[index]))

    return names, jnp.stack(points)


def _format_bad_gradient_rows(problem, names, losses, grads):
    losses_np = np.asarray(losses)
    grads_np = np.asarray(grads)
    finite_grads = np.isfinite(grads_np)
    bad_rows = np.flatnonzero(~np.all(finite_grads, axis=1))

    lines = []
    for row in bad_rows:
        bad_cols = np.flatnonzero(~finite_grads[row])
        details = []
        for col in bad_cols:
            pair = problem.optimization_pairs[int(col)]
            details.append(f"{int(col)}:{pair}")
        lines.append(
            f"{names[int(row)]}: loss={losses_np[row]}, cols=[{', '.join(details)}]"
        )
    return "\n".join(lines)


def _problem_cases():
    from dfbench.problems import (
        ConstrainedVoyagerProblem,
        UIFOProblem,
        VoyagerProblem,
        VoyagerTuningProblem,
    )

    return [
        pytest.param(
            VoyagerProblem,
            {"n_frequencies": 8},
            id="voyager",
        ),
        pytest.param(
            VoyagerTuningProblem,
            {"n_frequencies": 8},
            id="voyager_tuning",
        ),
        pytest.param(
            ConstrainedVoyagerProblem,
            {"n_frequencies": 8},
            id="constrained_voyager",
        ),
        pytest.param(
            UIFOProblem,
            {"size": 2, "topology_seed": 42, "n_frequencies": 8},
            id="uifo",
        ),
    ]


@pytest.mark.parametrize("problem_cls, kwargs", _problem_cases())
def test_objective_values_are_finite_at_all_parameter_bounds(problem_cls, kwargs):
    """Every problem should be value-finite at its declared parameter bounds."""
    problem = problem_cls(**kwargs)
    names, points = _boundary_points(problem)
    losses = jax.vmap(problem.objective_function)(points)

    bad_rows = np.flatnonzero(~np.isfinite(np.asarray(losses)))
    assert bad_rows.size == 0, "\n".join(
        f"{names[int(row)]}: loss={np.asarray(losses)[row]}" for row in bad_rows
    )


@pytest.mark.parametrize("problem_cls, kwargs", _problem_cases())
def test_objective_gradients_are_finite_at_all_parameter_bounds(problem_cls, kwargs):
    """Gradient finiteness at declared bounds, reported by property on failure."""
    problem = problem_cls(**kwargs)
    names, points = _boundary_points(problem)
    losses, grads = jax.vmap(jax.value_and_grad(problem.objective_function))(points)

    assert jnp.all(jnp.isfinite(losses))
    assert jnp.all(jnp.isfinite(grads)), _format_bad_gradient_rows(
        problem, names, losses, grads
    )
