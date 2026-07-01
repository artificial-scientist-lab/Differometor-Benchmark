import numpy as np
import pytest

from dfbench.core.objective import Objective
from dfbench.core.parameters import DiscreteParameter, FloatParameter
from dfbench.core.search_space import SearchDimension, SearchSpace, TargetRef


def test_search_space_from_problem_exposes_existing_schema(mock_problem):
    space = mock_problem.search_space

    assert space.name == mock_problem.name
    assert space.n_params == mock_problem.n_params
    assert space.names() == ("comp_0.param", "comp_1.param")
    assert space.optimization_pairs() == tuple(mock_problem.optimization_pairs)
    np.testing.assert_allclose(space.bounds_array(), np.asarray(mock_problem.bounds))


def test_objective_exposes_problem_search_space(mock_problem):
    objective = Objective(mock_problem)

    assert objective.search_space.to_dict() == mock_problem.search_space.to_dict()


def test_coupled_optimization_pair_becomes_one_dimension():
    space = SearchSpace.from_bounds(
        bounds=np.array([[0.1], [4000.0]]),
        optimization_pairs=[
            [
                ["mr11_ml12", "length"],
                ["mr21_ml22", "length"],
            ]
        ],
        name="uifo_length_group",
    )

    dimension = space.dimensions[0]
    assert dimension.is_coupled
    assert dimension.name == "mr11_ml12.length.coupled_0"
    assert dimension.optimization_pair() == (
        ("mr11_ml12", "length"),
        ("mr21_ml22", "length"),
    )
    np.testing.assert_allclose(space.bounds_array(), np.array([[0.1], [4000.0]]))


def test_search_space_sampling_validation_and_serialization():
    space = SearchSpace(
        name="mixed_space",
        dimensions=(
            SearchDimension(
                name="mirror.tuning",
                parameter=FloatParameter("mirror.tuning", lower=-180, upper=180),
                targets=(TargetRef("mirror", "tuning"),),
            ),
            SearchDimension(
                name="boundary.kind",
                parameter=DiscreteParameter(
                    "boundary.kind",
                    choices=("laser", "squeezer"),
                ),
                targets=(TargetRef("boundary", "kind"),),
            ),
        ),
    )

    sample = space.sample(seed=0)

    space.validate(sample)
    assert set(space.to_dict()) == {"name", "n_dims", "dimensions", "metadata"}
    assert "mixed_space" in space.to_json()
    with pytest.raises(TypeError):
        space.bounds_array()


def test_rejects_mismatched_bounds_and_pairs():
    with pytest.raises(ValueError, match="same number"):
        SearchSpace.from_bounds(
            bounds=np.array([[0.0, 0.0], [1.0, 1.0]]),
            optimization_pairs=[("only_one", "param")],
        )
