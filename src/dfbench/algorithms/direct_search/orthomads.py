"""Orthogonal Mesh Adaptive Direct Search (OrthoMADS) algorithm via OMADS.

OrthoMADS is a refinement of MADS that constructs poll directions using
Halton-based orthogonal matrices.  This guarantees that at each iteration
the 2n poll directions span the full n-dimensional space, which gives
stronger convergence properties on smooth objectives while retaining the
mesh-based framework for non-smooth problems.

This implementation uses the **ORTHO** mesh back-end in OMADS (the
library's default).

Bounded vs unbounded behaviour
--------------------------------
OrthoMADS operates exclusively in **bounded physical space** (``unbounded=False``).
The poll candidates are clipped to the problem bounds by OMADS internally.
Passing ``unbounded=True`` to :meth:`prepare` is not supported and the
algorithm will raise :class:`RuntimeError` if attempted.

Constraints
-----------
Only box constraints (via ``problem.bounds``) are supported.  Passing a
problem with additional constraints will work silently but the extra
constraints are not communicated to OMADS.

Example
-------
>>> from dfbench import Objective
>>> from dfbench.problems import VoyagerProblem
>>> problem = VoyagerProblem()
>>> obj = Objective(problem, unbounded=False, max_evals=500)
>>> optimizer = OrthoMADS(poll_size_init=0.5)
>>> optimizer.optimize(obj, random_seed=42)
"""

from __future__ import annotations

import numpy as np
from jaxtyping import Array, Float

from dfbench.algorithms.direct_search._omads_wrapper import _run_omads_poll
from dfbench.core.algorithm import AlgorithmType, OptimizationAlgorithm
from dfbench.core.objective import Objective


class OrthoMADS(OptimizationAlgorithm):
    """Orthogonal Mesh Adaptive Direct Search (OrthoMADS) optimizer.

    Uses the OMADS library (ORTHO back-end) to run the OrthoMADS poll step
    as a rugged-landscape local explorer.  The orthogonal poll directions
    provide full-space coverage at each iteration, giving stronger
    theoretical guarantees than standard MADS on smooth objectives.

    Attributes:
        algorithm_str (str): ``"orthomads"``
        algorithm_type (AlgorithmType): ``DIRECT_SEARCH``

    Parameters
    ----------
    poll_size_init:
        Initial poll-frame size, expressed as a fraction of the variable
        scaling.  Smaller values restrict the initial neighbourhood;
        larger values allow broader early exploration.  Default: ``1.0``.
    min_poll_size:
        Frame-size tolerance at which the run is terminated (convergence
        criterion).  Default: ``1e-9``.

    Example
    -------
    >>> from dfbench import Objective
    >>> from dfbench.problems import VoyagerProblem
    >>> problem = VoyagerProblem()
    >>> obj = Objective(problem, unbounded=False, max_evals=300)
    >>> OrthoMADS(poll_size_init=0.5).optimize(obj, random_seed=0)
    """

    algorithm_str: str = "orthomads"
    algorithm_type: AlgorithmType = AlgorithmType.DIRECT_SEARCH

    def __init__(
        self,
        poll_size_init: float = 1.0,
        min_poll_size: float = 1e-9,
    ) -> None:
        """Initialise OrthoMADS.

        Parameters
        ----------
        poll_size_init:
            Initial poll-frame size.  Defaults to ``1.0``.
        min_poll_size:
            Frame-size convergence tolerance.  Defaults to ``1e-9``.
        """
        self.poll_size_init = poll_size_init
        self.min_poll_size = min_poll_size

    def optimize(
        self,
        problem_objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        poll_size_init: float | None = None,
        min_poll_size: float | None = None,
        opportunistic: bool = False,
        rich_direction: bool = False,
    ) -> None:
        """Run OrthoMADS optimization.

        Parameters
        ----------
        problem_objective:
            Pre-configured :class:`~dfbench.core.objective.Objective` instance.
        init_params:
            Starting point in **bounded** physical space.  If ``None``, a
            random point is sampled uniformly within the problem bounds.
        random_seed:
            Seed for reproducibility.  If ``None``, a random seed is
            generated via system entropy.
        poll_size_init:
            Override the initial poll-frame size set in ``__init__``.
        min_poll_size:
            Override the frame-size convergence tolerance set in ``__init__``.
        opportunistic:
            Stop evaluating a poll set as soon as a strict improvement is
            found.  Reduces per-iteration evaluations at the cost of less
            thorough exploration.  Default: ``False``.
        rich_direction:
            Bias the mesh update towards the last successful direction.
            Can accelerate convergence on smooth problems.  Default: ``False``.
        """
        obj = problem_objective
        problem = obj.problem

        random_seed, _ = self.prepare(obj, unbounded=False, random_seed=random_seed)

        # Resolve hyperparameter overrides
        _poll_size_init = poll_size_init if poll_size_init is not None else self.poll_size_init
        _min_poll_size = min_poll_size if min_poll_size is not None else self.min_poll_size

        # Resolve initial point
        if init_params is None:
            x0 = np.array(obj.random_params_bounded())
        else:
            x0 = np.array(init_params)

        lower = np.array(problem.bounds[0])
        upper = np.array(problem.bounds[1])

        # JIT warmup (does not log evaluations)
        obj.warmup_value()

        obj.start_logging()

        _run_omads_poll(
            obj=obj,
            init_params=x0,
            lower=lower,
            upper=upper,
            random_seed=random_seed,
            mesh_type="ORTHO",
            poll_size_init=_poll_size_init,
            min_poll_size=_min_poll_size,
            opportunistic=opportunistic,
            rich_direction=rich_direction,
        )
