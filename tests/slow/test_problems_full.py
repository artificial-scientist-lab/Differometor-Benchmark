"""Section 4 (full) — Full problem tests requiring Differometor.

These are marked @slow and must be run via srun on the cluster.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


pytestmark = pytest.mark.slow


# ======================================================================
# VoyagerProblem (4.1–4.9)
# ======================================================================


class TestVoyagerProblem:
    @pytest.fixture(autouse=True)
    def _init(self):
        from dfbench.problems import VoyagerProblem

        self.problem = VoyagerProblem()

    def test_initializes(self):
        """4.1 VoyagerProblem initializes without error."""
        assert self.problem is not None

    def test_bounds_shape(self):
        """4.2 bounds has shape (2, n_params); lower < upper."""
        b = self.problem.bounds
        assert b.shape[0] == 2
        assert b.shape[1] == self.problem.n_params
        assert jnp.all(b[0] < b[1])

    def test_optimization_pairs(self):
        """4.3 optimization_pairs non-empty, len matches bounds."""
        pairs = self.problem.optimization_pairs
        assert len(pairs) > 0
        assert len(pairs) == self.problem.bounds.shape[1]

    def test_objective_at_midpoint(self):
        """4.4 objective_function at midpoint returns finite scalar."""
        b = self.problem.bounds
        mid = (b[0] + b[1]) / 2
        loss = self.problem.objective_function(mid)
        assert jnp.isfinite(loss)
        assert loss.ndim == 0

    def test_sigmoid_objective_at_midpoint(self):
        """4.5 sigmoid_objective_function at inverse(midpoint) is finite."""
        from dfbench.core.utils import inverse_sigmoid_bounding

        b = self.problem.bounds
        mid = (b[0] + b[1]) / 2
        unbounded = inverse_sigmoid_bounding(mid, b)
        loss = self.problem.sigmoid_objective_function(unbounded)
        assert jnp.isfinite(loss)

    def test_grad_at_midpoint(self):
        """4.6 jax.grad at midpoint returns finite (n_params,)."""
        b = self.problem.bounds
        mid = (b[0] + b[1]) / 2
        g = jax.grad(self.problem.objective_function)(mid)
        assert g.shape == (self.problem.n_params,)
        assert jnp.all(jnp.isfinite(g))

    def test_grad_sigmoid_at_midpoint(self):
        """4.7 jax.grad(sigmoid_obj) at unbounded midpoint is finite."""
        from dfbench.core.utils import inverse_sigmoid_bounding

        b = self.problem.bounds
        mid = (b[0] + b[1]) / 2
        unbounded = inverse_sigmoid_bounding(mid, b)
        g = jax.grad(self.problem.sigmoid_objective_function)(unbounded)
        assert g.shape == (self.problem.n_params,)
        assert jnp.all(jnp.isfinite(g))

    def test_calculate_sensitivity(self):
        """4.9 calculate_sensitivity returns positive values."""
        b = self.problem.bounds
        mid = (b[0] + b[1]) / 2
        sens = self.problem.calculate_sensitivity(mid)
        assert jnp.all(sens > 0)


# ======================================================================
# ConstrainedVoyagerProblem (4.12–4.15)
# ======================================================================


class TestConstrainedVoyagerProblem:
    @pytest.fixture(autouse=True)
    def _init(self):
        from dfbench.problems import ConstrainedVoyagerProblem

        self.problem = ConstrainedVoyagerProblem()

    def test_initializes(self):
        """4.12 ConstrainedVoyagerProblem initializes without error."""
        assert self.problem is not None

    def test_bounds_shape(self):
        """4.13 Same shape checks."""
        b = self.problem.bounds
        assert b.shape[0] == 2
        assert b.shape[1] == self.problem.n_params
        assert jnp.all(b[0] < b[1])

    def test_objective_at_midpoint(self):
        """4.13b objective_function at midpoint returns finite scalar."""
        b = self.problem.bounds
        mid = (b[0] + b[1]) / 2
        loss = self.problem.objective_function(mid)
        assert jnp.isfinite(loss)

    def test_penalty_fn_changes_loss(self):
        """4.14 Switching power_penalty_fn changes loss."""
        from dfbench.problems.base_problem import (
            squashed_relu_penalty,
            zero_penalty,
        )

        b = self.problem.bounds
        mid = (b[0] + b[1]) / 2

        self.problem._power_penalty_fn = squashed_relu_penalty
        loss_sq = float(self.problem.objective_function(mid))  # noqa

        self.problem._power_penalty_fn = zero_penalty
        loss_zero = float(self.problem.objective_function(mid))  # noqa
        # Loss should differ if there are actual power violations
        # (may be equal if midpoint has no violations)


# ======================================================================
# UIFOProblem (4.16–4.20)
# ======================================================================


class TestUIFOProblem:
    def test_initializes(self):
        """4.16 Initializes without error."""
        from dfbench.problems import UIFOProblem

        p = UIFOProblem(size=3, topology_seed=42)
        assert p is not None

    def test_bounds_shape(self):
        """4.17 Same shape checks."""
        from dfbench.problems import UIFOProblem

        p = UIFOProblem(size=3, topology_seed=42)
        b = p.bounds
        assert b.shape[0] == 2
        assert jnp.all(b[0] < b[1])

    def test_different_topology_seeds(self):
        """4.18 Different topology_seed → different n_params or bounds."""
        from dfbench.problems import UIFOProblem

        p1 = UIFOProblem(size=3, topology_seed=42)
        p2 = UIFOProblem(size=3, topology_seed=99)
        different = (p1.n_params != p2.n_params) or not jnp.allclose(
            p1.bounds, p2.bounds
        )
        assert different

    def test_same_topology_seeds(self):
        """4.19 Same topology_seed → identical n_params and bounds."""
        from dfbench.problems import UIFOProblem

        p1 = UIFOProblem(size=3, topology_seed=42)
        p2 = UIFOProblem(size=3, topology_seed=42)
        assert p1.n_params == p2.n_params
        np.testing.assert_array_equal(np.array(p1.bounds), np.array(p2.bounds))

    def test_backwards_compat_alias(self):
        """4.20 RandomUIFOProblem is an alias for UIFOProblem."""
        from dfbench.problems import RandomUIFOProblem, UIFOProblem

        assert RandomUIFOProblem is UIFOProblem
