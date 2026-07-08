"""Section 1: Utilities (dfbench.core.utils).

Tests 1.1-1.6: t2j, j2t, inverse_sigmoid_bounding.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from dfbench.core.utils import t2j, j2t, inverse_sigmoid_bounding


class TestT2J:
    """1.1 t2j: torch -> JAX round-trip."""

    def test_preserves_values_and_shape(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        j = t2j(t)
        assert j.shape == (3,)
        np.testing.assert_allclose(np.array(j), t.numpy(), atol=1e-6)

    def test_2d(self):
        t = torch.randn(4, 5)
        j = t2j(t)
        assert j.shape == (4, 5)
        np.testing.assert_allclose(np.array(j), t.numpy(), atol=1e-6)


class TestJ2T:
    """1.2-1.3 j2t: JAX -> torch round-trip, writable tensor."""

    def test_preserves_values_and_shape(self):
        j = jnp.array([1.0, 2.0, 3.0])
        t = j2t(j)
        assert t.shape == (3,)
        np.testing.assert_allclose(t.numpy(), np.array(j), atol=1e-6)

    def test_writable_tensor(self):
        """1.3 Result must be writable (no read-only np buffer)."""
        j = jnp.array([1.0, 2.0])
        t = j2t(j)
        # This would warn/error on a read-only buffer
        t[0] = 99.0
        assert t[0].item() == 99.0


class TestRoundTrips:
    """1.4 t2j ∘ j2t and j2t ∘ t2j are identity."""

    def test_t2j_j2t_identity(self):
        original = torch.randn(10)
        result = j2t(t2j(original))
        np.testing.assert_allclose(result.numpy(), original.numpy(), atol=1e-6)

    def test_j2t_t2j_identity(self):
        original = jnp.array(np.random.randn(10).astype(np.float32))
        result = t2j(j2t(original))
        np.testing.assert_allclose(np.array(result), np.array(original), atol=1e-6)


class TestInverseSigmoidBounding:
    """1.5-1.6 inverse_sigmoid_bounding."""

    @pytest.fixture
    def bounds(self):
        return jnp.array([[-5.0, 0.0], [5.0, 10.0]])  # shape (2, 2)

    def test_round_trip(self, bounds):
        """1.5 inverse is exact inverse of sigmoid_bounding."""
        from differometor.utils import sigmoid_bounding

        # Pick interior points (not at boundary)
        bounded = jnp.array([-2.0, 3.0])
        unbounded = inverse_sigmoid_bounding(bounded, bounds)
        recovered = sigmoid_bounding(unbounded, bounds)
        np.testing.assert_allclose(np.array(recovered), np.array(bounded), atol=1e-5)

    def test_multiple_interior_points(self, bounds):
        """Round-trip for multiple interior points."""
        from differometor.utils import sigmoid_bounding

        key = jax.random.PRNGKey(0)
        bounded_pts = jax.random.uniform(
            key, shape=(50, 2), minval=bounds[0] + 0.1, maxval=bounds[1] - 0.1
        )
        unbounded = jax.vmap(lambda p: inverse_sigmoid_bounding(p, bounds))(bounded_pts)
        recovered = jax.vmap(lambda p: sigmoid_bounding(p, bounds))(unbounded)
        np.testing.assert_allclose(
            np.array(recovered), np.array(bounded_pts), atol=1e-5
        )

    def test_no_nan_or_inf(self, bounds):
        """1.6 Clips extreme values; no NaN/inf."""
        # Values at/outside boundary
        extreme = jnp.array([-5.0, 10.0])  # exactly at bounds
        result = inverse_sigmoid_bounding(extreme, bounds)
        assert jnp.all(jnp.isfinite(result)), f"Got non-finite: {result}"

    def test_beyond_bounds_no_nan(self, bounds):
        """Values beyond bounds are clipped, not NaN."""
        beyond = jnp.array([-100.0, 100.0])
        result = inverse_sigmoid_bounding(beyond, bounds)
        assert jnp.all(jnp.isfinite(result))
