"""Section 5 (randomness) — Objective seed & random sampling.

Tests 5.5–5.12: set_seed, random_params_bounded, random_params_unbounded.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dfbench.core.objective import Objective
from differometor.utils import sigmoid_bounding


class TestRandomParamsBoundedShape:
    """5.5–5.6 Shape of random_params_bounded."""

    def test_single_sample_shape(self, seeded_obj):
        """5.5 n=1 → (n_params,)."""
        sample = seeded_obj.random_params_bounded(n_samples=1)
        assert sample.shape == (seeded_obj.n_params,)

    def test_batch_sample_shape(self, seeded_obj):
        """5.6 n=10 → (10, n_params)."""
        samples = seeded_obj.random_params_bounded(n_samples=10)
        assert samples.shape == (10, seeded_obj.n_params)


class TestRandomParamsBoundedBounds:
    """5.7 All samples within bounds."""

    def test_within_bounds(self, seeded_obj):
        samples = seeded_obj.random_params_bounded(n_samples=100)
        lower, upper = seeded_obj.bounds[0], seeded_obj.bounds[1]
        assert jnp.all(samples >= lower), "Samples below lower bound"
        assert jnp.all(samples <= upper), "Samples above upper bound"


class TestRandomParamsUnbounded:
    """5.8–5.9 Unbounded sampling."""

    def test_finite(self, seeded_obj):
        """5.8 All values non-NaN, non-inf."""
        samples = seeded_obj.random_params_unbounded(n_samples=100)
        assert jnp.all(jnp.isfinite(samples))

    def test_sigmoid_round_trip(self, seeded_obj):
        """5.9 sigmoid_bounding recovers in-bounds points."""
        samples = seeded_obj.random_params_unbounded(n_samples=100)
        bounds = seeded_obj.problem.bounds
        bounded = jax.vmap(lambda x: sigmoid_bounding(x, bounds))(samples)
        lower, upper = bounds[0], bounds[1]
        assert jnp.all(bounded >= lower - 1e-6)
        assert jnp.all(bounded <= upper + 1e-6)


class TestSeedReproducibility:
    """5.10–5.11b Seed-based reproducibility."""

    def test_same_seed_same_output(self, mock_problem):
        """5.10 Two separate instances, same seed → identical output."""
        obj1 = Objective(mock_problem)
        obj1.set_seed(42)

        obj2 = Objective(mock_problem)
        obj2.set_seed(42)

        b1 = obj1.random_params_bounded(n_samples=5)
        b2 = obj2.random_params_bounded(n_samples=5)
        np.testing.assert_allclose(np.array(b1), np.array(b2), rtol=1e-5)

        u1 = obj1.random_params_unbounded(n_samples=5)
        u2 = obj2.random_params_unbounded(n_samples=5)
        np.testing.assert_allclose(np.array(u1), np.array(u2), rtol=1e-5)

    def test_successive_calls_differ(self, seeded_obj):
        """5.11 Successive calls produce different samples."""
        s1 = seeded_obj.random_params_bounded(n_samples=5)
        s2 = seeded_obj.random_params_bounded(n_samples=5)
        assert not jnp.allclose(s1, s2), "Successive samples should differ"

    def test_different_seeds_differ(self, mock_problem):
        """5.11b Different seeds → different samples."""
        obj1 = Objective(mock_problem)
        obj1.set_seed(42)

        obj2 = Objective(mock_problem)
        obj2.set_seed(99)

        s1 = obj1.random_params_bounded(n_samples=5)
        s2 = obj2.random_params_bounded(n_samples=5)
        assert not jnp.allclose(s1, s2), "Different seeds should differ"


class TestExplicitRngKey:
    """5.12 Explicit rng_key overrides internal state."""

    def test_explicit_key_no_consume(self, seeded_obj):
        """Explicit key does not consume internal key."""
        # Draw one sample to set up state
        _ = seeded_obj.random_params_bounded(n_samples=1)

        # Get the internal key state before explicit-key call
        key_before = seeded_obj._rng_key

        # Call with explicit key
        explicit_key = jax.random.PRNGKey(999)
        _ = seeded_obj.random_params_bounded(n_samples=1, rng_key=explicit_key)

        # Internal key should not have been consumed
        key_after = seeded_obj._rng_key
        assert jnp.array_equal(key_before, key_after), (
            "Explicit rng_key should not consume internal key"
        )

    def test_explicit_key_overrides(self, mock_problem):
        """Explicit key produces different result than internal key."""
        obj = Objective(mock_problem)
        obj.set_seed(42)
        internal_sample = obj.random_params_bounded(n_samples=3)

        obj.set_seed(42)
        explicit_key = jax.random.PRNGKey(9999)
        explicit_sample = obj.random_params_bounded(n_samples=3, rng_key=explicit_key)

        # The explicit key result should not match the internal-key result
        assert not jnp.allclose(internal_sample, explicit_sample)
