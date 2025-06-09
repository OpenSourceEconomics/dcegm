# test_jax_sorted_interpolate.py

import jax.numpy as jnp
import numpy as np
from jax import jit

from dcegm.upper_envelope_fedor_jax import fast_interp_sorted_nan_safe


def test_fast_interp_sorted_nan_safe_basic():
    x = jnp.array([0.0, 1.0, 2.0, 3.0, jnp.nan, jnp.nan, jnp.nan, jnp.nan])
    y = jnp.array([0.0, 1.0, 4.0, 9.0, jnp.nan, jnp.nan, jnp.nan, jnp.nan])
    x_new = jnp.array([-1.0, 0.0, 0.5, 1.5, 2.0, 3.0, 4.0, jnp.nan, jnp.nan])

    seg_len = 4
    seg_len_new = 7

    expected = jnp.array(
        [
            -jnp.inf,  # -1.0 out of domain
            0.0,  # exact match
            0.5,  # linear interp
            2.5,  # linear interp
            4.0,  # exact match
            9.0,  # exact match
            -jnp.inf,  # out of domain
            -jnp.inf,  # invalid segment
            -jnp.inf,  # invalid segment
        ]
    )

    result = fast_interp_sorted_nan_safe(x, y, x_new, seg_len, seg_len_new)

    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_fast_interp_with_short_segments():
    # All input values are valid but the segment length is short
    x = jnp.array([0.0, 2.0, 4.0])
    y = jnp.array([0.0, 4.0, 16.0])
    x_new = jnp.array([1.0, 3.0, 5.0])

    seg_len = 2  # Only use first two of x/y
    seg_len_new = 3  # All new queries are "valid"

    expected = jnp.array(
        [
            2.0,  # interp between 0 and 2
            -jnp.inf,  # 3.0 outside known range
            -jnp.inf,  # 5.0 outside known range
        ]
    )

    result = fast_interp_sorted_nan_safe(x, y, x_new, seg_len, seg_len_new)

    np.testing.assert_array_equal(np.asarray(result), np.asarray(expected))


def test_fast_interp_trailing_invalids():
    # Test that once seg_len_new is reached, rest are -inf
    x = jnp.linspace(0, 1, 5)
    y = jnp.square(x)
    x_new = jnp.linspace(0, 1.2, 10)

    seg_len = 5
    seg_len_new = 6  # Only first 6 are valid

    result = fast_interp_sorted_nan_safe(x, y, x_new, seg_len, seg_len_new)

    # Manually compute expected
    expected = jnp.interp(x_new[:6], x, y, left=-jnp.inf, right=-jnp.inf)
    expected = jnp.concatenate([expected, jnp.full((4,), -jnp.inf)])

    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-5)


def test_fast_interp_jit():
    # Check if function works under JIT
    x = jnp.array([0.0, 1.0, 2.0])
    y = jnp.array([0.0, 1.0, 4.0])
    x_new = jnp.array([0.5, 1.5, 2.5])

    seg_len = 3
    seg_len_new = 3

    expected = jnp.array(
        [
            0.5,  # interp between 0 and 1
            2.5,  # interp between 1 and 2
            -jnp.inf,  # outside range
        ]
    )

    interp_jit = jit(fast_interp_sorted_nan_safe)
    result = interp_jit(x, y, x_new, seg_len, seg_len_new)

    np.testing.assert_allclose(np.asarray(result), np.asarray(expected), rtol=1e-5)
