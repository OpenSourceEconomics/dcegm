import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dcegm.pre_processing.shared import (
    determine_function_arguments_and_partial_model_specs,
)

# Shared params
PARAMS = {
    "discount_factor": 0.95,
    "rho": 0.5,
    "delta": -1,
    "interest_rate": 0.05,
    "consumption_floor": 100,
    "pension": 1000,
    "labor_income": 2000,
}


def utility_crra(consumption, choice, params, continuous_state=None):
    rho = params["rho"]
    util_c = jnp.where(
        jnp.allclose(rho, 1.0),
        jnp.log(consumption),
        (consumption ** (1 - rho) - 1) / (1 - rho),
    )
    delta = params.get("delta", 0.0)
    return util_c - (1 - choice) * delta


def get_compute_utility():
    return determine_function_arguments_and_partial_model_specs(
        utility_crra,
        model_specs={},
        continuous_state_name="continuous_state",
    )


def mask_and_assert_allclose(left, right, rtol, atol, skip_msg):
    mask = ~np.isnan(right)
    if np.sum(mask) == 0:
        pytest.skip(skip_msg)
    np.testing.assert_allclose(
        np.asarray(left)[mask], right[mask], rtol=rtol, atol=atol
    )
