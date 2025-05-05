import numpy as np
from jax import numpy as jnp


def prob_exog_ltc(
    period,
    ltc,
    params,
):
    prob_ltc = (ltc == 0) * (
        params["ltc_prob_constant"] + period * params["ltc_prob_age"]
    ) + (ltc == 1)
    prob_no_ltc = 1 - prob_ltc

    return jnp.array([prob_no_ltc, prob_ltc])


PARAMS = {
    "rho": 0.5,
    "delta": 0.5,
    "interest_rate": 0.02,
    "ltc_cost": 5,
    "wage_avg": 8,
    "sigma": 1,
    "taste_shock_scale": 10,
    "beta": 0.95,
    # Exogenous parameters
    "ltc_prob_constant": 0.3,
    "ltc_prob_age": 0.1,
    "job_offer_constant": 0.5,
    "job_offer_age": 0,
    "job_offer_educ": 0,
    "job_offer_type_two": 0.4,
}

OPTIONS = {
    "model_params": {
        "n_quad_points_stochastic": 5,
        "n_choices": 2,
    },
    "state_space": {
        "n_periods": 2,
        "choices": np.arange(2),
        "endogenous_states": {
            "married": [0, 1],
        },
        "continuous_states": {
            "wealth": np.linspace(0, 50, 100),
        },
        "exogenous_processes": {
            "ltc": {"transition": prob_exog_ltc, "states": [0, 1]},
        },
    },
}


def budget_dcegm_exog_ltc(
    ltc,
    lagged_choice,
    savings_end_of_previous_period,
    income_shock_previous_period,
    params,
):
    resource = (
        (1 + params["interest_rate"]) * savings_end_of_previous_period
        + (params["wage_avg"] + income_shock_previous_period)
        * (1 - lagged_choice)  # if worked last period
        - ltc * params["ltc_cost"]
    )
    return jnp.maximum(resource, 0.5)
