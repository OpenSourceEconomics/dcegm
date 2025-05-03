import numpy as np
from jax import numpy as jnp

from tests.test_models.exog_ltc_model import prob_exog_ltc


def prob_exog_job_offer(
    job_offer,
    params,
):
    prob_job_offer = (job_offer == 0) * params["job_offer_constant"] + (
        job_offer == 1
    ) * (params["job_offer_constant"] + params["job_offer_type_two"])
    prob_no_job_offer = 1 - prob_job_offer

    return jnp.array([prob_no_job_offer, prob_job_offer])


PARAMS = {
    "rho": 0.5,
    "delta": 0.5,
    "interest_rate": 0.02,
    "ltc_cost": 5,
    "wage_avg": 8,
    "sigma": 1,
    "lambda": 1,
    "ltc_prob": 0.3,
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
            "job_offer": {"transition": prob_exog_job_offer, "states": [0, 1]},
        },
    },
}


def budget_dcegm_exog_ltc_and_job_offer(
    lagged_choice,
    ltc,
    savings_end_of_previous_period,
    income_shock_previous_period,
    params,
):
    # lagged_job_offer = jnp.abs(state[-1] - 2) * (state[-1] > 0) * state[0]  # [1, 3]
    ltc_patient = ltc == 1  # [2, 3]

    resource = (
        (1 + params["interest_rate"]) * savings_end_of_previous_period
        + (params["wage_avg"] + income_shock_previous_period)
        * (1 - lagged_choice)  # if worked last period
        - ltc_patient * params["ltc_cost"]
    )
    return jnp.maximum(resource, 0.5)
