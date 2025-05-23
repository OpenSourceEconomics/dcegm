import numpy as np

from dcegm.toy_models.cons_ret_model_stochastic_ltc.ltc import prob_exog_ltc
from dcegm.toy_models.cons_ret_model_stochastic_ltc_and_job_offer.ltc_and_job_offer import (
    prob_exog_job_offer,
)


def example_params():
    params = {
        "rho": 0.5,
        "delta": 0.5,
        "interest_rate": 0.02,
        "ltc_cost": 5,
        "wage_avg": 8,
        "income_shock_std": 1,
        "income_shock_mean": 0.0,
        "taste_shock_scale": 1,
        "ltc_prob": 0.3,
        "discount_factor": 0.95,
        # Exogenous parameters
        "ltc_prob_constant": 0.3,
        "ltc_prob_age": 0.1,
        "job_offer_constant": 0.5,
        "job_offer_age": 0,
        "job_offer_educ": 0,
        "job_offer_type_two": 0.4,
    }
    return params


def example_model_config():
    model_config = {
        "n_periods": 2,
        "choices": [0, 1],
        "continuous_states": {
            "assets_end_of_period": np.linspace(0, 50, 100),
        },
        "deterministic_states": {
            "married": [0, 1],
        },
        "stochastic_states": {
            "ltc": [0, 1],
            "job_offer": [0, 1],
        },
        "n_quad_points": 5,
    }
    return model_config


def example_model_specs():
    model_specs = {
        "n_choices": 2,
    }
    return model_specs
