import numpy as np

from dcegm.toy_models.cons_ret_model_exog_ltc.ltc import prob_exog_ltc


def example_params():
    params = {
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
    return params


def example_options():
    options = {
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
    return options
