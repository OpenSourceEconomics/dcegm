import jax
import numpy as np
import pytest

from dcegm.pre_processing.setup_model import setup_model
from dcegm.solve import get_solve_func_for_model
from toy_models.load_example_model import load_example_models


def utility_crra(
    consumption,
    choice,
    married,
    params,
):

    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])

    utility = (
        utility_consumption
        + married * params["married_util"]
        - (1 - choice) * params["delta"]
    )

    return utility


@pytest.fixture
def state_space_options():
    state_space_options_sol = {
        "state_space": {
            "n_periods": 5,
            "choices": np.arange(2),
            "endogenous_states": {
                "married": np.arange(2, dtype=int),
            },
            "continuous_states": {
                "wealth": np.arange(0, 100, 5, dtype=float),
            },
        },
        "model_params": {"quadrature_points_stochastic": 5, "min_age": 18},
    }
    state_space_options = {"solution": state_space_options_sol}

    return state_space_options


def test_sim_and_sol_model(state_space_options, load_example_model):
    params, model_params = load_example_model("retirement_taste_shocks")
    params["married_util"] = 0.5

    model_funcs = load_example_models("dcegm_paper")
    utility_functions = model_funcs["utility_functions"]
    utility_functions["utility"] = utility_crra

    options_sol = {
        "state_space": state_space_options["solution"]["state_space"],
        "model_params": model_params,
    }

    model_sol = setup_model(
        options=options_sol,
        state_space_functions=model_funcs["state_space_functions"],
        utility_functions=utility_functions,
        utility_functions_final_period=model_funcs["final_period_utility_functions"],
        budget_constraint=model_funcs["budget_constraint"],
    )
    solve_func = get_solve_func_for_model(model_sol)

    value, policy, endog_grid = solve_func(params)
    breakpoint()
