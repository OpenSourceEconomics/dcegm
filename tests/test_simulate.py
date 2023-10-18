import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.solve import solve_dcegm
from jax import config
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period_scalar,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)
from toy_models.consumption_retirement_model.state_space_objects import update_state

from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
    budget_dcegm_two_exog_processes,
)
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import flow_util
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
    func_exog_job_offer,
)
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import func_exog_ltc
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
    inverse_marginal_utility,
)
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
    marginal_utility,
)


config.update("jax_enable_x64", True)


WEALTH_GRID_POINTS = 100


@pytest.fixture(scope="module")
def state_space_functions():
    out = {
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
        "update_endog_state_by_state_and_choice": update_state,
    }
    return out


@pytest.fixture(scope="module")
def utility_functions():
    out = {
        "utility": flow_util,
        "inverse_marginal_utility": inverse_marginal_utility,
        "marginal_utility": marginal_utility,
    }
    return out


def test_simulate(
    utility_functions,
    state_space_functions,
    load_example_model,
):
    num_agents = 100

    index = pd.MultiIndex.from_tuples(
        [("utility_function", "rho"), ("utility_function", "delta")],
        names=["category", "name"],
    )
    params = pd.DataFrame(data=[0.5, 0.5], columns=["value"], index=index)
    params.loc[("assets", "interest_rate"), "value"] = 0.02
    params.loc[("assets", "ltc_cost"), "value"] = 5
    params.loc[("wage", "wage_avg"), "value"] = 8
    params.loc[("shocks", "sigma"), "value"] = 1
    params.loc[("shocks", "lambda"), "value"] = 1
    params.loc[("transition", "ltc_prob"), "value"] = 0.3
    params.loc[("beta", "beta"), "value"] = 0.95

    # exog params
    params.loc[("ltc_prob_constant", "ltc_prob_constant"), "value"] = 0.3
    params.loc[("ltc_prob_age", "ltc_prob_age"), "value"] = 0.1
    params.loc[("job_offer_constant", "job_offer_constant"), "value"] = 0.5
    params.loc[("job_offer_age", "job_offer_age"), "value"] = 0
    params.loc[("job_offer_educ", "job_offer_educ"), "value"] = 0
    params.loc[("job_offer_type_two", "job_offer_type_two"), "value"] = 0.4

    options = {
        "model_params": {
            "n_grid_points": WEALTH_GRID_POINTS,
            "n_choices": 2,
            "max_wealth": 50,
            "quadrature_points_stochastic": 5,
        },
        "state_space": {
            "n_periods": 2,
            "choices": np.arange(2),
            "endogenous_states": {
                "married": [0, 1],
            },
            "exogenous_processes": {
                "ltc": {"transition": func_exog_ltc, "states": [0, 1]},
                "job_offer": {"transition": func_exog_job_offer, "states": [0, 1]},
            },
        },
    }

    exog_savings_grid = jnp.linspace(
        0,
        options["model_params"]["max_wealth"],
        options["model_params"]["n_grid_points"],
    )

    (
        _model_funcs,
        _compute_upper_envelope,
        get_state_specific_choice_set,
        update_endog_state_by_state_and_choice,
    ) = process_model_functions(
        options,
        user_utility_functions=utility_functions,
        user_budget_constraint=budget_dcegm_two_exog_processes,
        user_final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
    )

    (
        period_specific_state_objects,
        state_space,
        map_state_choice_to_index,
    ) = create_state_space_and_choice_objects(
        options=options,
        get_state_specific_choice_set=get_state_specific_choice_set,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )

    (
        value,
        policy_left,
        policy_right,
        endog_grid,
    ) = solve_dcegm(
        params,
        options,
        exog_savings_grid=exog_savings_grid,
        utility_functions=utility_functions,
        budget_constraint=budget_dcegm_two_exog_processes,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
    )

    initial_states = {
        "period": np.zeros(num_agents, dtype=np.int16),
        "lagged_choice": np.zeros(num_agents, dtype=np.int16),
        "married": np.zeros(num_agents, dtype=np.int16),
        "ltc": np.zeros(num_agents, dtype=np.int16),
        "job_offer": np.zeros(num_agents, dtype=np.int16),
    }

    map_state_choice_to_index[
        tuple((initial_states[key],) for key in initial_states.keys())
    ]
    # breakpoint()
