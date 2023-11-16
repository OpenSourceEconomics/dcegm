import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.simulate import simulate_all_periods
from dcegm.simulate import simulate_single_period
from dcegm.solve import solve_dcegm
from jax import config
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)
from toy_models.consumption_retirement_model.state_space_objects import update_state
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_final_consume_all,
)
from toy_models.consumption_retirement_model.utility_functions import (
    utility_final_consume_all,
)

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
        "marginal_utility": marginal_utility,
        "inverse_marginal_utility": inverse_marginal_utility,
    }
    return out


@pytest.fixture(scope="module")
def utility_functions_final_period():
    """Return dict with utility functions for final period."""
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }


def test_simulate(
    state_space_functions, utility_functions, utility_functions_final_period
):
    num_agents = 100

    params = {}
    params["rho"] = 0.5
    params["delta"] = 0.5
    params["interest_rate"] = 0.02
    params["ltc_cost"] = 5
    params["wage_avg"] = 8
    params["sigma"] = 1
    params["lambda"] = 1
    params["beta"] = 0.95

    # exog params
    params["ltc_prob_constant"] = 0.3
    params["ltc_prob_age"] = 0.1
    params["job_offer_constant"] = 0.5
    params["job_offer_age"] = 0
    params["job_offer_educ"] = 0
    params["job_offer_type_two"] = 0.4

    options = {
        "model_params": {
            "n_choices": 2,
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
        50,
        WEALTH_GRID_POINTS,
    )

    (
        model_funcs,
        _compute_upper_envelope,
        get_state_specific_choice_set,
        update_endog_state_by_state_and_choice,
    ) = process_model_functions(
        options,
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_final_period=utility_functions_final_period,
        budget_constraint=budget_dcegm_two_exog_processes,
    )

    # === Solve ===

    (
        period_specific_state_objects,
        state_space,
        map_state_choice_to_index,
        exog_state_mapping,
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
        state_space_functions=state_space_functions,
        utility_functions=utility_functions,
        utility_final_period=utility_functions_final_period,
        budget_constraint=budget_dcegm_two_exog_processes,
    )

    # === Simulate ===

    # compute_utility_final_period = determine_function_arguments_and_partial_options(
    #     compute_utility_consume_everything, options=options
    # )

    initial_states = {
        "period": np.zeros(num_agents, dtype=np.int16),
        "lagged_choice": np.zeros(num_agents, dtype=np.int16) + 1,
        "married": np.zeros(num_agents, dtype=np.int16),
        "ltc": np.zeros(num_agents, dtype=np.int16),
        "job_offer": np.zeros(num_agents, dtype=np.int16),
    }
    wealth_initial = np.ones(num_agents) * 10

    states_and_wealth_beginning_of_period_zero = (initial_states, wealth_initial)

    _carry, _result = simulate_single_period(
        states_and_wealth_beginning_of_period=states_and_wealth_beginning_of_period_zero,
        period=0,
        params=params,
        basic_seed=111,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_left_solved=policy_left,
        policy_right_solved=policy_right,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        choice_range=jnp.arange(map_state_choice_to_index.shape[-1]),
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
        compute_utility=model_funcs["compute_utility"],
        compute_beginning_of_period_wealth=model_funcs[
            "compute_beginning_of_period_wealth"
        ],
        exog_state_mapping=exog_state_mapping,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )

    sim_dict = simulate_all_periods(
        states_period_zero=initial_states,
        wealth_period_zero=wealth_initial,
        num_periods=options["state_space"]["n_periods"],
        params=params,
        seed=111,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_left_solved=policy_left,
        policy_right_solved=policy_right,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        choice_range=jnp.arange(map_state_choice_to_index.shape[-1], dtype=jnp.int16),
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
        compute_utility=model_funcs["compute_utility"],
        compute_beginning_of_period_wealth=model_funcs[
            "compute_beginning_of_period_wealth"
        ],
        exog_state_mapping=exog_state_mapping,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )

    df = create_data_frame(sim_dict)

    period = 0
    choice = 1  # everyone chooses 1 (i.e. retires in the final period)

    value_period_zero = df.xs(period, level=0)["utility"] + params["beta"] * (
        df.xs(period + 1, level=0)["utility"]
    )
    expected = (
        df.xs(period, level=0)["value"]
        - df.xs(period, level=0)[f"taste_shock_{choice}"]
    )

    # utility_0(states_0, wealth_0) + beta * utility_1(state_agent_1, wealth_agent_1)
    # = value_0(states_0, wealth_0)
    aaae(value_period_zero.mean(), expected.mean(), decimal=2)
    # not exactly the same because of income shocks?


def create_data_frame(sim_dict):
    n_periods, n_agents, n_choices = sim_dict["taste_shocks"].shape

    keys_to_drop = ["taste_shocks", "period"]
    dict_to_df = {key: sim_dict[key] for key in sim_dict if key not in keys_to_drop}

    df_without_taste_shocks = pd.DataFrame(
        {key: val.ravel() for key, val in dict_to_df.items()},
        index=pd.MultiIndex.from_product(
            [np.arange(n_periods), np.arange(n_agents)],
            names=["period", "agent"],
        ),
    )

    taste_shocks = sim_dict["taste_shocks"]
    df_taste_shocks = pd.DataFrame(
        {
            f"taste_shock_{choice}": taste_shocks[..., choice].flatten()
            for choice in range(n_choices)
        },
        index=pd.MultiIndex.from_product(
            [np.arange(n_periods), np.arange(n_agents)],
            names=["period", "agent"],
        ),
    )

    df_combined = pd.concat([df_without_taste_shocks, df_taste_shocks], axis=1)
    _df_combined = df_without_taste_shocks.join(df_taste_shocks)

    return df_combined
