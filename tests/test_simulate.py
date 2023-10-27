import jax.numpy as jnp
import numpy as np
import pytest
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.simulate import simulate_all_periods
from dcegm.simulate import simulate_single_period
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
        utility_functions=utility_functions,
        budget_constraint=budget_dcegm_two_exog_processes,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
    )

    initial_states = {
        "period": np.zeros(num_agents, dtype=np.int16),
        "lagged_choice": np.zeros(num_agents, dtype=np.int16) + 1,
        "married": np.zeros(num_agents, dtype=np.int16),
        "ltc": np.zeros(num_agents, dtype=np.int16),
        "job_offer": np.zeros(num_agents, dtype=np.int16),
    }
    wealth_initial = np.ones(num_agents) * 10

    states_and_wealth_beginning_of_period_zero = (initial_states, wealth_initial)

    simulate_single_period(
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
        compute_exog_transition_vec=_model_funcs["compute_exog_transition_vec"],
        compute_utility=_model_funcs["compute_utility"],
        compute_beginning_of_period_wealth=_model_funcs[
            "compute_beginning_of_period_wealth"
        ],
        exog_state_mapping=exog_state_mapping,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )

    simulate_all_periods(
        states_period_0=initial_states,
        wealth_period_0=wealth_initial,
        num_periods=options["state_space"]["n_periods"],
        params=params,
        seed=111,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_left_solved=policy_left,
        policy_right_solved=policy_right,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        choice_range=jnp.arange(map_state_choice_to_index.shape[-1], dtype=jnp.int16),
        compute_exog_transition_vec=_model_funcs["compute_exog_transition_vec"],
        compute_utility=_model_funcs["compute_utility"],
        compute_beginning_of_period_wealth=_model_funcs[
            "compute_beginning_of_period_wealth"
        ],
        exog_state_mapping=exog_state_mapping,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )
    # utility_0(states_0, wealth_0) + beta * utility_1(state_agent_1, wealth_agent_1)
    # = value_0(states_0, wealth_0)
