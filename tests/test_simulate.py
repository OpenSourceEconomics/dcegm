import jax.numpy as jnp
import numpy as np
import pytest
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.simulation.simulate import simulate_single_period
from dcegm.solve import solve_dcegm
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

from tests.two_period_models.exog_ltc.model_functions import budget_dcegm
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import flow_util
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import func_exog_ltc
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
    inverse_marginal_utility,
)
from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
    marginal_utility,
)

# from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
#     budget_dcegm_two_exog_processes,
# )
# from tests.two_period_models.exog_ltc_and_job_offer.model_functions import (
#     func_exog_job_offer,
# )

WEALTH_GRID_POINTS = 100


@pytest.fixture(scope="module")
def state_space_functions():
    """Return dict with state space functions."""
    out = {
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
        "update_endog_state_by_state_and_choice": update_state,
    }
    return out


@pytest.fixture(scope="module")
def utility_functions():
    """Return dict with utility functions."""
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
    n_agents = 100_000

    params = {}
    params["rho"] = 0.5
    params["delta"] = 0.5 * 10
    params["interest_rate"] = 0.02
    params["ltc_cost"] = 5
    params["wage_avg"] = 8
    params["sigma"] = 1
    params["lambda"] = 1e-16
    params["lambda"] = 10
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
                # "job_offer": {"transition": func_exog_job_offer, "states": [0, 1]},
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
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_dcegm,
    )

    # === Solve ===
    (
        _period_specific_state_objects,
        _state_space,
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
        utility_functions_final_period=utility_functions_final_period,
        budget_constraint=budget_dcegm,
    )

    # === Simulate ===

    initial_states = {
        "period": np.zeros(n_agents, dtype=np.int16),
        "lagged_choice": np.zeros(
            n_agents, dtype=np.int16
        ),  # all agents start as workers
        "married": np.zeros(n_agents, dtype=np.int16),
        "ltc": np.zeros(n_agents, dtype=np.int16),
        # "job_offer": np.ones(n_agents, dtype=np.int16),
        # "job_offer": np.zeros(n_agents, dtype=np.int16),
    }

    resources_initial = np.ones(n_agents) * 10
    states_and_wealth_beginning_of_period_zero = (initial_states, resources_initial)

    _carry, _result = simulate_single_period(
        states_and_resources_beginning_of_period=states_and_wealth_beginning_of_period_zero,
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
        compute_beginning_of_period_resources=model_funcs[
            "compute_beginning_of_period_resources"
        ],
        exog_state_mapping=exog_state_mapping,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
    )

    sim_dict = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=resources_initial,
        n_periods=options["state_space"]["n_periods"],
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
        compute_beginning_of_period_resources=model_funcs[
            "compute_beginning_of_period_resources"
        ],
        exog_state_mapping=exog_state_mapping,
        update_endog_state_by_state_and_choice=update_endog_state_by_state_and_choice,
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )

    df = create_simulation_df(sim_dict)

    period = 0

    # absrobing retirement state
    # this should contain nobody
    sim_dict["choice"][
        :, (sim_dict["choice"][period] == 0) & (sim_dict["choice"][period + 1] == 1)
    ]

    _cond = [df["choice"] == 0, df["choice"] == 1]
    _val = [df["taste_shock_0"], df["taste_shock_1"]]
    df["taste_shock_selected_choice"] = np.select(_cond, _val)

    # taste_shocks_final = df.xs(period + 1, level=0).filter(like="taste_shock_")

    value_period_zero = (
        df.xs(period, level=0)["utility"].mean()
        + params["beta"] * df.xs(period + 1, level=0)["value"].mean()
    )
    expected = (
        df.xs(period, level=0)["value"]
        - df.xs(period, level=0)["taste_shock_selected_choice"]
    ).mean()

    _value_period_zero = df.xs(period, level=0)["utility"] + params["beta"] * (
        df.xs(period + 1, level=0)["value"]
    )
    _expected = df.xs(period, level=0)["value"]

    aaae(value_period_zero.mean(), expected.mean(), decimal=2)
    # aaae(_value_period_zero, _expected)
    # aaae(_value_period_zero.mean(), _expected.mean(), decimal=2)
