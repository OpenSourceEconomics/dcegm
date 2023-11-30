"""Tests for simulation of consumption-retirement model with exogenous processes."""
import jax
from functools import partial
import jax.numpy as jnp
import numpy as np
import pytest
from dcegm.pre_processing.model_functions import process_model_functions
from dcegm.pre_processing.state_space import create_state_space_and_choice_objects
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import simulate_all_periods
from dcegm.simulation.simulate import simulate_final_period
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
    n_agents = 1_000_000

    params = {}
    params["rho"] = 0.5
    params["delta"] = 0.5
    params["interest_rate"] = 0.02
    params["ltc_cost"] = 5
    params["wage_avg"] = 8
    params["sigma"] = 1
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

    options["state_space"]["n_periods"]
    choice_range = jnp.arange(options["model_params"]["n_choices"])

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
    }

    initial_resources = np.ones(n_agents) * 10
    initial_states_and_resources = initial_states, initial_resources

    seed = 111
    n_periods = 2
    #
    num_keys = len(initial_resources) + 2
    sim_specific_keys = jnp.array(
        [
            jax.random.split(jax.random.PRNGKey(seed + period), num=num_keys)
            for period in range(n_periods)
        ]
    )

    (
        _states_and_resources_beginning_of_final_period,
        _sim_dict_zero,
    ) = simulate_single_period(
        states_and_resources_beginning_of_period=initial_states_and_resources,
        params=params,
        sim_specific_keys=sim_specific_keys[0],
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
    )

    simulate_body = partial(
        simulate_single_period,
        params=params,
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
    )
    # states_and_resources_beginning_of_final_period, sim_dict_zero = jax.lax.scan(
    #     f=simulate_body,
    #     init=initial_states_and_resources,
    #     xs=sim_specific_keys[:-1],
    # )
    (
        states_and_resources_beginning_of_final_period,
        sim_dict_zero,
    ) = simulate_single_period(
        initial_states_and_resources,
        sim_specific_keys[0],
        params=params,
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
    )

    final_period_dict = simulate_final_period(
        states_and_resources_beginning_of_final_period,
        sim_specific_keys=sim_specific_keys[-1],
        params=params,
        choice_range=choice_range,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )

    result, sim_specific_keys_result = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=initial_resources,
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

    aaae(sim_specific_keys_result, sim_specific_keys, decimal=32)

    df = create_simulation_df(result)

    _cond = [df["choice"] == 0, df["choice"] == 1]
    _val = [df["taste_shock_0"], df["taste_shock_1"]]
    df["taste_shock_selected_choice"] = np.select(_cond, _val)

    value_period_zero = (
        df.xs(0, level=0)["utility"].mean()
        + params["beta"]
        * (
            df.xs(1, level=0)["utility"]
            + df.xs(1, level=0)["taste_shock_selected_choice"]
        ).mean()
    )
    expected = (
        df.xs(0, level=0)["value"].mean()
        - df.xs(0, level=0)["taste_shock_selected_choice"].mean()
    )

    _cond = [final_period_dict["choice"] == 0, final_period_dict["choice"] == 1]
    _val = [
        final_period_dict["taste_shocks"][0, :, 0],
        final_period_dict["taste_shocks"][0, :, 1],
    ]
    taste_shock_selected_choice_final = np.select(_cond, _val)

    _cond = [sim_dict_zero["choice"] == 0, sim_dict_zero["choice"] == 1]
    _val = [
        sim_dict_zero["taste_shocks"][:, 0],  # np.squeeze
        sim_dict_zero["taste_shocks"][:, 1],  # np.squeeze
    ]
    taste_shock_selected_zero = np.select(_cond, _val)

    _value_period_zero = (
        sim_dict_zero["utility"].mean()
        + params["beta"]
        * (final_period_dict["utility"] + taste_shock_selected_choice_final).mean()
    )
    _expected = sim_dict_zero["value"].mean() - taste_shock_selected_zero.mean()

    ids_violating_absorbing_retirement = df.query(
        "(period == 0 and choice == 1) and (period == 1 and choice == 0)"
    )
    assert len(ids_violating_absorbing_retirement) == 0

    aaae(expected, _expected)
    aaae(_value_period_zero.mean(), _expected.mean(), decimal=2)
    aaae(value_period_zero.mean(), expected.mean(), decimal=2)
    aaae(sim_dict_zero["utility"].mean(), df.xs(0, level=0)["utility"].mean())
    aaae(df.xs(1, level=0)["utility"].mean(), final_period_dict["utility"].mean())
    aaae(
        df.xs(1, level=0)["taste_shock_selected_choice"],
        taste_shock_selected_choice_final,
    )
    aaae(value_period_zero.mean(), _value_period_zero.mean())
    aaae(_value_period_zero.mean(), expected.mean(), decimal=2)
