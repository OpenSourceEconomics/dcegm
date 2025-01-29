"""Tests for simulation of consumption-retirement model with exogenous processes."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.pre_processing.shared import determine_function_arguments_and_partial_options
from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import (
    simulate_all_periods,
    simulate_final_period,
    simulate_single_period,
)
from toy_models.cons_ret_model_with_cont_exp.state_space_objects import (
    next_period_experience,
)


def _create_test_objects_from_df(df, params):
    _cond = [df["choice"] == 0, df["choice"] == 1]
    _val = [df["taste_shocks_0"], df["taste_shocks_1"]]
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
        df.xs(0, level=0)["value_max"].mean()
        - df.xs(0, level=0)["taste_shock_selected_choice"].mean()
    )

    return value_period_zero, expected


@pytest.fixture()
def model_setup(toy_model_exog_ltc):
    options = toy_model_exog_ltc["options"]

    seed = 111
    n_agents = 1_000
    n_periods = options["state_space"]["n_periods"]

    initial_states = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "married": np.zeros(n_agents),
        "ltc": np.zeros(n_agents),
    }
    initial_wealth = np.ones(n_agents) * 10
    initial_states_and_wealth = initial_states, initial_wealth

    n_keys = len(initial_wealth) + 2
    sim_specific_keys = jnp.array(
        [
            jax.random.split(jax.random.PRNGKey(seed + period), num=n_keys)
            for period in range(n_periods)
        ]
    )

    return {
        **toy_model_exog_ltc,
        "initial_states": initial_states,
        "initial_wealth": initial_wealth,
        "initial_states_and_wealth": initial_states_and_wealth,
        "sim_specific_keys": sim_specific_keys,
        "seed": seed,
    }


def test_simulate_lax_scan(model_setup):
    params = model_setup["params"]
    options = model_setup["options"]
    choice_range = options["state_space"]["choices"]
    model_structure = model_setup["model"]["model_structure"]
    model_funcs = model_setup["model"]["model_funcs"]

    discrete_states_names = model_structure["discrete_states_names"]
    map_state_choice_to_index = model_structure["map_state_choice_to_index_with_proxy"]

    exog_state_mapping = model_funcs["exog_state_mapping"]
    next_period_endogenous_state = model_funcs["next_period_endogenous_state"]

    value = model_setup["value"]
    policy = model_setup["policy"]
    endog_grid = model_setup["endog_grid"]

    states_initial = model_setup["initial_states"]
    wealth_initial = model_setup["initial_wealth"]

    sim_specific_keys = model_setup["sim_specific_keys"]

    state_space_dict = model_structure["state_space_dict"]
    states_initial = {
        key: value.astype(state_space_dict[key].dtype)
        for key, value in states_initial.items()
    }
    initial_states_and_wealth = states_initial, wealth_initial

    simulate_body = partial(
        simulate_single_period,
        params=params,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        model_funcs_sim=model_funcs,
        model_structure_sol=model_structure,
    )

    # a) lax.scan
    (
        lax_states_and_wealth_beginning_of_final_period,
        lax_sim_dict_zero,
    ) = jax.lax.scan(
        f=simulate_body,
        init=initial_states_and_wealth,
        xs=sim_specific_keys[:-1],
    )

    # b) single call
    (
        states_and_wealth_beginning_of_final_period,
        sim_dict_zero,
    ) = simulate_body(initial_states_and_wealth, sim_specific_keys=sim_specific_keys[0])

    lax_final_period_dict = simulate_final_period(
        lax_states_and_wealth_beginning_of_final_period,
        sim_specific_keys=sim_specific_keys[-1],
        params=params,
        discrete_states_names=discrete_states_names,
        choice_range=choice_range,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )
    final_period_dict = simulate_final_period(
        states_and_wealth_beginning_of_final_period,
        sim_specific_keys=sim_specific_keys[-1],
        params=params,
        discrete_states_names=discrete_states_names,
        choice_range=choice_range,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )

    aaae(np.squeeze(lax_sim_dict_zero["taste_shocks"]), sim_dict_zero["taste_shocks"])

    for key in final_period_dict.keys():
        aaae(
            lax_final_period_dict["taste_shocks"],
            final_period_dict["taste_shocks"],
        )
        aaae(lax_final_period_dict[key], final_period_dict[key])


def test_simulate(model_setup):
    params = model_setup["params"]
    options = model_setup["options"]

    value = model_setup["value"]
    policy = model_setup["policy"]
    endog_grid = model_setup["endog_grid"]

    n_agents = 100_000

    discrete_initial_states = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "married": np.zeros(n_agents),
        "ltc": np.zeros(n_agents),
    }
    wealth_initial = np.ones(n_agents) * 10

    result = simulate_all_periods(
        states_initial=discrete_initial_states,
        wealth_initial=wealth_initial,
        n_periods=options["state_space"]["n_periods"],
        params=params,
        seed=111,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        model=model_setup["model"],
    )

    df = create_simulation_df(result)

    value_period_zero, expected = _create_test_objects_from_df(df, params)
    ids_violating_absorbing_retirement = df.query(
        "(period == 0 and choice == 1) and (period == 1 and choice == 0)"
    )

    assert len(ids_violating_absorbing_retirement) == 0
    aaae(value_period_zero.mean(), expected.mean(), decimal=2)


def test_simulate_second_continuous_choice(model_setup):

    model = model_setup["model"].copy()
    model["options"]["state_space"]["continuous_states"]["experience"] = jnp.linspace(
        0, 1, 6
    )
    model["model_funcs"]["next_period_continuous_state"] = (
        determine_function_arguments_and_partial_options(
            func=next_period_experience,
            options=model["options"]["model_params"],
            continuous_state_name="experience",
        )
    )

    params = model_setup["params"]
    options = model_setup["options"]
    options["model_params"]["max_init_experience"] = 1

    key = jax.random.PRNGKey(0)
    noise = jax.random.normal(key, shape=(24, 6, 120)) * 0

    value = jnp.repeat(model_setup["value"][:, None, :], 6, axis=1) + noise
    policy = jnp.repeat(model_setup["policy"][:, None, :], 6, axis=1) + noise
    endog_grid = jnp.repeat(model_setup["endog_grid"][:, None, :], 6, axis=1) + noise

    n_agents = 100_000

    states_initial = {
        "period": np.zeros(n_agents),
        "lagged_choice": np.zeros(n_agents),  # all agents start as workers
        "married": np.zeros(n_agents),
        "ltc": np.zeros(n_agents),
        "experience": np.ones(n_agents),
    }
    wealth_initial = np.ones(n_agents) * 10

    result = simulate_all_periods(
        states_initial=states_initial,
        wealth_initial=wealth_initial,
        n_periods=options["state_space"]["n_periods"],
        params=params,
        seed=111,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        model=model,
    )

    df = create_simulation_df(result)

    value_period_zero, expected = _create_test_objects_from_df(df, params)
    ids_violating_absorbing_retirement = df.query(
        "(period == 0 and choice == 1) and (period == 1 and choice == 0)"
    )

    assert len(ids_violating_absorbing_retirement) == 0
    aaae(value_period_zero.mean(), expected.mean(), decimal=2)
