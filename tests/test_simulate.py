"""Tests for simulation of consumption-retirement model with exogenous processes."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

from dcegm.simulation.sim_utils import create_simulation_df
from dcegm.simulation.simulate import (
    simulate_all_periods,
    simulate_final_period,
    simulate_single_period,
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
        "period": np.zeros(n_agents, dtype=np.int64),
        "lagged_choice": np.zeros(
            n_agents, dtype=np.int64
        ),  # all agents start as workers
        "married": np.zeros(n_agents, dtype=np.int64),
        "ltc": np.zeros(n_agents, dtype=np.int64),
    }
    initial_resources = np.ones(n_agents) * 10
    initial_states_and_resources = initial_states, initial_resources

    n_keys = len(initial_resources) + 2
    sim_specific_keys = jnp.array(
        [
            jax.random.split(jax.random.PRNGKey(seed + period), num=n_keys)
            for period in range(n_periods)
        ]
    )

    return {
        **toy_model_exog_ltc,
        "initial_states": initial_states,
        "initial_resources": initial_resources,
        "initial_states_and_resources": initial_states_and_resources,
        "sim_specific_keys": sim_specific_keys,
        "seed": seed,
    }


def test_simulate_lax_scan(model_setup):
    params = model_setup["params"]
    options = model_setup["options"]
    choice_range = options["state_space"]["choices"]
    model_structure = model_setup["model"]["model_structure"]
    model_funcs = model_setup["model"]["model_funcs"]

    state_space_names = model_structure["state_space_names"]
    map_state_choice_to_index = model_structure["map_state_choice_to_index"]

    exog_state_mapping = model_funcs["exog_state_mapping"]
    get_next_period_state = model_funcs["get_next_period_state"]

    value = model_setup["value"]
    policy = model_setup["policy"]
    endog_grid = model_setup["endog_grid"]

    initial_states_and_resources = (
        model_setup["initial_states"],
        model_setup["initial_resources"],
    )
    sim_specific_keys = model_setup["sim_specific_keys"]

    simulate_body = partial(
        simulate_single_period,
        params=params,
        state_space_names=state_space_names,
        endog_grid_solved=endog_grid,
        value_solved=value,
        policy_solved=policy,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        choice_range=jnp.arange(map_state_choice_to_index.shape[-1], dtype=jnp.int64),
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
        compute_utility=model_funcs["compute_utility"],
        compute_beginning_of_period_resources=model_funcs[
            "compute_beginning_of_period_resources"
        ],
        exog_state_mapping=exog_state_mapping,
        get_next_period_state=get_next_period_state,
    )

    # lax.scan
    (
        lax_states_and_resources_beginning_of_final_period,
        lax_sim_dict_zero,
    ) = jax.lax.scan(
        f=simulate_body,
        init=initial_states_and_resources,
        xs=sim_specific_keys[:-1],
    )

    # single call
    (
        states_and_resources_beginning_of_final_period,
        sim_dict_zero,
    ) = simulate_body(
        initial_states_and_resources, sim_specific_keys=sim_specific_keys[0]
    )

    lax_final_period_dict = simulate_final_period(
        lax_states_and_resources_beginning_of_final_period,
        sim_specific_keys=sim_specific_keys[-1],
        params=params,
        state_space_names=state_space_names,
        choice_range=choice_range,
        map_state_choice_to_index=jnp.array(map_state_choice_to_index),
        compute_utility_final_period=model_funcs["compute_utility_final"],
    )
    final_period_dict = simulate_final_period(
        states_and_resources_beginning_of_final_period,
        sim_specific_keys=sim_specific_keys[-1],
        params=params,
        state_space_names=state_space_names,
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

    # We need 64 because we do not alter the model array dtypes.
    initial_states = {
        "period": np.zeros(n_agents, dtype=np.int64),
        "lagged_choice": np.zeros(
            n_agents, dtype=np.int64
        ),  # all agents start as workers
        "married": np.zeros(n_agents, dtype=np.int64),
        "ltc": np.zeros(n_agents, dtype=np.int64),
    }
    initial_resources = np.ones(n_agents) * 10

    result = simulate_all_periods(
        states_initial=initial_states,
        resources_initial=initial_resources,
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
