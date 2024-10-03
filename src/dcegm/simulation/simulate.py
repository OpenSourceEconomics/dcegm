"""The simulation function."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import vmap

from dcegm.interface import get_state_choice_index_per_discrete_state
from dcegm.simulation.sim_utils import (
    compute_final_utility_for_each_choice,
    draw_taste_shocks,
    interpolate_policy_and_value_for_all_agents,
    transition_to_next_period,
    vectorized_utility,
)


def simulate_all_periods(
    states_initial,
    resources_initial,
    n_periods,
    params,
    seed,
    endog_grid_solved,
    policy_solved,
    value_solved,
    model,
):

    second_continuous_state_name_and_grid = next(
        (
            {key: value}
            for key, value in model["options"]["state_space"][
                "continuous_states"
            ].items()
            if key != "wealth"
        ),
        None,
    )
    second_continuous_state_name = (
        next(iter(second_continuous_state_name_and_grid.keys()))
        if second_continuous_state_name_and_grid
        else None
    )

    discrete_state_space = model["model_structure"]["state_space_dict"]

    # Set initial states to internal dtype
    states_initial_dtype = {
        key: value.astype(discrete_state_space[key].dtype)
        for key, value in states_initial.items()
        if key in discrete_state_space
    }

    if "dummy_exog" in model["model_structure"]["exog_states_names"]:
        states_initial_dtype["dummy_exog"] = np.zeros_like(
            states_initial_dtype["period"]
        )

    if second_continuous_state_name_and_grid:
        states_initial_dtype[second_continuous_state_name] = states_initial[
            second_continuous_state_name
        ]

    # Prepare random seeds for taste shocks
    n_keys = len(resources_initial) + 2
    sim_specific_keys = jnp.array(
        [
            jax.random.split(jax.random.PRNGKey(seed + period), num=n_keys)
            for period in range(n_periods)
        ]
    )

    model_structure = model["model_structure"]
    model_funcs = model["model_funcs"]

    compute_next_period_states = {
        "get_next_period_state": model_funcs["get_next_period_state"],
        # "update_continuous_state": model_funcs["update_continuous_state"],
        "update_continuous_state_for_next_period": model_funcs[
            "update_continuous_state_for_next_period"
        ],
    }

    simulate_body = partial(
        simulate_single_period,
        params=params,
        state_space_names=model_structure["state_space_names"],
        endog_grid_solved=endog_grid_solved,
        value_solved=value_solved,
        policy_solved=policy_solved,
        map_state_choice_to_index=jnp.asarray(
            model_structure["map_state_choice_to_index"]
        ),
        choice_range=model_structure["choice_range"],
        compute_exog_transition_vec=model_funcs["compute_exog_transition_vec"],
        compute_utility=model_funcs["compute_utility"],
        compute_beginning_of_period_resources=model_funcs[
            "compute_beginning_of_period_resources"
        ],
        exog_state_mapping=model_funcs["exog_state_mapping"],
        compute_next_period_states=compute_next_period_states,
        second_continuous_state=second_continuous_state_name_and_grid,
    )

    states_and_resources_beginning_of_first_period = (
        states_initial_dtype,
        resources_initial,
    )

    states_and_resources_beginning_of_final_period, sim_dict = jax.lax.scan(
        f=simulate_body,
        init=states_and_resources_beginning_of_first_period,
        xs=sim_specific_keys[:-1],
        unroll=1,
    )

    final_period_dict = simulate_final_period(
        states_and_resources_beginning_of_final_period,
        sim_specific_keys=sim_specific_keys[-1],
        params=params,
        state_space_names=model_structure["state_space_names"],
        choice_range=model_structure["choice_range"],
        map_state_choice_to_index=model_structure["map_state_choice_to_index"],
        compute_utility_final_period=model_funcs["compute_utility_final"],
        second_continuous_state=second_continuous_state_name_and_grid,
    )

    result = {
        key: np.vstack([sim_dict[key], final_period_dict[key]])
        for key in sim_dict.keys()
    }

    return result


def simulate_single_period(
    states_and_resources_beginning_of_period,
    sim_specific_keys,
    params,
    state_space_names,
    endog_grid_solved,
    value_solved,
    policy_solved,
    map_state_choice_to_index,
    choice_range,
    compute_exog_transition_vec,
    compute_utility,
    compute_beginning_of_period_resources,
    exog_state_mapping,
    compute_next_period_states,
    second_continuous_state=None,
):
    (
        states_beginning_of_period,
        resources_beginning_of_period,
    ) = states_and_resources_beginning_of_period

    if second_continuous_state:
        continuous_state_name = list(second_continuous_state.keys())[0]
        continuous_grid = second_continuous_state[continuous_state_name]

        continuous_state_beginning_of_period = states_beginning_of_period[
            continuous_state_name
        ]
        discrete_states_beginning_of_period = {
            key: value
            for key, value in states_beginning_of_period.items()
            if key != continuous_state_name
        }
    else:
        discrete_states_beginning_of_period = states_beginning_of_period
        continuous_state_beginning_of_period = None
        continuous_grid = None

    # Interpolate policy and value function for all agents.
    policy, values_pre_taste_shock = interpolate_policy_and_value_for_all_agents(
        discrete_states_beginning_of_period=discrete_states_beginning_of_period,
        continuous_state_beginning_of_period=continuous_state_beginning_of_period,
        resources_beginning_of_period=resources_beginning_of_period,
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
        map_state_choice_to_index=map_state_choice_to_index,
        choice_range=choice_range,
        params=params,
        state_space_names=state_space_names,
        compute_utility=compute_utility,
        continuous_grid=continuous_grid,
    )

    # Draw taste shocks and calculate final value.
    taste_shocks = draw_taste_shocks(
        n_agents=len(resources_beginning_of_period),
        n_choices=len(choice_range),
        taste_shock_scale=params["lambda"],
        key=sim_specific_keys[0, :],
    )
    values_across_choices = values_pre_taste_shock + taste_shocks

    # Determine choice index of period by max value and select corresponding choice,
    # consumption and value.
    choice_index = jnp.nanargmax(values_across_choices, axis=1)
    choice = choice_range[choice_index]

    value_max = jnp.take_along_axis(
        values_across_choices, choice_index[:, None], axis=1
    )[:, 0]

    consumption = jnp.take_along_axis(policy, choice_index[:, None], axis=1)[:, 0]
    utility_period = vmap(vectorized_utility, in_axes=(0, 0, 0, None, None))(
        consumption,
        states_beginning_of_period,
        choice,
        params,
        compute_utility,
    )
    savings_current_period = resources_beginning_of_period - consumption

    (
        resources_beginning_of_next_period,
        discrete_states_next_period,
        continuous_state_next_period,
        income_shocks_next_period,
    ) = transition_to_next_period(
        discrete_states_beginning_of_period=discrete_states_beginning_of_period,
        continuous_state_beginning_of_period=continuous_state_beginning_of_period,
        savings_current_period=savings_current_period,
        choice=choice,
        params=params,
        compute_exog_transition_vec=compute_exog_transition_vec,
        exog_state_mapping=exog_state_mapping,
        compute_beginning_of_period_resources=compute_beginning_of_period_resources,
        compute_next_period_states=compute_next_period_states,
        sim_specific_keys=sim_specific_keys,
    )

    states_next_period = discrete_states_next_period

    if second_continuous_state:
        states_next_period[continuous_state_name] = continuous_state_next_period

    carry = states_next_period, resources_beginning_of_next_period

    result = {
        "choice": choice,
        "consumption": consumption,
        "utility": utility_period,
        "taste_shocks": taste_shocks,
        "value_max": value_max,
        "value_choice": values_across_choices,
        "savings": savings_current_period,
        "income_shock": income_shocks_next_period,
        **states_beginning_of_period,
    }

    return carry, result


def simulate_final_period(
    states_and_resources_beginning_of_period,
    sim_specific_keys,
    params,
    state_space_names,
    choice_range,
    map_state_choice_to_index,
    compute_utility_final_period,
    second_continuous_state=None,
):
    invalid_number = np.iinfo(map_state_choice_to_index.dtype).max

    (
        states_beginning_of_final_period,
        resources_beginning_of_final_period,
    ) = states_and_resources_beginning_of_period

    n_choices = len(choice_range)
    n_agents = len(resources_beginning_of_final_period)

    utilities_pre_taste_shock = vmap(
        vmap(
            compute_final_utility_for_each_choice,
            in_axes=(None, 0, None, None, None),  # choices
        ),
        in_axes=(0, None, 0, None, None),  # agents
    )(
        states_beginning_of_final_period,
        choice_range,
        resources_beginning_of_final_period,
        params,
        compute_utility_final_period,
    )
    state_choice_indexes = get_state_choice_index_per_discrete_state(
        map_state_choice_to_index=map_state_choice_to_index,
        states=states_beginning_of_final_period,
        state_space_names=state_space_names,
    )
    utilities_pre_taste_shock = jnp.where(
        state_choice_indexes == invalid_number, np.nan, utilities_pre_taste_shock
    )

    # Draw taste shocks and calculate final value.
    taste_shocks = draw_taste_shocks(
        n_agents=n_agents,
        n_choices=n_choices,
        taste_shock_scale=params["lambda"],
        key=sim_specific_keys[0, :],
    )
    values_across_choices = utilities_pre_taste_shock + taste_shocks

    choice_index = jnp.nanargmax(values_across_choices, axis=1)
    choice = choice_range[choice_index]

    utility_period = jnp.take_along_axis(
        utilities_pre_taste_shock, choice_index[:, None], axis=1
    )[:, 0]
    value_period = jnp.take_along_axis(
        values_across_choices, choice_index[:, None], axis=1
    )[:, 0]

    result = {
        "choice": choice,
        "consumption": resources_beginning_of_final_period,
        "utility": utility_period,
        "value_max": value_period,
        "value_choice": values_across_choices[np.newaxis],
        "taste_shocks": taste_shocks[np.newaxis, :, :],
        "savings": np.zeros_like(utility_period),
        "income_shock": np.zeros(n_agents),
        **states_beginning_of_final_period,
    }

    return result
