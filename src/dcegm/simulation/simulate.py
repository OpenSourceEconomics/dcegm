"""The simulation function."""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from dcegm.interface import get_state_choice_index_per_state
from dcegm.simulation.sim_utils import compute_final_utility_for_each_choice
from dcegm.simulation.sim_utils import draw_taste_shocks
from dcegm.simulation.sim_utils import interpolate_policy_and_value_for_all_agents
from dcegm.simulation.sim_utils import transition_to_next_period
from dcegm.simulation.sim_utils import vectorized_utility
from jax import vmap


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
        get_next_period_state=model_funcs["get_next_period_state"],
    )

    states_and_resources_beginning_of_first_period = states_initial, resources_initial
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
    )

    result = {
        key: np.row_stack([sim_dict[key], final_period_dict[key]])
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
    get_next_period_state,
):
    (
        states_beginning_of_period,
        resources_beginning_of_period,
    ) = states_and_resources_beginning_of_period

    # Interpolate policy and value function for all agents.
    policy, values_pre_taste_shock = interpolate_policy_and_value_for_all_agents(
        states_beginning_of_period=states_beginning_of_period,
        resources_beginning_of_period=resources_beginning_of_period,
        value_solved=value_solved,
        policy_solved=policy_solved,
        endog_grid_solved=endog_grid_solved,
        map_state_choice_to_index=map_state_choice_to_index,
        choice_range=choice_range,
        params=params,
        state_space_names=state_space_names,
        compute_utility=compute_utility,
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
        states_next_period,
        income_shocks_next_period,
    ) = transition_to_next_period(
        states_beginning_of_period=states_beginning_of_period,
        savings_current_period=savings_current_period,
        choice=choice,
        params=params,
        compute_exog_transition_vec=compute_exog_transition_vec,
        exog_state_mapping=exog_state_mapping,
        compute_beginning_of_period_resources=compute_beginning_of_period_resources,
        get_next_period_state=get_next_period_state,
        sim_specific_keys=sim_specific_keys,
    )
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
):
    (
        states_beginning_of_final_period,
        resources_beginning_of_final_period,
    ) = states_and_resources_beginning_of_period

    n_choices = len(choice_range)
    n_agents = len(resources_beginning_of_final_period)

    utilities_pre_taste_shock = vmap(
        vmap(
            compute_final_utility_for_each_choice,
            in_axes=(None, 0, None, None, None),
        ),
        in_axes=(0, None, 0, None, None),
    )(
        states_beginning_of_final_period,
        choice_range,
        resources_beginning_of_final_period,
        params,
        compute_utility_final_period,
    )
    state_choice_indexes = get_state_choice_index_per_state(
        map_state_choice_to_index=map_state_choice_to_index,
        states=states_beginning_of_final_period,
        state_space_names=state_space_names,
    )
    utilities_pre_taste_shock = jnp.where(
        state_choice_indexes < 0, np.nan, utilities_pre_taste_shock
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
