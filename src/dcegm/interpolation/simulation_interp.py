from jax import numpy as jnp
from jax import vmap

from dcegm.interfaces.index_functions import get_state_choice_index_per_discrete_states
from dcegm.interpolation.interp1d import interp1d_policy_and_value_on_wealth
from dcegm.interpolation.interp2d_irregular import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)


def interpolate_policy_and_value_for_all_agents(
    discrete_states_beginning_of_period,
    continuous_state_beginning_of_period,
    assets_begin_of_period,
    value_solved,
    policy_solved,
    endog_grid_solved,
    map_state_choice_to_index,
    choice_range,
    params,
    discrete_states_names,
    compute_utility,
    continuous_state_space,
    upper_envelope_method,
    has_additional_continuous_state,
    discount_factor,
):

    # 1D interpolation path is independent of upper-envelope method and only
    # depends on whether an additional continuous state exists.
    if not has_additional_continuous_state:
        discrete_state_choice_indexes = get_state_choice_index_per_discrete_states(
            states=discrete_states_beginning_of_period,
            map_state_choice_to_index=map_state_choice_to_index,
            discrete_states_names=discrete_states_names,
        )

        value_grid_agent = jnp.take(
            value_solved,
            discrete_state_choice_indexes,
            axis=0,
            mode="fill",
            fill_value=jnp.nan,
        )[:, :, 0, :]
        policy_grid_agent = jnp.take(
            policy_solved, discrete_state_choice_indexes, axis=0
        )[:, :, 0, :]
        endog_grid_agent = jnp.take(
            endog_grid_solved, discrete_state_choice_indexes, axis=0
        )[:, :, 0, :]

        vectorized_interp = vmap(
            vmap(
                interp1d_policy_and_value_function,
                in_axes=(None, None, 0, 0, 0, 0, None, None, None),  # choices
            ),
            in_axes=(0, 0, 0, 0, 0, None, None, None, None),  # agents
        )

        policy_agent, value_agent = vectorized_interp(
            assets_begin_of_period,
            discrete_states_beginning_of_period,
            endog_grid_agent,
            value_grid_agent,
            policy_grid_agent,
            choice_range,
            params,
            compute_utility,
            discount_factor,
        )

        return policy_agent, value_agent

    if upper_envelope_method == "fues":

        discrete_state_choice_indexes = get_state_choice_index_per_discrete_states(
            states=discrete_states_beginning_of_period,
            map_state_choice_to_index=map_state_choice_to_index,
            discrete_states_names=discrete_states_names,
        )

        value_grid_agent = jnp.take(
            value_solved,
            discrete_state_choice_indexes,
            axis=0,
            mode="fill",
            fill_value=jnp.nan,
        )
        policy_grid_agent = jnp.take(
            policy_solved, discrete_state_choice_indexes, axis=0
        )
        endog_grid_agent = jnp.take(
            endog_grid_solved, discrete_state_choice_indexes, axis=0
        )

        continuous_state_name = list(continuous_state_beginning_of_period.keys())[0]

        vectorized_interp = vmap(
            vmap(
                interp2d_policy_and_value_function,
                in_axes=(
                    None,
                    None,
                    None,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),  # choices
            ),
            in_axes=(
                0,
                0,
                0,
                0,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )

        # =================================================================================

        policy_agent, value_agent = vectorized_interp(
            assets_begin_of_period,
            continuous_state_beginning_of_period,
            discrete_states_beginning_of_period,
            endog_grid_agent,
            value_grid_agent,
            policy_grid_agent,
            choice_range,
            continuous_state_space,
            continuous_state_name,
            params,
            compute_utility,
            discount_factor,
        )

        return policy_agent, value_agent

    if upper_envelope_method == "druedahl_jorgensen":
        raise NotImplementedError(
            "Simulation interpolation for 'druedahl_jorgensen' with additional "
            "continuous states is not implemented yet."
        )

    raise ValueError(
        "Unknown upper envelope method. Use 'fues' or 'druedahl_jorgensen'."
    )


def interp1d_policy_and_value_function(
    wealth_beginning_of_period,
    state,
    endog_grid_agent,
    value_agent,
    policy_agent,
    choice,
    params,
    compute_utility,
    discount_factor,
):
    state_choice_vec = {**state, "choice": choice}

    policy_interp, value_interp = interp1d_policy_and_value_on_wealth(
        wealth=wealth_beginning_of_period,
        wealth_grid=endog_grid_agent,
        policy_grid=policy_agent,
        value_grid=value_agent,
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
        discount_factor=discount_factor,
    )

    return policy_interp, value_interp


def interp2d_policy_and_value_function(
    wealth_beginning_of_period,
    continuous_state_beginning_of_period,
    state,
    endog_grid_agent,
    value_agent,
    policy_agent,
    choice,
    continuous_state_space,
    continuous_state_name,
    params,
    compute_utility,
    discount_factor,
):
    state_choice_vec = {**state, "choice": choice}

    policy_interp, value_interp = interp2d_policy_and_value_on_wealth_and_regular_grid(
        continuous_state_space=continuous_state_space,
        wealth_grid=endog_grid_agent,
        policy_grid=policy_agent,
        value_grid=value_agent,
        wealth_point_to_interp=wealth_beginning_of_period,
        regular_point_to_interp=continuous_state_beginning_of_period[
            continuous_state_name
        ],
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
        discount_factor=discount_factor,
    )

    return policy_interp, value_interp
