from jax import numpy as jnp
from jax import vmap

from dcegm.interfaces.index_functions import get_state_choice_index_per_discrete_states
from dcegm.interpolation.interp1d import interp1d_policy_and_value_on_wealth
from dcegm.interpolation.interp1d_dj import interp1d_policy_and_value_on_wealth_dj
from dcegm.interpolation.interp2d_irregular import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)
from dcegm.interpolation.interpnd_regular import (
    interpnd_policy_and_value_for_child_states_on_regular_grids,
)
from dcegm.law_of_motion import (
    compute_own_continuous_grid_combos,
    compute_own_continuous_grids_raw,
    compute_own_dj_wealth_grid,
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
    continuous_grid_functions,
    upper_envelope_method,
    has_additional_continuous_state,
    discount_factor,
    skip_endog_grid_storage,
    dj_wealth_grid,
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
            policy_solved,
            discrete_state_choice_indexes,
            axis=0,
            mode="fill",
            fill_value=jnp.nan,
        )[:, :, 0, :]
        if skip_endog_grid_storage:
            # DJ-constant: endog_grid isn't stored at all in this case (every
            # state-choice's "endogenous" grid is by construction the fixed
            # assets_begin_of_period grid), so this is just a placeholder -- the
            # real (possibly state-choice-specific) wealth grid is computed on
            # demand per agent-choice inside interp1d_policy_and_value_function
            # instead, self-referential via that row's own identity.
            endog_grid_agent = jnp.zeros(1)
            endog_grid_in_axes = None
        else:
            endog_grid_agent = jnp.take(
                endog_grid_solved,
                discrete_state_choice_indexes,
                axis=0,
                mode="fill",
                fill_value=jnp.nan,
            )[:, :, 0, :]
            endog_grid_in_axes = 0

        vectorized_interp = vmap(
            vmap(
                interp1d_policy_and_value_function,
                in_axes=(
                    None,
                    None,
                    endog_grid_in_axes,
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
            ),
            in_axes=(
                0,
                0,
                endog_grid_in_axes,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            ),
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
            upper_envelope_method == "druedahl_jorgensen",
            skip_endog_grid_storage,
            continuous_grid_functions,
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
            policy_solved,
            discrete_state_choice_indexes,
            axis=0,
            mode="fill",
            fill_value=jnp.nan,
        )
        endog_grid_agent = jnp.take(
            endog_grid_solved,
            discrete_state_choice_indexes,
            axis=0,
            mode="fill",
            fill_value=jnp.nan,
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
            continuous_grid_functions,
            continuous_state_name,
            params,
            compute_utility,
            discount_factor,
        )

        return policy_agent, value_agent

    if upper_envelope_method == "druedahl_jorgensen":
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
            policy_solved,
            discrete_state_choice_indexes,
            axis=0,
            mode="fill",
            fill_value=jnp.nan,
        )
        if skip_endog_grid_storage:
            # DJ-constant: broadcast only across the combo axis, never across
            # agents/choices.
            endog_grid_agent = jnp.broadcast_to(
                dj_wealth_grid, value_grid_agent.shape[2:]
            )
            endog_grid_in_axes = None
        else:
            endog_grid_agent = jnp.take(
                endog_grid_solved,
                discrete_state_choice_indexes,
                axis=0,
                mode="fill",
                fill_value=jnp.nan,
            )
            endog_grid_in_axes = 0

        additional_continuous_state_names = list(
            continuous_state_beginning_of_period.keys()
        )

        vectorized_interp = vmap(
            vmap(
                interpnd_policy_and_value_function,
                in_axes=(
                    None,
                    None,
                    None,
                    endog_grid_in_axes,
                    0,
                    0,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            ),
            in_axes=(
                0,
                0,
                0,
                endog_grid_in_axes,
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

        policy_agent, value_agent = vectorized_interp(
            assets_begin_of_period,
            continuous_state_beginning_of_period,
            discrete_states_beginning_of_period,
            endog_grid_agent,
            value_grid_agent,
            policy_grid_agent,
            choice_range,
            continuous_grid_functions,
            additional_continuous_state_names,
            params,
            compute_utility,
            discount_factor,
        )

        return policy_agent, value_agent

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
    use_dj_interpolation,
    skip_endog_grid_storage,
    continuous_grid_functions,
):
    state_choice_vec = {**state, "choice": choice}

    if use_dj_interpolation:
        # This agent's own Druedahl-Jorgensen wealth grid, evaluated on demand when
        # endog_grid isn't stored at all -- self-referential, this is exactly the
        # state-choice whose own stored solution is being read.
        wealth_grid = (
            compute_own_dj_wealth_grid(state_choice_vec, continuous_grid_functions)
            if skip_endog_grid_storage
            else endog_grid_agent
        )
        policy_interp, value_interp = interp1d_policy_and_value_on_wealth_dj(
            wealth=wealth_beginning_of_period,
            wealth_grid=wealth_grid,
            policy_grid=policy_agent,
            value_grid=value_agent,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
            discount_factor=discount_factor,
        )
    else:
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
    continuous_grid_functions,
    continuous_state_name,
    params,
    compute_utility,
    discount_factor,
):
    state_choice_vec = {**state, "choice": choice}

    # Self-referential: the stored solution for this state-choice was solved on
    # its own grid, so evaluating the grid function on itself reproduces it.
    own_continuous_state_space = {
        continuous_state_name: continuous_grid_functions[continuous_state_name](
            **state_choice_vec
        )
    }

    policy_interp, value_interp = interp2d_policy_and_value_on_wealth_and_regular_grid(
        continuous_state_space=own_continuous_state_space,
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


def interpnd_policy_and_value_function(
    wealth_beginning_of_period,
    continuous_state_beginning_of_period,
    state,
    endog_grid_agent,
    value_agent,
    policy_agent,
    choice,
    continuous_grid_functions,
    additional_continuous_state_names,
    params,
    compute_utility,
    discount_factor,
):
    state_choice_vec = {**state, "choice": choice}

    # Self-referential: the stored solution for this state-choice was solved on
    # its own grid, so evaluating the grid function on itself reproduces it.
    own_continuous_state_grids = compute_own_continuous_grids_raw(
        state_choice_vec, continuous_grid_functions, additional_continuous_state_names
    )
    own_continuous_state_space = compute_own_continuous_grid_combos(
        state_choice_vec, continuous_grid_functions, additional_continuous_state_names
    )

    continuous_state_child_states = {
        name: continuous_state_beginning_of_period[name][None, None]
        for name in additional_continuous_state_names
    }
    state_choice_child_states = {
        key: value[None] for key, value in state_choice_vec.items()
    }

    policy_interp, value_interp = (
        interpnd_policy_and_value_for_child_states_on_regular_grids(
            additional_continuous_state_grids=own_continuous_state_grids,
            wealth_grid=endog_grid_agent[0],
            policy_grid_child_states=policy_agent[None, ...],
            value_grid_child_states=value_agent[None, ...],
            continuous_state_child_states=continuous_state_child_states,
            wealth_child_states=wealth_beginning_of_period[None, None, None, None],
            state_choice_child_states=state_choice_child_states,
            compute_utility=compute_utility,
            params=params,
            discount_factor=discount_factor,
        )
    )

    exact_mask = jnp.ones_like(
        next(iter(own_continuous_state_space.values())), dtype=bool
    )
    for name in additional_continuous_state_names:
        exact_mask = exact_mask & jnp.isclose(
            own_continuous_state_space[name],
            continuous_state_beginning_of_period[name],
        )
    has_exact_combo = jnp.any(exact_mask)
    combo_idx = jnp.argmax(exact_mask)

    policy_exact, value_exact = interp1d_policy_and_value_on_wealth_dj(
        wealth=wealth_beginning_of_period,
        wealth_grid=endog_grid_agent[combo_idx],
        policy_grid=policy_agent[combo_idx],
        value_grid=value_agent[combo_idx],
        compute_utility=compute_utility,
        state_choice_vec=state_choice_vec,
        params=params,
        discount_factor=discount_factor,
    )

    policy = jnp.where(has_exact_combo, policy_exact, policy_interp[0, 0, 0, 0])
    value = jnp.where(has_exact_combo, value_exact, value_interp[0, 0, 0, 0])

    return policy, value
