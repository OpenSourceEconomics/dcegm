import jax
import jax.numpy as jnp
from dcegm.egm.interpolate_marginal_utility import interpolate_policy_and_check_value
from dcegm.interpolation import get_index_high_and_low
from jax import vmap


def simulate_single_period(
    states_and_wealth_beginning_of_period,
    period,
    params,
    basic_seed,
    endog_grid_solved,
    value_solved,
    policy_left_solved,
    policy_right_solved,
    map_state_choice_to_index,
    choice_range,
    compute_exog_transition_vec,
    compute_utility,
    compute_beginning_of_period_wealth,
):
    (
        states_beginning_of_period,
        wealth_beginning_of_period,
    ) = states_and_wealth_beginning_of_period
    state_choice_indexes = get_state_choice_index_per_state(
        map_state_choice_to_index, states_beginning_of_period
    )
    value_grid_agent = jnp.take(value_solved, state_choice_indexes, axis=0)
    policy_left_grid_agent = jnp.take(policy_left_solved, state_choice_indexes, axis=0)
    policy_right_grid_agent = jnp.take(
        policy_right_solved, state_choice_indexes, axis=0
    )
    endog_grid_agent = jnp.take(endog_grid_solved, state_choice_indexes, axis=0)

    vectorized_interp = vmap(
        vmap(
            interpolate_policy_and_value_function_for_agents,
            in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
        ),
        in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
    )

    policy_agent, value_agent = vectorized_interp(
        wealth_beginning_of_period,
        states_beginning_of_period,
        endog_grid_agent,
        value_grid_agent,
        policy_left_grid_agent,
        policy_right_grid_agent,
        choice_range,
        params,
        compute_utility,
    )
    basic_seed + period
    jax.random.PRNGKey(basic_seed + period)

    policy_agent + value_agent

    breakpoint()


def get_trans_mat(exog_func, state_choice_vec, params):
    transition_vec = exog_func(params=params, **state_choice_vec)
    return transition_vec


def get_state_choice_index_per_state(map_state_choice_to_index, states):
    indexes = map_state_choice_to_index[tuple((states[key],) for key in states.keys())][
        0
    ]
    return indexes


def interpolate_policy_and_value_function_for_agents(
    wealth_beginning_of_period,
    state,
    endog_grid_agent,
    value_agent,
    policy_left_agent,
    policy_right_agent,
    choice,
    params,
    compute_utility,
):
    ind_high, ind_low = get_index_high_and_low(
        x=endog_grid_agent, x_new=wealth_beginning_of_period
    )
    state_choice_vec = {**state, "choice": choice}
    policy_interp, value_interp = interpolate_policy_and_check_value(
        policy_high=policy_left_agent[ind_high],
        value_high=value_agent[ind_high],
        wealth_high=endog_grid_agent[ind_high],
        policy_low=policy_right_agent[ind_low],
        value_low=value_agent[ind_low],
        wealth_low=endog_grid_agent[ind_low],
        new_wealth=wealth_beginning_of_period,
        compute_utility=compute_utility,
        endog_grid_min=endog_grid_agent[1],
        value_min=value_agent[0],
        state_choice_vec=state_choice_vec,
        params=params,
    )
    return policy_interp, value_interp
