# from dcegm.simulation.sim_utils import get_state_choice_index_per_state
# import jax.numpy as jnp
# from dcegm.interpolation import get_index_high_and_low
# from dcegm.egm.interpolate_marginal_utility import interpolate_policy_and_check_value
# from dcegm.egm.aggregate_marginal_utility import calculate_choice_probs_and_unsqueezed_logsum
#
#
# def interp_value(
#         observed_states,
#         oberseved_wealth,
#         value_solved,
#         endog_grid_solved,
#         map_state_choice_to_index,
#         choice_range,
#         params,
#         state_space_names,
#         compute_utility,
# ):
#     """This function interpolates the policy and value function for all agents.
#
#         It uses the states at the beginning of period to select the solved policy and value
#         and then interpolates the wealth at the beginning of period on them.
#
#         """
#     state_choice_indexes = get_state_choice_index_per_state(
#         map_state_choice_to_index=map_state_choice_to_index,
#         states=observed_states,
#         state_space_names=state_space_names,
#     )
#
#     value_grid_agent = jnp.take(
#         value_solved, state_choice_indexes, axis=0, mode="fill", fill_value=jnp.nan
#     )
#     endog_grid_agent = jnp.take(endog_grid_solved, state_choice_indexes, axis=0)
#     vectorized_interp = vmap(
#         vmap(
#             interpolate_policy_and_value_function,
#             in_axes=(None, None, 0, 0, 0, 0, 0, None, None),
#         ),
#         in_axes=(0, 0, 0, 0, 0, 0, None, None, None),
#     )
#
#     policy_agent, value_per_agent_interp = vectorized_interp(
#         resources_beginning_of_period,
#         states_beginning_of_period,
#         endog_grid_agent,
#         value_grid_agent,
#         policy_left_grid_agent,
#         policy_right_grid_agent,
#         choice_range,
#         params,
#         compute_utility,
#     )
#
#
# def interpolate_value_and_calc_choice_probabilities(
#     resources_beginning_of_period,
#     state,
#     endog_grid_agent,
#     value_agent,
#     policy_left_agent,
#     policy_right_agent,
#     choice,
#     params,
#     compute_utility,
# ):
#     ind_high, ind_low = get_index_high_and_low(
#         x=endog_grid_agent, x_new=resources_beginning_of_period
#     )
#     state_choice_vec = {**state, "choice": choice}
#     policy_interp, value_interp = interpolate_policy_and_check_value(
#         policy_high=policy_left_agent[ind_high],
#         value_high=value_agent[ind_high],
#         wealth_high=endog_grid_agent[ind_high],
#         policy_low=policy_right_agent[ind_low],
#         value_low=value_agent[ind_low],
#         wealth_low=endog_grid_agent[ind_low],
#         new_wealth=resources_beginning_of_period,
#         compute_utility=compute_utility,
#         endog_grid_min=endog_grid_agent[1],
#         value_at_zero_wealth=value_agent[0],
#         state_choice_vec=state_choice_vec,
#         params=params,
#     )
#     choice_probs, _, _ = calculate_choice_probs_and_unsqueezed_logsum(
#         choice_values_per_state=value_interp,
#         taste_shock_scale = params["lambda"]
#     )
#
#     return policy_interp, value_interp