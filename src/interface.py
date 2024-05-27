from dcegm.interpolation import get_index_high_and_low


def get_policy_and_value_for_state_choice(
    value, policy, endog_grid, model_structure, wealth, **state_choice_vars
):
    state_space_names = model_structure["state_space_names"]
    state_choice_names = state_space_names + ["choice"]
    map_state_choice_to_index = model_structure["map_state_choice_to_index"]

    state_choice_tuple = tuple(state_choice_vars[key] for key in state_choice_names)
    state_choice_index = map_state_choice_to_index[state_choice_tuple]
    wealth_state_choice = endog_grid[state_choice_index, :]

    ind_high, ind_low = get_index_high_and_low(wealth_state_choice, wealth)
