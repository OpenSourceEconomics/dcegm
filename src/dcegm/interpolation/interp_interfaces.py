from dcegm.interpolation.interp1d import interp_value_on_wealth
from dcegm.interpolation.interp2d import interp2d_value_on_wealth_and_regular_grid


def interpolate_value_for_state_and_choice(
    value_grid_state_choice,
    endog_grid_state_choice,
    state_choice_vec,
    params,
    model_config,
    model_funcs,
):
    """Interpolate the value for a state and choice given the respective grids."""
    continuous_states_info = model_config["continuous_states_info"]
    discount_factor = model_funcs["read_funcs"]["discount_factor"](params)

    compute_utility = model_funcs["compute_utility"]

    if continuous_states_info["second_continuous_exists"]:
        second_continuous = state_choice_vec[
            continuous_states_info["second_continuous_state_name"]
        ]

        value = interp2d_value_on_wealth_and_regular_grid(
            regular_grid=continuous_states_info["second_continuous_grid"],
            wealth_grid=endog_grid_state_choice,
            value_grid=value_grid_state_choice,
            regular_point_to_interp=second_continuous,
            wealth_point_to_interp=state_choice_vec["assets_begin_of_period"],
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
            discount_factor=discount_factor,
        )
    else:

        value = interp_value_on_wealth(
            wealth=state_choice_vec["assets_begin_of_period"],
            endog_grid=endog_grid_state_choice,
            value=value_grid_state_choice,
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
            discount_factor=discount_factor,
        )
    return value
