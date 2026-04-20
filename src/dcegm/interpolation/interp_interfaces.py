from dcegm.interpolation.interp1d import (
    interp1d_policy_and_value_on_wealth,
    interp_policy_on_wealth,
    interp_value_on_wealth,
)
from dcegm.interpolation.interp2d_irregular import (
    interp2d_policy_and_value_on_wealth_and_regular_grid,
)


def interpolate_value_for_state_and_choice(
    value_grid_state_choice,
    endog_grid_state_choice,
    state_choice_vec,
    params,
    model_config,
    model_funcs,
    model_structure,
):
    """Interpolate the value for a state and choice given the respective grids."""
    continuous_states_info = model_config["continuous_states_info"]
    upper_envelope_method = model_config["upper_envelope"]["method"]
    discount_factor = model_funcs["read_funcs"]["discount_factor"](params)

    compute_utility = model_funcs["compute_utility"]

    multidim = continuous_states_info["has_additional_continuous_state"]

    continuous_state_space = model_structure["continuous_state_space"]

    if (upper_envelope_method == "fues") & multidim:
        continuous_state_name = continuous_states_info[
            "additional_continuous_state_names"
        ][0]
        continuous_state = state_choice_vec[continuous_state_name]

        _, value = interp2d_policy_and_value_on_wealth_and_regular_grid(
            continuous_state_space=continuous_state_space,
            wealth_grid=endog_grid_state_choice,
            policy_grid=endog_grid_state_choice,
            value_grid=value_grid_state_choice,
            regular_point_to_interp=continuous_state,
            wealth_point_to_interp=state_choice_vec["assets_begin_of_period"],
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
            discount_factor=discount_factor,
        )

    elif (upper_envelope_method == "druedahl_jorgensen") & multidim:
        raise NotImplementedError(
            "Interface interpolation for 'druedahl_jorgensen' is not implemented yet."
        )
    else:
        value = interp_value_on_wealth(
            wealth=state_choice_vec["assets_begin_of_period"],
            wealth_grid=endog_grid_state_choice[0],
            value=value_grid_state_choice[0],
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
            discount_factor=discount_factor,
        )
    return value


def interpolate_policy_for_state_and_choice(
    policy_grid_state_choice,
    endog_grid_state_choice,
    state_choice_vec,
    model_config,
    model_structure,
):
    """Interpolate the value for a state and choice given the respective grids."""
    continuous_states_info = model_config["continuous_states_info"]
    upper_envelope_method = model_config["upper_envelope"]["method"]
    multidim = continuous_states_info["has_additional_continuous_state"]
    continuous_state_space = model_structure["continuous_state_space"]

    if (upper_envelope_method == "fues") & multidim:
        continuous_state_name = continuous_states_info[
            "additional_continuous_state_names"
        ][0]
        continuous_state = state_choice_vec[continuous_state_name]
        policy, _ = interp2d_policy_and_value_on_wealth_and_regular_grid(
            continuous_state_space=continuous_state_space,
            wealth_grid=endog_grid_state_choice,
            policy_grid=policy_grid_state_choice,
            value_grid=policy_grid_state_choice,
            regular_point_to_interp=continuous_state,
            wealth_point_to_interp=state_choice_vec["assets_begin_of_period"],
            compute_utility=lambda consumption, params, **kwargs: consumption,
            state_choice_vec=state_choice_vec,
            params={},
            discount_factor=0.0,
        )
    elif (upper_envelope_method == "druedahl_jorgensen") & multidim:
        raise NotImplementedError(
            "Interface interpolation for 'druedahl_jorgensen' is not implemented yet."
        )
    else:
        policy = interp_policy_on_wealth(
            wealth=state_choice_vec["assets_begin_of_period"],
            endog_grid=endog_grid_state_choice[0],
            policy=policy_grid_state_choice[0],
        )

    return policy


def interpolate_policy_and_value_for_state_and_choice(
    value_grid_state_choice,
    policy_grid_state_choice,
    endog_grid_state_choice,
    state_choice_vec,
    params,
    model_config,
    model_funcs,
    model_structure,
):
    continuous_states_info = model_config["continuous_states_info"]
    upper_envelope_method = model_config["upper_envelope"]["method"]

    compute_utility = model_funcs["compute_utility"]
    discount_factor = model_funcs["read_funcs"]["discount_factor"](params)
    continuous_state_space = model_structure["continuous_state_space"]

    multidim = continuous_states_info["has_additional_continuous_state"]

    if (upper_envelope_method == "fues") & multidim:
        continuous_state_name = continuous_states_info[
            "additional_continuous_state_names"
        ][0]
        continuous_state = state_choice_vec[continuous_state_name]
        policy, value = interp2d_policy_and_value_on_wealth_and_regular_grid(
            continuous_state_space=continuous_state_space,
            wealth_grid=endog_grid_state_choice,
            policy_grid=policy_grid_state_choice,
            value_grid=value_grid_state_choice,
            regular_point_to_interp=continuous_state,
            wealth_point_to_interp=state_choice_vec["assets_begin_of_period"],
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
            discount_factor=discount_factor,
        )
    elif (upper_envelope_method == "druedahl_jorgensen") & multidim:
        raise NotImplementedError(
            "Interface interpolation for 'druedahl_jorgensen' is not implemented yet."
        )
    else:
        policy, value = interp1d_policy_and_value_on_wealth(
            wealth=state_choice_vec["assets_begin_of_period"],
            wealth_grid=endog_grid_state_choice[0],
            policy_grid=policy_grid_state_choice[0],
            value_grid=value_grid_state_choice[0],
            compute_utility=compute_utility,
            state_choice_vec=state_choice_vec,
            params=params,
            discount_factor=discount_factor,
        )

    return policy, value
