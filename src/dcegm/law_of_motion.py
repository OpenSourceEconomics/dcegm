from jax import vmap

from dcegm.check_func_outputs import (
    check_budget_equation_and_return_wealth_plus_optional_aux,
)


def calc_cont_grids_next_period(
    params,
    income_shock_draws_unscaled,
    model_structure,
    model_config,
    model_funcs,
    has_additional_continuous_states=None,
):

    continuous_states_info = model_config["continuous_states_info"]
    state_space_dict = model_structure["state_space_dict"]

    if has_additional_continuous_states is None:
        has_additional_continuous_states = continuous_states_info[
            "has_additional_continuous_state"
        ]

    # Scale income shock draws
    income_shock_mean = model_funcs["read_funcs"]["income_shock_mean"](params)
    income_shock_std = model_funcs["read_funcs"]["income_shock_std"](params)
    income_shocks_scaled = (
        income_shock_draws_unscaled * income_shock_std + income_shock_mean
    )

    continuous_state_space = model_structure["continuous_state_space"]
    continuous_state_next_period = _get_continuous_state_next_period(
        has_additional_continuous_states=has_additional_continuous_states,
        state_space_dict=state_space_dict,
        continuous_state_space=continuous_state_space,
        params=params,
        model_funcs=model_funcs,
    )

    def fix_assets_and_shocks_for_broadcast(
        states,
        continuous_state_vec,
        asset_end_of_previous_period,
        income_draw,
    ):
        all_states = {**states, **continuous_state_vec}
        assets_begin_of_period = calc_beginning_of_period_assets_for_single_state(
            state_vec=all_states,
            asset_end_of_previous_period=asset_end_of_previous_period,
            income_shock_draw=income_draw,
            params=params,
            compute_assets_begin_of_period=model_funcs[
                "compute_assets_begin_of_period"
            ],
            aux_outs=False,
        )
        return assets_begin_of_period

    assets_begin_of_next_period = vmap(
        vmap(
            vmap(
                vmap(
                    fix_assets_and_shocks_for_broadcast,
                    in_axes=(None, None, None, 0),
                ),
                in_axes=(None, None, 0, None),
            ),
            in_axes=(None, 0, None, None),
        ),
        in_axes=(0, 0, None, None),
    )(
        state_space_dict,
        continuous_state_next_period,
        continuous_states_info["assets_grid_end_of_period"],
        income_shocks_scaled,
    )

    # Generate result dict
    return {
        "continuous_state_space": continuous_state_next_period,
        "assets_begin_of_period": assets_begin_of_next_period,
    }


def _get_continuous_state_next_period(
    has_additional_continuous_states,
    state_space_dict,
    continuous_state_space,
    params,
    model_funcs,
):
    if not has_additional_continuous_states:
        # Use dummy continuous state space to keep shapes constant.
        return continuous_state_space

    continuous_state_next_period = calculate_continuous_state(
        discrete_states_beginning_of_period=state_space_dict,
        continuous_states_end_of_last_period=continuous_state_space,
        params=params,
        compute_continuous_state=model_funcs["next_period_continuous_state"],
    )
    _check_continuous_state_output_keys(
        continuous_state_output=continuous_state_next_period,
        continuous_state_space=continuous_state_space,
    )
    return continuous_state_next_period


def _check_continuous_state_output_keys(
    continuous_state_output,
    continuous_state_space,
):
    expected_keys = set(continuous_state_space.keys())
    output_keys = set(continuous_state_output.keys())
    if output_keys != expected_keys:
        raise ValueError(
            "next_period_continuous_state output keys must match continuous_state_space keys. "
            f"Expected {sorted(expected_keys)}, got {sorted(output_keys)}."
        )


def calc_beginning_of_period_assets_for_single_state(
    state_vec,
    asset_end_of_previous_period,
    income_shock_draw,
    params,
    compute_assets_begin_of_period,
    aux_outs,
):
    out_budget = compute_assets_begin_of_period(
        **state_vec,
        asset_end_of_previous_period=asset_end_of_previous_period,
        income_shock_previous_period=income_shock_draw,
        params=params,
    )
    checked_out = check_budget_equation_and_return_wealth_plus_optional_aux(
        out_budget, optional_aux=aux_outs
    )
    return checked_out


# =====================================================================================
# Second continuous state
# =====================================================================================


def calc_assets_beginning_of_period_2cont_vec(
    state_vec,
    continuous_state_beginning_of_period,
    asset_end_of_previous_period,
    income_shock_draw,
    params,
    compute_assets_begin_of_period,
    aux_outs,
):
    all_states = {
        **state_vec,
        "continuous_state": continuous_state_beginning_of_period,
    }
    checked_out = calc_beginning_of_period_assets_for_single_state(
        state_vec=all_states,
        asset_end_of_previous_period=asset_end_of_previous_period,
        income_shock_draw=income_shock_draw,
        params=params,
        compute_assets_begin_of_period=compute_assets_begin_of_period,
        aux_outs=aux_outs,
    )
    return checked_out


def calculate_continuous_state(
    discrete_states_beginning_of_period,
    continuous_states_end_of_last_period,
    params,
    compute_continuous_state,
):
    continuous_state_beginning_of_period = vmap(
        vmap(
            calc_continuous_state_for_each_grid_point,
            in_axes=(None, 0, None, None),  # continuous state
        ),
        in_axes=(0, None, None, None),  # discrete states
    )(
        discrete_states_beginning_of_period,
        continuous_states_end_of_last_period,
        params,
        compute_continuous_state,
    )
    return continuous_state_beginning_of_period


def calc_continuous_state_for_each_grid_point(
    state_vec,
    continuous_state_vec,
    params,
    compute_continuous_state,
):
    out = compute_continuous_state(
        **state_vec,
        **continuous_state_vec,
        params=params,
    )
    return out


# =====================================================================================
# Simulation
# =====================================================================================


def calculate_assets_begin_of_period_for_all_agents(
    states_beginning_of_period,
    asset_grid_point_end_of_previous_period,
    income_shocks_of_period,
    params,
    compute_assets_begin_of_period,
):
    """Simulation."""
    assets_begin_of_next_period = vmap(
        calc_beginning_of_period_assets_for_single_state,
        in_axes=(0, 0, 0, None, None, None),
    )(
        states_beginning_of_period,
        asset_grid_point_end_of_previous_period,
        income_shocks_of_period,
        params,
        compute_assets_begin_of_period,
        True,
    )
    return assets_begin_of_next_period


def calculate_second_continuous_state_for_all_agents(
    discrete_states_beginning_of_period,
    continuous_state_beginning_of_period,
    params,
    compute_continuous_state,
):
    """Simulation."""
    continuous_state_beginning_of_next_period = vmap(
        calc_continuous_state_for_each_grid_point,
        in_axes=(0, 0, None, None),
    )(
        discrete_states_beginning_of_period,
        continuous_state_beginning_of_period,
        params,
        compute_continuous_state,
    )
    return continuous_state_beginning_of_next_period
