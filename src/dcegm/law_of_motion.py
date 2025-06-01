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
):

    continuous_states_info = model_config["continuous_states_info"]
    state_space_dict = model_structure["state_space_dict"]

    # Scale income shock draws
    income_shock_mean = model_funcs["read_funcs"]["income_shock_mean"](params)
    income_shock_std = model_funcs["read_funcs"]["income_shock_std"](params)
    income_shocks_scaled = (
        income_shock_draws_unscaled * income_shock_std + income_shock_mean
    )

    if continuous_states_info["second_continuous_exists"]:
        continuous_state_next_period = calculate_continuous_state(
            discrete_states_beginning_of_period=state_space_dict,
            continuous_grid=continuous_states_info["second_continuous_grid"],
            params=params,
            compute_continuous_state=model_funcs["next_period_continuous_state"],
        )

        # Extra dimension for continuous state
        assets_beginning_of_next_period = calc_assets_beginning_of_period_2cont(
            discrete_states_beginning_of_next_period=state_space_dict,
            continuous_state_beginning_of_next_period=continuous_state_next_period,
            assets_grid_end_of_period=continuous_states_info[
                "assets_grid_end_of_period"
            ],
            income_shocks=income_shocks_scaled,
            params=params,
            compute_assets_begin_of_period=model_funcs[
                "compute_assets_begin_of_period"
            ],
        )

        cont_grids_next_period = {
            "assets_begin_of_period": assets_beginning_of_next_period,
            "second_continuous": continuous_state_next_period,
        }

    else:
        assets_begin_of_next_period = calc_beginning_of_period_assets_1cont(
            discrete_states_beginning_of_period=state_space_dict,
            assets_grid_end_of_period=continuous_states_info[
                "assets_grid_end_of_period"
            ],
            income_shocks_current_period=income_shocks_scaled,
            params=params,
            compute_assets_begin_of_period=model_funcs[
                "compute_assets_begin_of_period"
            ],
        )
        cont_grids_next_period = {
            "assets_begin_of_period": assets_begin_of_next_period,
        }

    return cont_grids_next_period


def calc_beginning_of_period_assets_1cont(
    discrete_states_beginning_of_period,
    assets_grid_end_of_period,
    income_shocks_current_period,
    params,
    compute_assets_begin_of_period,
):
    assets_begin_of_period = vmap(
        vmap(
            vmap(
                calc_beginning_of_period_assets_1cont_vec,
                in_axes=(None, None, 0, None, None, None),  # income shocks
            ),
            in_axes=(None, 0, None, None, None, None),  # assets
        ),
        in_axes=(0, None, None, None, None, None),  # discrete states
    )(
        discrete_states_beginning_of_period,
        assets_grid_end_of_period,
        income_shocks_current_period,
        params,
        compute_assets_begin_of_period,
        False,
    )
    return assets_begin_of_period


def calc_beginning_of_period_assets_1cont_vec(
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
    asset_grid_point_end_of_previous_period,
    income_shock_draw,
    params,
    compute_assets_begin_of_period,
    aux_outs,
):

    out_budget = compute_assets_begin_of_period(
        **state_vec,
        continuous_state=continuous_state_beginning_of_period,
        asset_end_of_previous_period=asset_grid_point_end_of_previous_period,
        income_shock_previous_period=income_shock_draw,
        params=params,
    )
    checked_out = check_budget_equation_and_return_wealth_plus_optional_aux(
        out_budget, optional_aux=aux_outs
    )
    return checked_out


def calculate_continuous_state(
    discrete_states_beginning_of_period,
    continuous_grid,
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
        continuous_grid,
        params,
        compute_continuous_state,
    )
    return continuous_state_beginning_of_period


def calc_continuous_state_for_each_grid_point(
    state_vec,
    exog_continuous_grid_point,
    params,
    compute_continuous_state,
):
    out = compute_continuous_state(
        **state_vec,
        continuous_state=exog_continuous_grid_point,
        params=params,
    )
    return out


def calc_assets_beginning_of_period_2cont(
    discrete_states_beginning_of_next_period,
    continuous_state_beginning_of_next_period,
    assets_grid_end_of_period,
    income_shocks,
    params,
    compute_assets_begin_of_period,
):

    assets_begin_of_period = vmap(
        vmap(
            vmap(
                vmap(
                    calc_assets_beginning_of_period_2cont_vec,
                    in_axes=(None, None, None, 0, None, None, None),  # income shocks
                ),
                in_axes=(None, None, 0, None, None, None, None),  # assets
            ),
            in_axes=(None, 0, None, None, None, None, None),  # continuous state
        ),
        in_axes=(0, 0, None, None, None, None, None),  # discrete states
    )(
        discrete_states_beginning_of_next_period,
        continuous_state_beginning_of_next_period,
        assets_grid_end_of_period,
        income_shocks,
        params,
        compute_assets_begin_of_period,
        False,
    )
    return assets_begin_of_period


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
        calc_beginning_of_period_assets_1cont_vec,
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


def calc_assets_begin_of_period_for_all_agents(
    states_beginning_of_period,
    continuous_state_beginning_of_period,
    assets_end_of_period,
    income_shocks_of_period,
    params,
    compute_assets_begin_of_period,
):
    """Simulation."""

    assets_begin_of_next_period, aux_dict = vmap(
        calc_assets_beginning_of_period_2cont_vec,
        in_axes=(0, 0, 0, 0, None, None, None),
    )(
        states_beginning_of_period,
        continuous_state_beginning_of_period,
        assets_end_of_period,
        income_shocks_of_period,
        params,
        compute_assets_begin_of_period,
        True,
    )
    return assets_begin_of_next_period, aux_dict
