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

    # Generate result dict
    cont_grids_next_period = {}

    if continuous_states_info["second_continuous_exists"]:
        continuous_state_next_period = calculate_continuous_state(
            discrete_states_beginning_of_period=state_space_dict,
            continuous_grid=continuous_states_info["second_continuous_grid"],
            params=params,
            compute_continuous_state=model_funcs["next_period_continuous_state"],
        )
        # Fill in result dict
        cont_grids_next_period["second_continuous"] = continuous_state_next_period

        # Prepare dict used to calculate beginning of period assets
        state_specific_grids = {
            "states": state_space_dict,
            "continuous_state": continuous_state_next_period,
        }
    else:
        state_specific_grids = {
            "states": state_space_dict,
        }

    def fix_assets_and_shocks_for_broadcast(
        states,
        asset_end_of_previous_period,
        income_draw,
    ):
        assets_begin_of_period = calc_beginning_of_period_assets_for_single_state(
            state_vec=states,
            asset_end_of_previous_period=asset_end_of_previous_period,
            income_shock_draw=income_draw,
            params=params,
            compute_assets_begin_of_period=model_funcs[
                "compute_assets_begin_of_period"
            ],
            aux_outs=False,
        )
        return assets_begin_of_period

    broadcast_function = lambda states: vmap(
        vmap(
            fix_assets_and_shocks_for_broadcast,
            in_axes=(None, None, 0),  # income shocks
        ),
        in_axes=(None, 0, None),  # assets
    )(
        states,
        continuous_states_info["assets_grid_end_of_period"],
        income_shocks_scaled,
    )

    final_args = ()
    # Default is no chaining of vmaps. Then I add consequently vmap over specific grids
    vmap_chain = broadcast_function

    for grid_name in state_specific_grids.keys():
        if grid_name != "states":
            # Use default argument to capture current values
            vmap_chain = add_vmap_chain_for_grid(vmap_chain, grid_name)
            final_args += (state_specific_grids[grid_name],)

    final_args = (state_specific_grids["states"],) + final_args
    assets_begin_of_next_period = vmap(vmap_chain)(*final_args)
    cont_grids_next_period["assets_begin_of_period"] = assets_begin_of_next_period
    return cont_grids_next_period


def add_vmap_chain_for_grid(inner_func, gname):
    """The function adds a vmap layer for a specific grid.

    It vmaps over the remaining dimension of the grid. So if we have a grid that is
    (n_discrete_states, n_grid_points), we can later vmap over the discrete states and
    this function will add the n_grid_points dimension to be vmapped over. The function
    only expects later the grid to arrive in n_grid_points. So we can also use the
    function in the final period calculation.

    """

    def grid_wrapper(states, new_state_grid):
        all_states = {**states, gname: new_state_grid}
        return inner_func(all_states)

    return vmap(grid_wrapper, in_axes=(None, 0))


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
