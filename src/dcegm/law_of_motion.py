from jax import vmap


def calc_cont_grids_next_period(
    state_space_dict,
    exog_grids,
    income_shock_draws_unscaled,
    params,
    model_funcs,
    has_second_continuous_state,
):
    scaled_income_shocks = (income_shock_draws_unscaled - params["income_shock_mean"]) * params["sigma"]
    if has_second_continuous_state:
        continuous_state_next_period = calculate_continuous_state(
            discrete_states_beginning_of_period=state_space_dict,
            continuous_grid=exog_grids["second_continuous"],
            params=params,
            compute_continuous_state=model_funcs["next_period_continuous_state"],
        )

        # Extra dimension for continuous state
        wealth_beginning_of_next_period = calculate_wealth_for_second_continuous_state(
            discrete_states_beginning_of_next_period=state_space_dict,
            continuous_state_beginning_of_next_period=continuous_state_next_period,
            savings_grid=exog_grids["wealth"],
            income_shocks=scaled_income_shocks,
            params=params,
            compute_beginning_of_period_wealth=model_funcs[
                "compute_beginning_of_period_wealth"
            ],
        )

        cont_grids_next_period = {
            "wealth": wealth_beginning_of_next_period,
            "second_continuous": continuous_state_next_period,
        }

    else:
        wealth_next_period = calculate_wealth(
            discrete_states_beginning_of_period=state_space_dict,
            savings_grid=exog_grids["wealth"],
            income_shocks_current_period=scaled_income_shocks,
            params=params,
            compute_beginning_of_period_wealth=model_funcs[
                "compute_beginning_of_period_wealth"
            ],
        )
        cont_grids_next_period = {
            "wealth": wealth_next_period,
        }
    return cont_grids_next_period


def calculate_wealth(
    discrete_states_beginning_of_period,
    savings_grid,
    income_shocks_current_period,
    params,
    compute_beginning_of_period_wealth,
):
    wealth_beginning_of_period = vmap(
        vmap(
            vmap(
                calc_wealth_for_each_savings_grid_point,
                in_axes=(None, None, 0, None, None),  # income shocks
            ),
            in_axes=(None, 0, None, None, None),  # savings
        ),
        in_axes=(0, None, None, None, None),  # discrete states
    )(
        discrete_states_beginning_of_period,
        savings_grid,
        income_shocks_current_period,
        params,
        compute_beginning_of_period_wealth,
    )
    return wealth_beginning_of_period


def calc_wealth_for_each_savings_grid_point(
    state_vec,
    exog_savings_grid_point,
    income_shock_draw,
    params,
    compute_beginning_of_period_wealth,
):
    out = compute_beginning_of_period_wealth(
        **state_vec,
        savings_end_of_previous_period=exog_savings_grid_point,
        income_shock_previous_period=income_shock_draw,
        params=params,
    )
    return out


# =====================================================================================
# Second continuous state
# =====================================================================================


def calc_wealth_for_each_continuous_state_and_savings_grid_point(
    state_vec,
    continuous_state_beginning_of_period,
    exog_savings_grid_point,
    income_shock_draw,
    params,
    compute_beginning_of_period_wealth,
):

    out = compute_beginning_of_period_wealth(
        **state_vec,
        continuous_state=continuous_state_beginning_of_period,
        savings_end_of_previous_period=exog_savings_grid_point,
        income_shock_previous_period=income_shock_draw,
        params=params,
    )

    return out


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


def calculate_wealth_for_second_continuous_state(
    discrete_states_beginning_of_next_period,
    continuous_state_beginning_of_next_period,
    savings_grid,
    income_shocks,
    params,
    compute_beginning_of_period_wealth,
):

    wealth_beginning_of_period = vmap(
        vmap(
            vmap(
                vmap(
                    calc_wealth_for_each_continuous_state_and_savings_grid_point,
                    in_axes=(None, None, None, 0, None, None),  # income shocks
                ),
                in_axes=(None, None, 0, None, None, None),  # savings
            ),
            in_axes=(None, 0, None, None, None, None),  # continuous state
        ),
        in_axes=(0, 0, None, None, None, None),  # discrete states
    )(
        discrete_states_beginning_of_next_period,
        continuous_state_beginning_of_next_period,
        savings_grid,
        income_shocks,
        params,
        compute_beginning_of_period_wealth,
    )
    return wealth_beginning_of_period


# =====================================================================================
# Simulation
# =====================================================================================


def calculate_wealth_for_all_agents(
    states_beginning_of_period,
    savings_end_of_previous_period,
    income_shocks_of_period,
    params,
    compute_beginning_of_period_wealth,
):
    wealth_beginning_of_next_period = vmap(
        calc_wealth_for_each_savings_grid_point,
        in_axes=(0, 0, 0, None, None),
    )(
        states_beginning_of_period,
        savings_end_of_previous_period,
        income_shocks_of_period,
        params,
        compute_beginning_of_period_wealth,
    )
    return wealth_beginning_of_next_period


def calculate_second_continuous_state_for_all_agents(
    discrete_states_beginning_of_period,
    continuous_state_beginning_of_period,
    params,
    compute_continuous_state,
):
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


def calculate_wealth_given_second_continuous_state_for_all_agents(
    states_beginning_of_period,
    continuous_state_beginning_of_period,
    savings_end_of_previous_period,
    income_shocks_of_period,
    params,
    compute_beginning_of_period_wealth,
):
    wealth_beginning_of_next_period = vmap(
        calc_wealth_for_each_continuous_state_and_savings_grid_point,
        in_axes=(0, 0, 0, 0, None, None),
    )(
        states_beginning_of_period,
        continuous_state_beginning_of_period,
        savings_end_of_previous_period,
        income_shocks_of_period,
        params,
        compute_beginning_of_period_wealth,
    )
    return wealth_beginning_of_next_period
