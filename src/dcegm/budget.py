from jax import vmap

# def calculate_resources_second_continuous_state(
#     states_beginning_of_period,
#     savings_end_of_previous_period,
#     income_shocks_of_period,
#     params,
#     compute_beginning_of_period_resources,
# ):
#     return 0


def calculate_continuous_state(
    discrete_states_beginning_of_period,
    continuous_grid,
    params,
    compute_continuous_state,
):
    continuous_state_beginning_of_period = vmap(
        vmap(
            vmap(
                vmap(
                    calc_continuous_state_for_each_grid_point,
                    in_axes=(None, None, None, 0, None, None),  # income shocks
                ),
                in_axes=(None, None, 0, None, None, None),  # savings
            ),
            in_axes=(None, 0, None, None, None, None),  # continuous state
        ),
        in_axes=(0, None, None, None, None, None),  # discrete states
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
        continuous_grid_point=exog_continuous_grid_point,
        params=params,
    )
    return out


def _update_continuous_state(period, lagged_choice, continuous_grid_point, options):

    working_hours = _transform_lagged_choice_to_working_hours(lagged_choice)

    return (
        1
        / (period + 1)
        * (
            period * continuous_grid_point
            + (working_hours) / options["working_hours_max"]  # 3000
        )
    )


def _transform_lagged_choice_to_working_hours(lagged_choice):

    not_working = lagged_choice == 0
    part_time = lagged_choice == 1
    full_time = lagged_choice == 2

    return not_working * 0 + part_time * 2000 + full_time * 3000


def calculate_resources_for_second_continuous_state(
    discrete_states_beginning_of_next_period,
    continuous_state_beginning_of_next_period,
    savings_end_of_last_period,
    income_shocks_of_period,
    params,
    compute_beginning_of_period_resources,
):
    resources_beginning_of_period = vmap(
        vmap(
            vmap(
                vmap(
                    calc_resources_for_each_continuous_state_and_savings_grid_point,
                    in_axes=(None, None, None, 0, None, None),  # income shocks
                ),
                in_axes=(None, None, 0, None, None, None),  # savings
            ),
            in_axes=(None, 0, None, None, None, None),  # continuous state
        ),
        in_axes=(0, None, None, None, None, None),  # discrete states
    )(
        discrete_states_beginning_of_next_period,
        continuous_state_beginning_of_next_period,
        savings_end_of_last_period,
        income_shocks_of_period,
        params,
        compute_beginning_of_period_resources,
    )
    return resources_beginning_of_period


# ======================================================================================


def calculate_resources(
    discrete_states_beginning_of_period,
    savings_end_of_last_period,
    income_shocks_of_period,
    params,
    compute_beginning_of_period_resources,
):
    resources_beginning_of_period = vmap(
        vmap(
            vmap(
                calc_resources_for_each_savings_grid_point,
                in_axes=(None, None, 0, None, None),  # income shocks
            ),
            in_axes=(None, 0, None, None, None),  # savings
        ),
        in_axes=(0, None, None, None, None),  # discrete states
    )(
        discrete_states_beginning_of_period,
        savings_end_of_last_period,
        income_shocks_of_period,
        params,
        compute_beginning_of_period_resources,
    )
    return resources_beginning_of_period


def calc_resources_for_each_savings_grid_point(
    state_vec,
    exog_savings_grid_point,
    income_shock_draw,
    params,
    compute_beginning_of_period_resources,
):
    out = compute_beginning_of_period_resources(
        **state_vec,
        savings_end_of_previous_period=exog_savings_grid_point,
        income_shock_previous_period=income_shock_draw,
        params=params,
    )
    return out


def calc_resources_for_each_continuous_state_and_savings_grid_point(
    state_vec,
    continuous_state_beginning_of_period,
    exog_savings_grid_point,
    income_shock_draw,
    params,
    compute_beginning_of_period_resources,
):
    out = compute_beginning_of_period_resources(
        **state_vec,
        continuous_state_beginning_of_period=continuous_state_beginning_of_period,
        savings_end_of_previous_period=exog_savings_grid_point,
        income_shock_previous_period=income_shock_draw,
        params=params,
    )
    return out


def calculate_resources_for_all_agents(
    states_beginning_of_period,
    savings_end_of_previous_period,
    income_shocks_of_period,
    params,
    compute_beginning_of_period_resources,
):
    resources_beginning_of_next_period = vmap(
        calc_resources_for_each_savings_grid_point,
        in_axes=(0, 0, 0, None, None),
    )(
        states_beginning_of_period,
        savings_end_of_previous_period,
        income_shocks_of_period,
        params,
        compute_beginning_of_period_resources,
    )
    return resources_beginning_of_next_period
