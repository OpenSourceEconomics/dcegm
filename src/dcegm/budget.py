from jax import vmap


def calculate_resources(
    states_beginning_of_period,
    savings_end_of_last_period,
    income_shocks_of_period,
    params,
    compute_beginning_of_period_wealth,
):
    resources_beginning_of_period = vmap(
        vmap(
            vmap(
                calculate_resources_for_each_grid_point,
                in_axes=(None, None, 0, None, None),
            ),
            in_axes=(None, 0, None, None, None),
        ),
        in_axes=(0, None, None, None, None),
    )(
        states_beginning_of_period,
        savings_end_of_last_period,
        income_shocks_of_period,
        params,
        compute_beginning_of_period_wealth,
    )
    return resources_beginning_of_period


def calculate_resources_for_each_grid_point(
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
