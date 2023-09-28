from jax import vmap


def calculate_resources(
    state_space,
    exog_savings_grid,
    income_shock_draws_unscaled,
    params,
    compute_beginning_of_period_wealth,
):
    income_shock_draws = income_shock_draws_unscaled * params["sigma"]
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
        state_space,
        exog_savings_grid,
        income_shock_draws,
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
        state_beginning_of_period=state_vec,
        savings_end_of_previous_period=exog_savings_grid_point,
        income_shock_previous_period=income_shock_draw,
        params=params,
    )
    return out
