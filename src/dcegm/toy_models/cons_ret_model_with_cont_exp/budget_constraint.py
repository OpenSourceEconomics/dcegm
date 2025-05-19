from dcegm.toy_models.cons_ret_model_with_exp.budget_constraint_with_exp import (
    budget_constraint_exp,
)


def budget_constraint_cont_exp(
    period,
    lagged_choice,
    experience,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
    model_specs,
):
    max_init_experience_period = period + model_specs["max_init_experience"]
    experience_years = experience * max_init_experience_period

    return budget_constraint_exp(
        lagged_choice=lagged_choice,
        experience=experience_years,
        asset_end_of_previous_period=asset_end_of_previous_period,
        income_shock_previous_period=income_shock_previous_period,
        params=params,
    )
