from toy_models.cons_ret_model_with_exp.budget_constraint_with_exp import (
    budget_constraint_exp,
)


def budget_constraint_cont_exp(
    period,
    lagged_choice,
    experience,
    savings_end_of_previous_period,
    income_shock_previous_period,
    params,
):
    experience_years = experience * period
    return budget_constraint_exp(
        lagged_choice=lagged_choice,
        experience=experience_years,
        savings_end_of_previous_period=savings_end_of_previous_period,
        income_shock_previous_period=income_shock_previous_period,
        params=params,
    )
