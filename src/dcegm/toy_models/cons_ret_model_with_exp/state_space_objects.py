from dcegm.toy_models.cons_ret_model_dcegm_paper.state_space_objects import (
    get_state_specific_feasible_choice_set,
)


def create_state_space_function_dict():
    """Create dictionary with state space functions.

    Returns:
        state_space_functions (dict): Dictionary with state space functions.

    """
    return {
        "state_specific_choice_set": get_state_specific_feasible_choice_set,
        "next_period_deterministic_state": next_period_deterministic_state,
        "sparsity_condition": sparsity_condition,
    }


def next_period_deterministic_state(period, choice, experience):
    """Update state with experience."""
    next_state = {}

    next_state["period"] = period + 1
    next_state["lagged_choice"] = choice
    next_state["experience"] = experience + (choice == 0)

    return next_state


def sparsity_condition(
    period,
    experience,
    lagged_choice,
    model_specs,
):
    max_exp_period = period + model_specs["max_init_experience"]
    max_total_experience = model_specs["n_periods"] + model_specs["max_init_experience"]

    # Experience must be smaller than the maximum experience in a period
    if max_exp_period < experience:
        return False
    # Experience must be smaller than the maximum total experience
    elif max_total_experience <= experience:
        return False
    # If experience is the maximum experience in a period, you must have been working last period
    elif (experience == max_exp_period) & (lagged_choice == 1):
        return False
    # As retirement is absorbing, if you have been working last period
    # your experience must be at least as big as the period as you
    # had to been working all periods before
    elif (lagged_choice == 0) & (experience < period):
        return False
    else:
        return True
