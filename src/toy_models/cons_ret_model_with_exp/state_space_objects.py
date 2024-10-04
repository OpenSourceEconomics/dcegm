from toy_models.cons_ret_model_dcegm_paper.state_space_objects import (
    get_state_specific_feasible_choice_set,
)


def create_state_space_function_dict():
    """Create dictionary with state space functions.

    Returns:
        state_space_functions (dict): Dictionary with state space functions.

    """
    return {
        "get_state_specific_choice_set": get_state_specific_feasible_choice_set,
        "get_next_period_state": get_next_period_state,
    }


def get_next_period_state(period, choice, experience):
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
    options,
):
    if period < experience:
        return False
    elif experience >= options["n_periods"]:
        return False
    elif (experience == period) & (lagged_choice == 1):
        return False
    elif (period > 0) & (lagged_choice == 0) & (experience == 0):
        return False
    else:
        return True
