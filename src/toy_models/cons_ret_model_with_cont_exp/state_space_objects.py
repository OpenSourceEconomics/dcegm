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
        "update_continuous_state": get_next_period_experience,
    }


def get_next_period_experience(period, lagged_choice, experience, options):
    max_experience_period = period + options["max_init_experience"]
    return (1 / max_experience_period) * (
        (max_experience_period - 1) * experience + (lagged_choice == 0)
    )
