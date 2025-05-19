def create_state_space_functions():
    """Return dict with state space functions."""
    out = {
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_experience": next_period_experience,
        "sparsity_condition": sparsity_condition,
        "next_period_deterministic_state": next_period_deterministic_state,
    }
    return out


def state_specific_choice_set(period, lagged_choice, job_offer, model_specs):

    # Retirement is absorbing
    if lagged_choice == 0:
        return [0]
    # If period equal or larger max ret age you have to choose retirement
    elif period >= model_specs["max_ret_period"]:
        return [0]
    # If above minimum retirement period, retirement is possible
    elif period >= model_specs["min_ret_period"]:
        if job_offer == 1:
            return [0, 1, 2]
        else:
            return [0, 1]
    # If below then only working is possible
    else:
        if job_offer == 1:
            return [1, 2]
        else:
            return [1]


def sparsity_condition(
    period, lagged_choice, job_offer, already_retired, survival, model_specs
):
    last_period = model_specs["n_periods"] - 1

    # If above minimum retirement period, retirement is possible
    if (period <= model_specs["min_ret_period"]) & (lagged_choice == 0):
        return False
    elif (lagged_choice != 0) & (already_retired == 1):
        return False
    elif (period <= model_specs["min_ret_period"] + 1) & (already_retired == 1):
        return False
    # If period equal or larger max ret age you have to choose retirement
    elif (
        (period > model_specs["max_ret_period"] + 1)
        & (already_retired != 1)
        & (survival != 0)
    ):
        return False
    elif (
        (period > model_specs["max_ret_period"])
        & (lagged_choice != 0)
        & (survival != 0)
    ):
        return False
    else:
        if survival == 0:
            return {
                "period": last_period,
                "lagged_choice": lagged_choice,
                "already_retired": already_retired,
                "survival": survival,
                "job_offer": 0,
            }
        else:
            if lagged_choice == 0:
                return {
                    "period": period,
                    "lagged_choice": lagged_choice,
                    "already_retired": already_retired,
                    "survival": survival,
                    "job_offer": 0,
                }
            else:
                return {
                    "period": period,
                    "lagged_choice": lagged_choice,
                    "already_retired": already_retired,
                    "survival": survival,
                    "job_offer": job_offer,
                }


def next_period_deterministic_state(period, choice, lagged_choice):
    if (lagged_choice == 0) & (choice == 0):
        return {
            "period": period + 1,
            "lagged_choice": choice,
            "already_retired": 1,
        }
    else:
        return {
            "period": period + 1,
            "lagged_choice": choice,
            "already_retired": 0,
        }


def next_period_experience(lagged_choice, experience, model_specs):
    """If working add one year of experience years (1/exp_scale)"""
    working = lagged_choice == 2
    return experience + working * (1 / model_specs["exp_scale"])
