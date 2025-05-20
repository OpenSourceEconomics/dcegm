import jax
import jax.numpy as jnp


def create_state_space_function_dict():
    """Return dict with state space functions."""
    return {
        "state_specific_choice_set": state_specific_choice_set,
        "next_period_experience": next_period_experience,
        "sparsity_condition": sparsity_condition,
        "next_period_deterministic_state": next_period_deterministic_state,
    }


def state_specific_choice_set(period, lagged_choice, job_offer, model_specs):
    """Return available choices depending on state."""
    min_ret = model_specs["min_ret_period"]
    max_ret = model_specs["max_ret_period"]

    # Determine index based on state conditions
    def get_index(period, lagged_choice):
        is_retired = lagged_choice == 0
        must_retire = period >= max_ret
        can_retire = period >= min_ret
        return jnp.where(
            is_retired, 0, jnp.where(must_retire, 1, jnp.where(can_retire, 2, 3))
        )

    idx = get_index(period, lagged_choice)

    def retired_case(_):
        return [0]

    def must_retire_case(_):
        return [0]

    def can_retire_case(_):
        return jax.lax.cond(
            job_offer == 1, lambda _: [0, 1, 2], lambda _: [0, 1], operand=None
        )

    def only_work_case(_):
        return jax.lax.cond(
            job_offer == 1, lambda _: [1, 2], lambda _: [1], operand=None
        )

    return jax.lax.switch(
        idx,
        [retired_case, must_retire_case, can_retire_case, only_work_case],
        operand=None,
    )


def sparsity_condition(
    period, lagged_choice, job_offer, already_retired, survival, model_specs
):
    last_period = model_specs["n_periods"] - 1
    min_ret = model_specs["min_ret_period"]
    max_ret = model_specs["max_ret_period"]

    c1 = (period <= min_ret) & (lagged_choice == 0)
    c2 = (lagged_choice != 0) & (already_retired == 1)
    c3 = (period <= min_ret + 1) & (already_retired == 1)
    c4 = (period > max_ret + 1) & (already_retired != 1) & (survival != 0)
    c5 = (period > max_ret) & (lagged_choice != 0) & (survival != 0)

    if c1 or c2 or c3 or c4 or c5:
        return False

    job_offer_out = 0 if (survival == 0 or lagged_choice == 0) else job_offer
    period_out = last_period if survival == 0 else period

    return {
        "period": period_out,
        "lagged_choice": lagged_choice,
        "already_retired": already_retired,
        "survival": survival,
        "job_offer": job_offer_out,
    }


def next_period_deterministic_state(period, choice, lagged_choice):
    is_retired = (lagged_choice == 0) & (choice == 0)
    return {
        "period": period + 1,
        "lagged_choice": choice,
        "already_retired": jnp.where(is_retired, 1, 0),
    }


def next_period_experience(lagged_choice, experience, model_specs):
    working = lagged_choice == 2
    return experience + (1 / model_specs["exp_scale"]) * working
