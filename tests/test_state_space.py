from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest
from dcegm.pre_processing.debugging import inspect_state_space
from dcegm.pre_processing.state_space import create_state_space
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)

DATA_TYPE = np.uint8


@pytest.fixture()
def options(load_example_model):
    """Return options dictionary."""
    _, _raw_options = load_example_model("retirement_no_taste_shocks")
    _raw_options["n_choices"] = 2
    options = {}

    options["model_params"] = _raw_options
    options.update(
        {
            "state_space": {
                "n_periods": 25,
                "choices": np.arange(2),
                "endogenous_states": {
                    "thus": np.arange(25),
                    "that": [0, 1],
                },
                "exogenous_processes": {
                    "ltc": {"states": np.array([0]), "transition": jnp.array([0])}
                },
            },
        }
    )

    # Determine dtypes
    options["state_space"]["dtypes"] = {
        "state_space": DATA_TYPE,
        "state_choice_space": DATA_TYPE,
        "max_int_state_space": np.iinfo(DATA_TYPE).max,
        "max_int_state_choice_space": np.iinfo(DATA_TYPE).max,
    }

    return options


def expected_state_space_and_indexer(n_periods, n_choices, n_exog_states):
    _periods = np.arange(n_periods)
    _choices = np.arange(n_choices)
    _exog_process = np.arange(n_exog_states)
    state_space = np.column_stack(
        [
            np.repeat(_periods, n_choices * n_exog_states),
            np.tile(np.repeat(_choices, n_exog_states), n_periods),
            np.tile(_exog_process, n_periods * n_choices),
        ]
    )
    state_indexer = np.arange(n_periods * n_choices * n_exog_states).reshape(
        n_periods, n_choices, n_exog_states
    )

    return state_space, state_indexer


n_periods = [15, 25, 63, 100]
n_choices = [2, 3, 20, 50]
n_exog_processes = [2, 3, 5]
lagged_choices = [0, 1]

TEST_CASES = list(product(n_periods, n_choices, n_exog_processes))


TEST_CASES = list(product(lagged_choices, n_periods, n_choices, n_exog_processes))


@pytest.mark.parametrize(
    "lagged_choice, n_periods, n_choices, n_exog_states", TEST_CASES
)
def test_state_choice_set(lagged_choice, n_periods, n_choices, n_exog_states):
    choice_set = get_state_specific_feasible_choice_set(
        lagged_choice=lagged_choice, options={"n_choices": n_choices}
    )

    # retirement (lagged_choice == 1) is absorbing
    expected_choice_set = np.arange(n_choices) if lagged_choice == 0 else np.array([1])

    assert np.allclose(choice_set, expected_choice_set)


def create_state_space_test(options):
    n_periods = options["n_periods"]
    n_choices = options["n_discrete_choices"]
    resolution_age = options["resolution_age"]
    start_age = options["start_age"]

    # The highest policy state, we consider belongs to the expectation of the youngest.
    n_policy_states = (resolution_age - start_age) + 1

    # minimum retirement age is 4 years before the lowest statutory ret age
    min_ret_age = options["minimum_SRA"] - 4
    # maximum (conceivable) retirement age is given by lowest SRA plus the projection
    # of the youngest
    max_ret_age = options["maximum_retirement_age"]
    # number of possible actual retirement ages
    n_ret_ages = max_ret_age - min_ret_age + 1

    # shape = (n_periods, n_choices, n_exog_states)
    state_space = []

    shape = (n_periods, n_choices, n_periods, n_policy_states, n_ret_ages, 1)

    map_state_to_index = np.full(shape, fill_value=-9999, dtype=np.int64)
    i = 0

    for period in range(n_periods):
        for lag_choice in range(n_choices):
            # You cannot have more experience than your age
            for exp in range(period + 1):
                # The policy state we need to consider increases by one increment
                # per period.
                for policy_state in range(period + 1):
                    for actual_retirement_id in range(n_ret_ages):
                        age = start_age + period
                        actual_retirement_age = min_ret_age + actual_retirement_id
                        # You cannot retire before the earliest retirement age
                        if (age <= min_ret_age) & (lag_choice == 2):
                            continue
                        # After the maximum retirement age, you must be retired
                        elif (age > max_ret_age) & (lag_choice != 2):
                            continue
                        # If you weren't retired last period, your actual
                        # retirement age is kept at minimum
                        elif (lag_choice != 2) & (actual_retirement_id > 0):
                            continue
                        # If you are retired, your actual retirement age can
                        # at most be your current age
                        elif (lag_choice == 2) & (age <= actual_retirement_age):
                            continue
                        # Starting from resolution age, there is no more adding
                        # of policy states.
                        elif policy_state > n_policy_states - 1:
                            continue
                        # If you have not worked last period, you can't have
                        # worked all your live
                        elif (lag_choice != 1) & (period == exp) & (period > 0):
                            continue
                        else:
                            state_space += [
                                [
                                    period,
                                    lag_choice,
                                    exp,
                                    policy_state,
                                    actual_retirement_id,
                                    0,
                                ]
                            ]
                            map_state_to_index[
                                period,
                                lag_choice,
                                exp,
                                policy_state,
                                actual_retirement_id,
                                0,
                            ] = i
                            i += 1

    return np.array(state_space), map_state_to_index


def sparsity_condition(
    period, lagged_choice, policy_state, retirement_age_id, experience, options
):
    min_ret_age = options["minimum_SRA"] - 4
    start_age = options["start_age"]
    resolution_age = options["resolution_age"]
    max_ret_age = options["maximum_retirement_age"]

    n_policy_states = (resolution_age - start_age) + 1
    age = start_age + period
    actual_retirement_age = min_ret_age + retirement_age_id
    # You cannot retire before the earliest retirement age
    if (age <= min_ret_age) & (lagged_choice == 2):
        return False
    # After the maximum retirement age, you must be retired
    elif (age > max_ret_age) & (lagged_choice != 2):
        return False
    # If you weren't retired last period, your actual retirement age is kept at minimum
    elif (lagged_choice != 2) & (retirement_age_id > 0):
        return False
    # If you are retired, your actual retirement age can at most be your current age
    elif (lagged_choice == 2) & (age <= actual_retirement_age):
        return False
    # Starting from resolution age, there is no more adding of policy states.
    elif policy_state > n_policy_states - 1:
        return False
    # If you have not worked last period, you can't have worked all your live
    elif (lagged_choice != 1) & (period == experience) & (period > 0):
        return False
    # You cannot have more experience than your age
    elif experience > period:
        return False
    # The policy state we need to consider increases by one increment
    # per period.
    elif policy_state > period:
        return False
    else:
        return True


def test_state_space():
    n_periods = 50
    options_sparse = {
        "state_space": {
            "n_periods": n_periods,  # 25 + 50 = 75
            "choices": np.arange(3, dtype=np.int64),
            "endogenous_states": {
                "experience": np.arange(n_periods, dtype=int),
                "policy_state": np.arange(36, dtype=int),
                "retirement_age_id": np.arange(10, dtype=int),
                "sparsity_condition": sparsity_condition,
            },
        },
        "model_params": {
            "n_periods": n_periods,  # 25 + 50 = 75
            "n_discrete_choices": 3,
            "start_age": 25,
            "resolution_age": 60,
            "minimum_SRA": 67,
            "maximum_retirement_age": 72,
        },
    }
    options_sparse["state_space"]["dtypes"] = {
        "state_space": DATA_TYPE,
        "state_choice_space": DATA_TYPE,
        "max_int_state_space": np.iinfo(DATA_TYPE).max,
        "max_int_state_choice_space": np.iinfo(DATA_TYPE).max,
    }

    state_space_test, _ = create_state_space_test(options_sparse["model_params"])
    (
        state_space,
        state_space_dict,
        map_state_to_index,
        states_names_without_exog,
        exog_states_names,
        exog_state_space,
    ) = create_state_space(options=options_sparse)

    # The dcegm package create the state vector in the order of the dictionary keys.
    # How these are ordered is not clear ex ante.
    state_space_sums_test = state_space_test.sum(axis=0)
    state_space_sums = state_space.sum(axis=0)
    state_space_sum_dict = {
        key: state_space_sums[i]
        for i, key in enumerate(states_names_without_exog + exog_states_names)
    }

    np.testing.assert_allclose(state_space_sum_dict["period"], state_space_sums_test[0])
    np.testing.assert_allclose(
        state_space_sum_dict["lagged_choice"], state_space_sums_test[1]
    )
    np.testing.assert_allclose(
        state_space_sum_dict["experience"], state_space_sums_test[2]
    )
    np.testing.assert_allclose(
        state_space_sum_dict["policy_state"], state_space_sums_test[3]
    )
    np.testing.assert_allclose(
        state_space_sum_dict["retirement_age_id"], state_space_sums_test[4]
    )
    np.testing.assert_allclose(
        state_space_sum_dict["dummy_exog"], state_space_sums_test[5]
    )

    ### Now test the inspection function.
    state_space_df = inspect_state_space(options=options_sparse)
    admissible_df = state_space_df[state_space_df["is_feasible"]]

    for i, column in enumerate(states_names_without_exog + exog_states_names):
        np.testing.assert_allclose(admissible_df[column].values, state_space[:, i])
