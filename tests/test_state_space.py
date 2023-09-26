from itertools import product

import numpy as np
import pytest
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)


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


@pytest.mark.parametrize("n_periods, n_choices, n_exog_states", TEST_CASES)
def test_state_space(n_periods, n_choices, n_exog_states):
    options = {
        "n_periods": n_periods,
        "n_discrete_choices": n_choices,
        "n_exog_states": n_exog_states,
    }

    state_space, map_state_to_state_space_index = create_state_space(options)

    expected_state_space, expected_state_indexer = expected_state_space_and_indexer(
        n_periods, n_choices, n_exog_states
    )

    assert np.allclose(state_space, expected_state_space)
    assert np.allclose(map_state_to_state_space_index, expected_state_indexer)


TEST_CASES = list(product(lagged_choices, n_periods, n_choices, n_exog_processes))


@pytest.mark.parametrize(
    "lagged_choice, n_periods, n_choices, n_exog_states", TEST_CASES
)
def test_state_choice_set(lagged_choice, n_periods, n_choices, n_exog_states):
    _, map_state_to_state_space_index = expected_state_space_and_indexer(
        n_periods, n_choices, n_exog_states
    )

    period = 0
    exog_state = 0
    state = np.array([period, lagged_choice, exog_state])
    choice_set = get_state_specific_feasible_choice_set(
        state, map_state_to_state_space_index
    )

    # retirement (lagged_choice == 1) is absorbing
    expected_choice_set = np.arange(n_choices) if lagged_choice == 0 else np.array([1])

    assert np.allclose(choice_set, expected_choice_set)
