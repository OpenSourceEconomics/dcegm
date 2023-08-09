from itertools import product

import numpy as np
import pytest
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)


def expected_state_space_and_indexer(n_periods, n_choices, n_exog_processes):
    _periods = np.arange(n_periods)
    _choices = np.arange(n_choices)
    _exog_process = np.arange(n_exog_processes)
    state_space = np.column_stack(
        [
            np.repeat(_periods, n_choices * n_exog_processes),
            np.tile(np.repeat(_choices, n_exog_processes), n_periods),
            np.tile(_exog_process, n_periods * n_choices),
        ]
    )
    state_indexer = np.arange(n_periods * n_choices * n_exog_processes).reshape(
        n_periods, n_choices, n_exog_processes
    )

    return state_space, state_indexer


num_periods = [15, 25, 63, 100]
num_choices = [2, 3, 20, 50]
num_exog_states = [2, 3, 5]
lagged_choices = [0, 1]

TEST_CASES = list(product(num_periods, num_choices, num_exog_states))


@pytest.mark.parametrize("n_periods, n_choices, n_exog_states", TEST_CASES)
def test_state_space(n_periods, n_choices, n_exog_states):
    options = {
        "n_periods": n_periods,
        "n_discrete_choices": n_choices,
        "n_exog_states": n_exog_states,
    }

    state_space, state_indexer = create_state_space(options)

    expected_state_space, expected_state_indexer = expected_state_space_and_indexer(
        n_periods, n_choices, n_exog_states
    )

    np.allclose(state_space, expected_state_space)
    np.allclose(state_indexer, expected_state_indexer)


TEST_CASES = list(product(lagged_choices, num_periods, num_choices, num_exog_states))


@pytest.mark.parametrize(
    "lagged_choice, n_periods, n_choices, n_exog_states", TEST_CASES
)
def test_state_choice_set(lagged_choice, n_periods, n_choices, n_exog_states):
    state_space, state_indexer = expected_state_space_and_indexer(
        n_periods, n_choices, n_exog_states
    )

    period = 0
    exog_process = 0
    state = np.array([period, lagged_choice, exog_process])
    choice_set = get_state_specific_feasible_choice_set(
        state, state_space, state_indexer
    )

    np.allclose(choice_set, np.arange(n_choices))
