from itertools import product

import numpy as np
import pytest
from dcegm.state_space import get_child_states
from toy_models.state_space_objects import create_state_space
from toy_models.state_space_objects import get_state_specific_choice_set


def expected_state_space_and_indexer(n_periods, n_choices):
    _periods = np.arange(n_periods)
    _choices = np.arange(n_choices)
    state_space = np.column_stack(
        [np.repeat(_periods, n_choices), np.tile(_choices, n_periods)]
    )
    state_indexer = np.arange(n_periods * n_choices).reshape(n_periods, n_choices)

    return state_space, state_indexer


num_periods = [15, 25, 63, 100]
num_choices = [2, 3, 20, 50]
lagged_choices = [0, 1]

TEST_CASES = list(product(num_periods, num_choices))


@pytest.mark.parametrize("n_periods, n_choices", TEST_CASES)
def test_state_space(n_periods, n_choices):
    options = {"n_periods": n_periods, "n_discrete_choices": n_choices}

    state_space, state_indexer = create_state_space(options)

    expected_state_space, expected_state_indexer = expected_state_space_and_indexer(
        n_periods, n_choices
    )

    np.allclose(state_space, expected_state_space)
    np.allclose(state_indexer, expected_state_indexer)


TEST_CASES = list(product(lagged_choices, num_periods, num_choices))


@pytest.mark.parametrize("lagged_choice, n_periods, n_choices", TEST_CASES)
def test_state_choice_set(lagged_choice, n_periods, n_choices):
    state_space, state_indexer = expected_state_space_and_indexer(n_periods, n_choices)

    period = 0
    state = np.array([period, lagged_choice])
    choice_set = get_state_specific_choice_set(state, state_space, state_indexer)

    np.allclose(choice_set, np.arange(n_choices))


@pytest.mark.parametrize("lagged_choice, n_periods, n_choices", TEST_CASES)
def test_get_child_states(lagged_choice, n_periods, n_choices):
    state_space, state_indexer = expected_state_space_and_indexer(n_periods, n_choices)

    n_admissible_choices = n_choices

    period = 0
    state = np.array([period, lagged_choice])

    child_nodes = get_child_states(
        state,
        state_space,
        state_indexer,
        get_choice_set_by_state=get_state_specific_choice_set,
    )

    expected_child_nodes = np.atleast_2d(
        np.column_stack(
            [
                np.repeat(period + 1, n_admissible_choices),
                np.arange(n_admissible_choices),
            ]
        )
    )
    np.allclose(child_nodes, expected_child_nodes)
