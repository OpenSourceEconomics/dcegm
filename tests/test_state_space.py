import numpy as np
import pytest
from dcegm.state_space import create_state_space


@pytest.mark.parametrize(
    "n_periods, n_choices", [(25, 2), [100, 3], [20, 20], [15, 50]]
)
def test_state_space(n_periods, n_choices):
    options = {"n_periods": n_periods, "n_discrete_choices": n_choices}

    _periods = np.arange(n_periods)
    _choices = np.arange(n_choices)
    state_space_expected = np.column_stack(
        [np.repeat(_periods, n_choices), np.tile(_choices, n_periods)]
    )
    state_indexer_expected = np.arange(n_periods * n_choices).reshape(
        n_periods, n_choices
    )

    state_space, state_indexer = create_state_space(options)

    np.allclose(state_space, state_space_expected)
    np.allclose(state_indexer, state_indexer_expected)
