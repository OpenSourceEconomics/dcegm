from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest
from dcegm.pre_processing.state_space import inspect_state_space
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_feasible_choice_set,
)


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


def test_inspect_state_space(options):
    inspect_state_space(options=options)
