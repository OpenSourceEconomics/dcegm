import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dcegm.pre_processing.check_options import check_options_and_set_defaults


@pytest.fixture
def valid_options():
    """Fixture providing a valid options dictionary."""
    return {
        "state_space": {
            "n_periods": 5,
            "choices": [1, 2, 3],
            "endogenous_states": {
                "education": np.arange(2, dtype=int),
            },
            "continuous_states": {
                "wealth": np.linspace(0, 10, 11),
                "experience": np.linspace(0, 5, 6),
            },
            "exogenous_processes": {
                "health": {
                    "transition": lambda x: x,
                    "states": np.arange(3, dtype=int),
                },
            },
        },
        "model_params": {},
        "tuning_params": {},
    }


def test_invalid_options_type():
    with pytest.raises(ValueError, match="Options must be a dictionary."):
        check_options_and_set_defaults([])


def test_missing_state_space(valid_options):
    del valid_options["state_space"]
    with pytest.raises(
        ValueError, match="Options must contain a state space dictionary."
    ):
        check_options_and_set_defaults(valid_options)


def test_invalid_state_space_type():
    with pytest.raises(ValueError, match="State space must be a dictionary."):
        check_options_and_set_defaults({"state_space": "not_a_dict"})


def test_missing_n_periods(valid_options):
    del valid_options["state_space"]["n_periods"]
    with pytest.raises(
        ValueError, match="State space must contain the number of periods."
    ):
        check_options_and_set_defaults(valid_options)


def test_invalid_n_periods_type(valid_options):
    valid_options["state_space"]["n_periods"] = "not_an_int"
    with pytest.raises(ValueError, match="Number of periods must be an integer."):
        check_options_and_set_defaults(valid_options)


def test_invalid_n_periods_value(valid_options):
    valid_options["state_space"]["n_periods"] = 1
    with pytest.raises(ValueError, match="Number of periods must be greater than 1."):
        check_options_and_set_defaults(valid_options)


@pytest.mark.parametrize(
    "choices, expected_array",
    [
        ([1, 2, 3], np.array([1, 2, 3], dtype=np.uint8)),
        (5, np.array([5], dtype=np.uint8)),
        (np.array([1, 2, 3]), np.array([1, 2, 3], dtype=np.uint8)),
    ],
)
def test_valid_choices_conversion(valid_options, choices, expected_array):
    valid_options["state_space"]["choices"] = choices
    options = check_options_and_set_defaults(valid_options)
    assert_array_equal(options["state_space"]["choices"], expected_array)


def test_missing_sinlge_choice(valid_options):
    del valid_options["state_space"]["choices"]
    options = check_options_and_set_defaults(valid_options)
    assert_array_equal(options["state_space"]["choices"], np.array([0], dtype=np.uint8))


def test_invalid_choices_type(valid_options):
    valid_options["state_space"]["choices"] = "not_a_valid_type"
    with pytest.raises(ValueError, match="Choices must be a list or an integer."):
        check_options_and_set_defaults(valid_options)


def test_missing_model_params(valid_options):
    del valid_options["model_params"]
    with pytest.raises(
        ValueError, match="Options must contain a model parameters dictionary."
    ):
        check_options_and_set_defaults(valid_options)


def test_invalid_model_params_type(valid_options):
    valid_options["model_params"] = "not_a_dict"
    with pytest.raises(ValueError, match="Model parameters must be a dictionary."):
        check_options_and_set_defaults(valid_options)


# Maybe also check this in the check_options_and_set_defaults function
def test_missing_continuous_states(valid_options):
    del valid_options["state_space"]["continuous_states"]
    with pytest.raises(KeyError):
        check_options_and_set_defaults(valid_options)


def test_tuning_params_defaults(valid_options):
    del valid_options["tuning_params"]
    options = check_options_and_set_defaults(valid_options)
    assert options["tuning_params"]["extra_wealth_grid_factor"] == 0.2
    assert options["tuning_params"]["n_constrained_points_to_add"] == 1


def test_tuning_params_invalid_grid_factors(valid_options):
    valid_options["tuning_params"]["extra_wealth_grid_factor"] = 0.01
    valid_options["tuning_params"]["n_constrained_points_to_add"] = 100
    with pytest.raises(
        ValueError, match="The extra wealth grid factor .* is too small"
    ):
        check_options_and_set_defaults(valid_options)


def test_second_continuous_state_handling(valid_options):
    options = check_options_and_set_defaults(valid_options)
    assert options["second_continuous_state_name"] == "experience"
    assert options["tuning_params"]["n_second_continuous_grid"] == len(
        valid_options["state_space"]["continuous_states"]["experience"]
    )
