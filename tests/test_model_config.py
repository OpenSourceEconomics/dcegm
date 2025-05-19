import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dcegm.pre_processing.check_options import check_model_config_and_process


@pytest.fixture
def valid_model_config():
    """Fixture providing a valid options dictionary."""
    return {
        "n_periods": 5,
        "choices": [1, 2, 3],
        "deterministic_states": {
            "education": np.arange(2, dtype=int),
        },
        "continuous_states": {
            "assets_end_of_period": np.linspace(0, 10, 11),
            "experience": np.linspace(0, 5, 6),
        },
        "stochastic_states": {
            "health": {
                "transition": lambda x: x,
                "states": np.arange(3, dtype=int),
            },
        },
        "n_quad_points": 5,
        "tuning_params": {},
    }


def test_invalid_options_type():
    with pytest.raises(ValueError, match="model_config must be a dictionary."):
        check_model_config_and_process([])


def test_missing_n_periods(valid_model_config):
    del valid_model_config["n_periods"]
    with pytest.raises(ValueError, match="model_config must contain n_periods."):
        check_model_config_and_process(valid_model_config)


def test_invalid_n_periods_type(valid_model_config):
    valid_model_config["n_periods"] = "not_an_int"
    with pytest.raises(ValueError, match="Number of periods must be an integer."):
        check_model_config_and_process(valid_model_config)


def test_invalid_n_periods_value(valid_model_config):
    valid_model_config["n_periods"] = 1
    with pytest.raises(ValueError, match="Number of periods must be greater than 1."):
        check_model_config_and_process(valid_model_config)


@pytest.mark.parametrize(
    "choices, expected_array",
    [
        ([1, 2, 3], np.array([1, 2, 3], dtype=np.uint8)),
        (5, np.array([5], dtype=np.uint8)),
        (np.array([1, 2, 3]), np.array([1, 2, 3], dtype=np.uint8)),
    ],
)
def test_valid_choices_conversion(valid_model_config, choices, expected_array):
    valid_model_config["choices"] = choices
    options = check_model_config_and_process(valid_model_config)
    assert_array_equal(options["choices"], expected_array)


def test_missing_sinlge_choice(valid_model_config):
    del valid_model_config["choices"]
    model_config = check_model_config_and_process(valid_model_config)
    assert_array_equal(model_config["choices"], np.array([0], dtype=np.uint8))


def test_invalid_choices_type(valid_model_config):
    valid_model_config["choices"] = "not_a_valid_type"
    with pytest.raises(ValueError, match="Choices must be a list or an integer."):
        check_model_config_and_process(valid_model_config)


def test_missing_continuous_states(valid_model_config):
    del valid_model_config["continuous_states"]
    with pytest.raises(KeyError):
        check_model_config_and_process(valid_model_config)


def test_tuning_params_defaults(valid_model_config):
    del valid_model_config["tuning_params"]
    options = check_model_config_and_process(valid_model_config)
    assert options["tuning_params"]["extra_wealth_grid_factor"] == 0.2
    assert options["tuning_params"]["n_constrained_points_to_add"] == 1


def test_tuning_params_invalid_grid_factors(valid_model_config):
    valid_model_config["tuning_params"]["extra_wealth_grid_factor"] = 0.01
    valid_model_config["tuning_params"]["n_constrained_points_to_add"] = 100
    with pytest.raises(
        ValueError, match="The extra wealth grid factor .* is too small"
    ):
        check_model_config_and_process(valid_model_config)


def test_second_continuous_state_handling(valid_model_config):
    options = check_model_config_and_process(valid_model_config)
    assert options["second_continuous_state_name"] == "experience"
    assert options["tuning_params"]["n_second_continuous_grid"] == len(
        valid_model_config["continuous_states"]["experience"]
    )
