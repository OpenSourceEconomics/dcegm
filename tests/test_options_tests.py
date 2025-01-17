import numpy as np
import pytest

from dcegm.pre_processing.check_options import check_options_and_set_defaults


@pytest.fixture
def valid_options():
    """Fixture providing a valid options dictionary."""
    return {
        "state_space": {
            "n_periods": 5,
            "choices": 3,
            "endogenous_states": {
                "education": 2,
            },
            "continuous_states": {
                "wealth": np.linspace(0, 10, 11),
                "experience": np.linspace(0, 5, 6),
            },
            "exogenous_processes": {
                "health": {
                    "transition": lambda x: x,
                    "states": 3,
                },
            },
        },
        "model_params": {
            "n_choices": 3,
        },
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
        (3, np.array([0, 1, 2], dtype=np.uint8)),
        (5, np.arange(5, dtype=np.uint8)),
    ],
)
def test_valid_choices_conversion(valid_options, choices, expected_array):
    valid_options["state_space"]["choices"] = choices
    if choices == 5:
        valid_options["model_params"]["n_choices"] = 5
    options = check_options_and_set_defaults(valid_options)
    np.testing.assert_array_equal(options["state_space"]["choices"], expected_array)


def test_invalid_choices_type(valid_options):
    valid_options["state_space"]["choices"] = "not_a_valid_type"
    with pytest.raises(ValueError, match="choices must be an integer."):
        check_options_and_set_defaults(valid_options)
    valid_options["state_space"]["choices"] = [1, 2, 3]
    with pytest.raises(ValueError, match="choices must be an integer."):
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


def test_invalid_endogenous_states_type(valid_options):
    valid_options["state_space"]["endogenous_states"] = "not_a_dict"
    with pytest.raises(
        ValueError,
        match="endogenous_states specified in the options must be a dictionary.",
    ):
        check_options_and_set_defaults(valid_options)


def test_invalid_endogenous_state_value_type(valid_options):
    valid_options["state_space"]["endogenous_states"]["education"] = "not_an_int"
    with pytest.raises(ValueError, match="education value must be an integer."):
        check_options_and_set_defaults(valid_options)


def test_invalid_exogenous_processes_type(valid_options):
    valid_options["state_space"]["exogenous_processes"] = "not_a_dict"
    with pytest.raises(ValueError, match="Exogenous processes must be a dictionary."):
        check_options_and_set_defaults(valid_options)


def test_invalid_exogenous_process_structure(valid_options):
    valid_options["state_space"]["exogenous_processes"]["health"] = "not_a_dict"
    with pytest.raises(
        ValueError,
        match="health value must be a dictionary in the options exogenous processes dictionary",
    ):
        check_options_and_set_defaults(valid_options)


def test_missing_transition_in_exogenous_process(valid_options):
    del valid_options["state_space"]["exogenous_processes"]["health"]["transition"]
    with pytest.raises(
        ValueError,
        match="health must contain a transition function in the options exogenous processes dictionary.",
    ):
        check_options_and_set_defaults(valid_options)


def test_missing_states_in_exogenous_process(valid_options):
    del valid_options["state_space"]["exogenous_processes"]["health"]["states"]
    with pytest.raises(
        ValueError,
        match="health must contain states in the options exogenous processes dictionary.",
    ):
        check_options_and_set_defaults(valid_options)


def test_invalid_states_type_in_exogenous_process(valid_options):
    valid_options["state_space"]["exogenous_processes"]["health"][
        "states"
    ] = "not_an_int"
    with pytest.raises(
        ValueError,
        match="health states must be an int in the options exogenous processes dictionary.",
    ):
        check_options_and_set_defaults(valid_options)


def test_missing_continuous_states(valid_options):
    del valid_options["state_space"]["continuous_states"]
    with pytest.raises(ValueError, match="State space must contain continuous states."):
        check_options_and_set_defaults(valid_options)


def test_missing_wealth_in_continuous_states(valid_options):
    del valid_options["state_space"]["continuous_states"]["wealth"]
    with pytest.raises(ValueError, match="Continuous states must contain wealth."):
        check_options_and_set_defaults(valid_options)


def test_invalid_wealth_type(valid_options):
    valid_options["state_space"]["continuous_states"]["wealth"] = "not_a_list_or_array"
    with pytest.raises(
        ValueError,
        match="Wealth must be a list, numpy array or jax numpy array is .* instead.",
    ):
        check_options_and_set_defaults(valid_options)


def test_invalid_continuous_state_type(valid_options):
    valid_options["state_space"]["continuous_states"][
        "experience"
    ] = "not_a_list_or_array"
    with pytest.raises(
        ValueError,
        match="experience must be a list, numpy array or jax numpy array is .* instead.",
    ):
        check_options_and_set_defaults(valid_options)


def test_valid_endogenous_states_conversion(valid_options):
    valid_options["state_space"]["endogenous_states"]["education"] = 3
    options = check_options_and_set_defaults(valid_options)
    np.testing.assert_array_equal(
        options["state_space"]["endogenous_states"]["education"],
        np.arange(3, dtype=np.uint8),
    )


def test_valid_exogenous_states_conversion(valid_options):
    valid_options["state_space"]["exogenous_processes"]["health"]["states"] = 4
    options = check_options_and_set_defaults(valid_options)
    np.testing.assert_array_equal(
        options["state_space"]["exogenous_processes"]["health"]["states"],
        np.arange(4, dtype=np.uint8),
    )


def test_missing_n_choices_in_model_params(valid_options):
    del valid_options["model_params"]["n_choices"]
    options = check_options_and_set_defaults(valid_options)
    assert (
        options["model_params"]["n_choices"]
        == options["state_space"]["choices"].shape[0]
    )


def test_mismatched_n_choices_warning(valid_options):
    valid_options["model_params"]["n_choices"] = 2  # Intentionally mismatched
    with pytest.warns(
        UserWarning,
        match="n_choices in the options model_params dictionary.*is not equal to the number of choices.*",
    ):
        options = check_options_and_set_defaults(valid_options)
    assert (
        options["model_params"]["n_choices"]
        == options["state_space"]["choices"].shape[0]
    )


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
