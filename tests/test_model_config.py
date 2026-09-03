import numpy as np
import pytest
from numpy.testing import assert_array_equal

from dcegm.pre_processing.check_model_config import check_model_config_and_process


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
        "upper_envelope": {"tuning_params": {}},
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
    with pytest.raises(ValueError, match="Choices must be given in model_config."):
        check_model_config_and_process(valid_model_config)


def test_invalid_choices_type(valid_model_config):
    valid_model_config["choices"] = "not_a_valid_type"
    with pytest.raises(ValueError, match="Choices must be a list or an integer."):
        check_model_config_and_process(valid_model_config)


def test_missing_continuous_states(valid_model_config):
    del valid_model_config["continuous_states"]
    with pytest.raises(
        ValueError, match="model_config must contain continuous_states as key."
    ):
        check_model_config_and_process(valid_model_config)


def test_tuning_params_defaults(valid_model_config):
    del valid_model_config["upper_envelope"]["tuning_params"]
    options = check_model_config_and_process(valid_model_config)
    assert options["upper_envelope"]["tuning_params"]["extra_wealth_grid_factor"] == 0.2
    assert (
        options["upper_envelope"]["tuning_params"]["n_constrained_points_to_add"] == 1
    )


def test_tuning_params_invalid_grid_factors(valid_model_config):
    valid_model_config["upper_envelope"]["tuning_params"][
        "extra_wealth_grid_factor"
    ] = 0.01
    valid_model_config["upper_envelope"]["tuning_params"][
        "n_constrained_points_to_add"
    ] = 100
    with pytest.raises(
        ValueError, match="The extra wealth grid factor .* is too small"
    ):
        check_model_config_and_process(valid_model_config)


def test_second_continuous_state_handling(valid_model_config):
    processed_model_config = check_model_config_and_process(valid_model_config)
    continuous_states_info = processed_model_config["continuous_states_info"]

    assert continuous_states_info["additional_continuous_state_names"] == ["experience"]
    assert len(
        continuous_states_info["additional_continuous_state_grids"]["experience"]
    ) == len(valid_model_config["continuous_states"]["experience"])


def test_upper_envelope_method_default(valid_model_config):
    assert (
        check_model_config_and_process(valid_model_config)["upper_envelope"]["method"]
        == "fues"
    )


def test_skip_endog_grid_storage_false_for_fues(valid_model_config):
    options = check_model_config_and_process(valid_model_config)
    assert options["upper_envelope"]["skip_endog_grid_storage"] is False


def test_fues_rejects_assets_begin_of_period(valid_model_config):
    # assets_begin_of_period is only meaningful for druedahl_jorgensen; declaring
    # it together with fues (the default method here) used to be silently
    # ignored rather than rejected.
    valid_model_config["continuous_states"]["assets_begin_of_period"] = np.linspace(
        0, 10, 11
    )
    with pytest.raises(ValueError, match="only used by the 'druedahl_jorgensen'"):
        check_model_config_and_process(valid_model_config)


def test_dj_wealth_grid_and_skip_flag(valid_model_config):
    valid_model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}
    valid_model_config["continuous_states"]["assets_begin_of_period"] = np.linspace(
        0, 10, 11
    )
    options = check_model_config_and_process(valid_model_config)

    dj_wealth_grid = options["continuous_states_info"]["dj_wealth_grid"]
    expected = np.concatenate(
        ([0.0], valid_model_config["continuous_states"]["assets_begin_of_period"])
    )
    assert_array_equal(np.asarray(dj_wealth_grid), expected)
    assert dj_wealth_grid.shape[0] == options["n_total_wealth_grid"]
    assert options["upper_envelope"]["skip_endog_grid_storage"] is True


def test_none_assets_begin_of_period_defers_dj_wealth_grid(valid_model_config):
    # `None` means the grid is fully state-choice-specific -- no default array to
    # read a length from, so dj_wealth_grid/n_total_wealth_grid are left unresolved
    # (None) here, pinned later once a real state-choice can be evaluated against.
    valid_model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}
    valid_model_config["continuous_states"]["assets_begin_of_period"] = None
    options = check_model_config_and_process(valid_model_config)

    assert options["continuous_states_info"]["assets_begin_of_period"] is None
    assert options["continuous_states_info"]["dj_wealth_grid"] is None
    assert options["n_total_wealth_grid"] is None
    # skip_endog_grid_storage itself only depends on method/n_choices, not on
    # whether the grid is resolved yet, so it's still known immediately.
    assert options["upper_envelope"]["skip_endog_grid_storage"] is True


def test_skip_endog_grid_storage_false_for_single_choice_dj(valid_model_config):
    # With a single discrete choice, the upper envelope is skipped entirely, so the
    # stored endog_grid is not the fixed Druedahl-Jorgensen grid and must still be
    # stored/read normally, even though the method is "druedahl_jorgensen".
    valid_model_config["choices"] = [1]
    valid_model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}
    valid_model_config["continuous_states"]["assets_begin_of_period"] = np.linspace(
        0, 10, 11
    )
    options = check_model_config_and_process(valid_model_config)
    assert options["upper_envelope"]["skip_endog_grid_storage"] is False
