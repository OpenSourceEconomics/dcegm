from pathlib import Path

from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent


def test_partial_solve_func():
    model_funcs = toy_models.load_example_model_functions("dcegm_paper")

    model_name = "retirement_with_shocks"
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_" + model_name
        )
    )

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_funcs,
    )

    model_solved = model.solve(params)

    partial_sol = model.solve_partially(
        params=params,
        n_periods=model_config["n_periods"],
        return_candidates=True,
    )

    # Now without loop
    aaae(model_solved.policy, partial_sol["policy"])
    aaae(model_solved.value, partial_sol["value"])
    aaae(model_solved.endog_grid, partial_sol["endog_grid"])
