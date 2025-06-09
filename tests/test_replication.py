import pickle
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from jax import config
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models
from tests.utils.interp1d_auxiliary import (
    linear_interpolation_with_extrapolation,
)

# Obtain the test directory of the package
TEST_DIR = Path(__file__).parent

REPLICATION_TEST_RESOURCES_DIR = TEST_DIR / "resources" / "replication_tests"


def debug_plot_overlay(policy_expec, value_expec, policy_calc, value_calc):
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for ax, data_expec, data_calc, title in zip(
        axes,
        [policy_expec, value_expec],
        [policy_calc, value_calc],
        ["Policy Function", "Value Function"],
    ):
        mask = (data_expec[0, :] >= 0) & (data_expec[0, :] <= 75)
        indices = np.where(mask)[0]

        # Plot expected
        ax.plot(
            data_expec[0, indices],
            data_expec[1, indices],
            linestyle="--",
            color="blue",
            label="Expected",
            alpha=0.5,
        )
        ax.scatter(
            data_expec[0, indices],
            data_expec[1, indices],
            color="blue",
            s=50,
            alpha=0.7,
        )

        # Plot calculated
        ax.plot(
            data_calc[0, indices],
            data_calc[1, indices],
            linestyle="-",
            color="orange",
            label="Calculated",
            alpha=0.5,
        )
        ax.scatter(
            data_calc[0, indices],
            data_calc[1, indices],
            color="orange",
            s=25,
            alpha=1,
        )

        ax.set_title(title)
        ax.set_xlabel("Wealth")
        ax.legend()

    axes[0].set_ylabel("Policy / Value")
    plt.tight_layout()
    plt.show()


@pytest.mark.parametrize(
    "model_name",
    [
        "retirement_no_shocks",
        "retirement_with_shocks",
        "deaton",
    ],
)
def test_benchmark_models(model_name):
    if model_name == "deaton":
        model_funcs = toy_models.load_example_model_functions("dcegm_paper_deaton")
    else:
        model_funcs = toy_models.load_example_model_functions("dcegm_paper")

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

    policy_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / f"{model_name}" / "policy.pkl").open("rb")
    )
    value_expected = pickle.load(
        (REPLICATION_TEST_RESOURCES_DIR / f"{model_name}" / "value.pkl").open("rb")
    )
    state_choice_space = model.model_structure["state_choice_space"]
    state_choice_space_to_test = state_choice_space[state_choice_space[:, 0] < 24]

    for state_choice_idx in range(state_choice_space_to_test.shape[0] - 1, -1, -1):

        choice = state_choice_space_to_test[state_choice_idx, -1]
        period = state_choice_space_to_test[state_choice_idx, 0]
        lagged_choice = state_choice_space_to_test[state_choice_idx, 1]

        if model_name == "deaton":
            policy_expec = policy_expected[period, choice]
            value_expec = value_expected[period, choice]
        else:
            policy_expec = policy_expected[period][1 - choice].T
            value_expec = value_expected[period][1 - choice].T

        endo_grid, policy_calc, value_calc = (
            model_solved.get_solution_for_discrete_state_choice(
                states={
                    "period": period,
                    "lagged_choice": lagged_choice,
                },
                choice=choice,
            )
        )
        first_nan_idx = np.where(np.isnan(endo_grid))[0][0]
        policy_calc = np.vstack((endo_grid, policy_calc))[:, :first_nan_idx]
        value_calc = np.vstack((endo_grid, value_calc))[:, :first_nan_idx]

        wealth_grid_to_test = jnp.linspace(
            policy_expec[0][1], policy_expec[0][-1] + 10, 1000
        )

        value_expec_interp = linear_interpolation_with_extrapolation(
            x_new=wealth_grid_to_test, x=value_expec[0], y=value_expec[1]
        )
        policy_expec_interp = linear_interpolation_with_extrapolation(
            x_new=wealth_grid_to_test, x=policy_expec[0], y=policy_expec[1]
        )
        value_calc_interp = linear_interpolation_with_extrapolation(
            x_new=wealth_grid_to_test, x=value_calc[0], y=value_calc[1]
        )
        policy_calc_interp = linear_interpolation_with_extrapolation(
            x_new=wealth_grid_to_test, x=policy_calc[0], y=policy_calc[1]
        )

        # TL;DR: I created a bug, i think it was always there, but refactoring made it pop up.

        # debug_plot_overlay(policy_expec, value_expec, policy_calc, value_calc)
        # In some periods the last 3 or 4 endo grid points have policy and value missing, i think
        # that is because the interpolation as it is currently implemented (with jnp.interp)
        # inside of the upper envelope function does not do the exact same thing as the
        # interpolation using numpy. Or more likely i introduced a bug in the UE somewhere,
        # since there are also some magic extra points in the endo grid and i dont know where
        # they come from, but they are near the kinks and always relatively close to another
        # point (always lie exactly on the graph of the policy / value function). And starting
        # in period 4 there are also some points missing in the value function segments, root
        # cause is also that they are not added to that points segment when interpolating, even
        # though they are in the endo grid and on that segment... @Max i hate debugging JAX but
        # at least i got this horrendous version of a jax implementation to work (halfway). All
        # the tests expect one run again if i add a point just eps away from the last point to the segment
        # before interpolating on the unified grid. For period 21 this has the sideeffect that we find a new
        # value and policy function that is too good so the test fails obviously.

        # if period == 21 and choice == 0 and lagged_choice == 0:
        #      debug_plot_overlay(policy_expec, value_expec, policy_calc, value_calc)
        # #     # breakpoint()

        aaae(policy_expec_interp, policy_calc_interp)
        aaae(value_expec_interp, value_calc_interp)
