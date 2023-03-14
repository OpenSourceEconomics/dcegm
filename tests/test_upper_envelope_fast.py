from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dcegm.pre_processing import calc_current_value
from dcegm.upper_envelope import upper_envelope
from dcegm.upper_envelope_fast import fast_upper_envelope
from dcegm.upper_envelope_fast_org import fast_upper_envelope_wrapper_org
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


# @pytest.mark.skip
def test_fast_upper_envelope_against_org_code():
    policy_egm = np.genfromtxt(TEST_RESOURCES_DIR / "pol10.csv", delimiter=",")
    value_egm = np.genfromtxt(TEST_RESOURCES_DIR / "val10.csv", delimiter=",")

    choice = 0
    max_wealth = 50
    n_grid_wealth = 500
    exogenous_savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    _index = pd.MultiIndex.from_tuples(
        [("utility_function", "theta"), ("delta", "delta")],
        names=["category", "name"],
    )
    params = pd.DataFrame(data=[1.95, 0.35], columns=["value"], index=_index)
    discount_factor = 1.95

    compute_utility = partial(utility_func_crra, params=params)
    compute_value = partial(
        calc_current_value,
        discount_factor=discount_factor,
        compute_utility=compute_utility,
    )

    endog_grid_refined, value_refined, policy_refined = fast_upper_envelope(
        endog_grid=policy_egm[0],
        value=value_egm[1],
        policy=policy_egm[1],
        exog_grid=np.append(0, exogenous_savings_grid),
    )

    policy_org, value_org = fast_upper_envelope_wrapper_org(
        policy=policy_egm,
        value=value_egm,
        exog_grid=exogenous_savings_grid,
        choice=choice,
        compute_value=compute_value,
    )

    policy_expected = policy_org[:, ~np.isnan(policy_org).any(axis=0)]
    value_expected = value_org[
        :,
        ~np.isnan(value_org).any(axis=0),
    ]

    assert np.all(np.in1d(value_expected[0, :], endog_grid_refined))
    assert np.all(np.in1d(value_expected[1, :], value_refined))
    assert np.all(np.in1d(policy_expected[0, :], endog_grid_refined))
    assert np.all(np.in1d(policy_expected[1, :], policy_refined))


def test_fast_upper_envelope_against_fedor():
    policy_egm = np.genfromtxt(TEST_RESOURCES_DIR / "pol10.csv", delimiter=",")
    policy_fedor = np.genfromtxt(TEST_RESOURCES_DIR / "expec_pol10.csv", delimiter=",")

    value_egm = np.genfromtxt(TEST_RESOURCES_DIR / "val10.csv", delimiter=",")
    value_fedor = np.genfromtxt(TEST_RESOURCES_DIR / "expec_val10.csv", delimiter=",")

    choice = 0
    max_wealth = 50
    n_grid_wealth = 500
    exogenous_savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    _index = pd.MultiIndex.from_tuples(
        [("utility_function", "theta"), ("delta", "delta")],
        names=["category", "name"],
    )
    params = pd.DataFrame(data=[1.95, 0.35], columns=["value"], index=_index)
    discount_factor = 1.95

    compute_utility = partial(utility_func_crra, params=params)
    compute_value = partial(
        calc_current_value,
        discount_factor=discount_factor,
        compute_utility=compute_utility,
    )

    endog_grid_refined, value_refined, policy_refined = fast_upper_envelope(
        endog_grid=policy_egm[0],
        value=value_egm[1],
        policy=policy_egm[1],
        exog_grid=np.append(0, exogenous_savings_grid),
    )

    _policy_refined_fedor, _value_refine_fedor = upper_envelope(  # noqa: U100
        policy=policy_egm,
        value=value_egm,
        exog_grid=exogenous_savings_grid,
        choice=choice,
        compute_value=compute_value,
    )

    policy_expected = policy_fedor[:, ~np.isnan(policy_fedor).any(axis=0)]  # noqa: F841
    value_expected = value_fedor[  # noqa: F841
        :,
        ~np.isnan(value_fedor).any(axis=0),
    ]

    # np.savetxt(
    #     "plot_fues_against_fedor_policy_10_fues.csv",
    #     policy_got,
    #     delimiter="," # noqa: E800
    # ) # noqa: E800
    # np.savetxt(
    #     "plot_fues_against_fedor_policy_10_fedor.csv",
    #     policy_expected, # noqa: E800
    #     delimiter="," # noqa: E800
    # ) # noqa: E800

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(value_expected[0][120:140], value_expected[1][120:140], "o", c="g", ms=0.5)
    ax.set_title("refined - Fedor")
    ax.set_xlabel("$m_t$")
    ax.set_ylabel("$c_t$")
    fig.savefig("fedor_pol10.png", dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(endog_grid_refined[120:140], value_refined[120:140], "o", ms=0.5)
    ax.set_title("refined - FUES")
    ax.set_xlabel("$m_t$")
    ax.set_ylabel("$c_t$")
    fig.savefig("fues_pol10.png", dpi=300)

    # In Fedor's upper envelope, there are two endogenous wealth grids;
    # one for the value function and a longer one for the policy function.
    # Since we want to unify the two endogoenous grids and want the refined value and
    # policy array to be of equal length, our refined value function is longer than
    # Fedor's.
    # Hence, we interpolate Fedor's refined value function to our refined grid.
    value_expected_interp = np.interp(
        endog_grid_refined, value_expected[0], value_expected[1]
    )
    aaae(value_refined, value_expected_interp)
    aaae(policy_refined, policy_expected[1])
    aaae(endog_grid_refined, policy_expected[0])
