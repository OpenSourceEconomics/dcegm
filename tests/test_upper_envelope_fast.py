from __future__ import annotations

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from dcegm.pre_processing import calc_current_value
from dcegm.upper_envelope import upper_envelope
from dcegm.upper_envelope_fast import _augment_grid
from dcegm.upper_envelope_fast import fast_upper_envelope
from dcegm.upper_envelope_fast_org import fast_upper_envelope_wrapper_org
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


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
    discount_factor = 0.95

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


@pytest.mark.parametrize("period", [18, 9])
def test_fast_upper_envelope_against_fedor_credit_constrained_passes(period):
    policy_egm = np.genfromtxt(TEST_RESOURCES_DIR / f"pol{period}.csv", delimiter=",")
    policy_fedor = np.genfromtxt(
        TEST_RESOURCES_DIR / f"expec_pol{period}.csv", delimiter=","
    )

    value_egm = np.genfromtxt(TEST_RESOURCES_DIR / f"val{period}.csv", delimiter=",")
    value_fedor = np.genfromtxt(
        TEST_RESOURCES_DIR / f"expec_val{period}.csv", delimiter=","
    )
    # endog_grid_egm = policy_egm[0]

    # policy_egm_augmented_fedor = np.genfromtxt(
    #     TEST_RESOURCES_DIR / f"policy{period}_augment.csv", delimiter=","
    # )
    # value_egm_augmented_fedor = np.genfromtxt(
    #     TEST_RESOURCES_DIR / f"value{period}_augment.csv", delimiter=","
    # )
    # # policy_egm_augmented = policy_egm_augmented_
    # # value_egm_augmented = value_egm_augmented_
    # endog_grid_egm_augmented = policy_egm_augmented_fedor[0]

    choice = 0
    max_wealth = 50
    n_grid_wealth = 500
    exogenous_savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    _index = pd.MultiIndex.from_tuples(
        [("utility_function", "theta"), ("delta", "delta")],
        names=["category", "name"],
    )
    params = pd.DataFrame(data=[1.95, 0.35], columns=["value"], index=_index)
    discount_factor = 0.95

    compute_utility = partial(utility_func_crra, params=params)
    compute_value = partial(
        calc_current_value,
        discount_factor=discount_factor,
        compute_utility=compute_utility,
    )

    # # ================================================================================
    min_wealth_grid = np.min(value_egm[0, 1:])
    # # if value_egm[0, 1] >min_wealth_grid:
    # # Non-concave region coincides with credit constraint.
    # # This happens when there is a non-monotonicity in the endogenous wealth grid
    # # that goes below the first point.
    # # Solution: Value function to the left of the first point is analytical,
    # # so we just need to add some points to the left of the first grid point.

    expected_value_zero_wealth = value_egm[1, 0]

    policy_egm_augmented, value_egm_augmented = _augment_grid(
        policy_egm,
        value_egm,
        choice,
        expected_value_zero_wealth,
        min_wealth_grid,
        n_grid_wealth,
        compute_value,
    )
    endog_grid_egm_augmented = policy_egm_augmented[0]

    # aaae(policy_egm_augmented, policy_egm_augmented_fedor)
    # aaae(value_egm_augmented, value_egm_augmented_fedor)
    # # ================================================================================

    exog_grid_augmented = np.linspace(
        exogenous_savings_grid[1], exogenous_savings_grid[2], n_grid_wealth // 10 + 1
    )
    endog_grid_refined, value_refined, policy_refined = fast_upper_envelope(
        endog_grid=np.append(0, endog_grid_egm_augmented),
        value=np.append(value_egm[1, 0], value_egm_augmented[1]),
        policy=np.append(policy_egm[1, 0], policy_egm_augmented[1]),
        exog_grid=np.append(
            [0], np.append(exog_grid_augmented, exogenous_savings_grid[2:])
        ),
    )
    # endog_grid_refined, value_refined, policy_refined = fast_upper_envelope(
    #     endog_grid=endog_grid_egm,
    #     value=value_egm[1],
    #     policy=policy_egm[1],
    #     exog_grid=np.append(0, exogenous_savings_grid),
    # )

    _policy_refined_fedor, _value_refine_fedor = upper_envelope(  # noqa: U100
        policy=policy_egm,
        value=value_egm,
        exog_grid=exogenous_savings_grid,
        choice=choice,
        compute_value=compute_value,
        period=9,
    )

    policy_expected = policy_fedor[:, ~np.isnan(policy_fedor).any(axis=0)]  # noqa: F841
    value_expected = value_fedor[  # noqa: F841
        :,
        ~np.isnan(value_fedor).any(axis=0),
    ]
    # breakpoint()

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
    ax.plot(value_expected[0], value_expected[1], "o", c="g", ms=0.5)
    ax.set_title("refined - Fedor")
    ax.set_xlabel("$m_t$")
    ax.set_ylabel("$c_t$")
    fig.savefig(f"fedor_val{period}.png", dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(endog_grid_refined, value_refined, "o", ms=0.5)
    ax.set_title("refined - FUES")
    ax.set_xlabel("$m_t$")
    ax.set_ylabel("$c_t$")
    fig.savefig(f"fues_val{period}.png", dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(policy_expected[0][:10], policy_expected[1][:10], "o", c="g", ms=0.5)
    ax.set_title("refined - Fedor")
    ax.set_xlabel("$m_t$")
    ax.set_ylabel("$c_t$")
    fig.savefig(f"fedor_pol{period}.png", dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(endog_grid_refined, policy_refined, "o", ms=0.5)
    ax.set_title("refined - FUES")
    ax.set_xlabel("$m_t$")
    ax.set_ylabel("$c_t$")
    fig.savefig(f"fues_pol{period}.png", dpi=300)

    # In Fedor's upper envelope, there are two endogenous wealth grids;
    # one for the value function and a longer one for the policy function.
    # Since we want to unify the two endogoenous grids and want the refined value and
    # policy array to be of equal length, our refined value function is longer than
    # Fedor's.
    # Hence, we interpolate Fedor's refined value function to our refined grid.

    aaae(endog_grid_refined, policy_expected[0])
    # breakpoint()
    aaae(policy_refined, policy_expected[1])
    value_expected_interp = np.interp(
        endog_grid_refined, value_expected[0], value_expected[1]
    )
    aaae(value_refined, value_expected_interp)


@pytest.mark.parametrize(
    "period",
    [2, 4, 10],
)
def test_fast_upper_envelope_against_fedor(period):
    policy_egm = np.genfromtxt(TEST_RESOURCES_DIR / f"pol{period}.csv", delimiter=",")
    policy_fedor = np.genfromtxt(
        TEST_RESOURCES_DIR / f"expec_pol{period}.csv", delimiter=","
    )

    value_egm = np.genfromtxt(TEST_RESOURCES_DIR / f"val{period}.csv", delimiter=",")
    value_fedor = np.genfromtxt(
        TEST_RESOURCES_DIR / f"expec_val{period}.csv", delimiter=","
    )

    choice = 0
    max_wealth = 50
    n_grid_wealth = 500
    exogenous_savings_grid = np.linspace(0, max_wealth, n_grid_wealth)

    _index = pd.MultiIndex.from_tuples(
        [("utility_function", "theta"), ("delta", "delta")],
        names=["category", "name"],
    )
    params = pd.DataFrame(data=[1.95, 0.35], columns=["value"], index=_index)
    discount_factor = 0.95

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
        period=period,
    )

    policy_expected = policy_fedor[:, ~np.isnan(policy_fedor).any(axis=0)]  # noqa: F841
    value_expected = value_fedor[  # noqa: F841
        :,
        ~np.isnan(value_fedor).any(axis=0),
    ]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(value_expected[0][300:320], value_expected[1][300:320], "o", c="g", ms=0.5)
    ax.set_title("refined - Fedor")
    ax.set_xlabel("$m_t$")
    ax.set_ylabel("$c_t$")
    fig.savefig(f"fedor_val{period}.png", dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(endog_grid_refined[300:320], value_refined[300:320], "o", ms=0.5)
    ax.set_title("refined - FUES")
    ax.set_xlabel("$m_t$")
    ax.set_ylabel("$c_t$")
    fig.savefig(f"fues_val{period}.png", dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(
        policy_expected[0][300:320], policy_expected[1][300:320], "o", c="g", ms=0.5
    )
    ax.set_title("refined - Fedor")
    ax.set_xlabel("$m_t$")
    ax.set_ylabel("$c_t$")
    fig.savefig(f"fedor_pol{period}.png", dpi=300)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(endog_grid_refined[300:320], policy_refined[300:320], "o", ms=0.5)
    ax.set_title("refined - FUES")
    ax.set_xlabel("$m_t$")
    ax.set_ylabel("$c_t$")
    fig.savefig(f"fues_pol{period}.png", dpi=300)

    # In Fedor's upper envelope, there are two endogenous wealth grids;
    # one for the value function and a longer one for the policy function.
    # Since we want to unify the two endogoenous grids and want the refined value and
    # policy array to be of equal length, our refined value function is longer than
    # Fedor's.
    # Hence, we interpolate Fedor's refined value function to our refined grid.
    aaae(endog_grid_refined, policy_expected[0])
    aaae(policy_refined, policy_expected[1])
    value_expected_interp = np.interp(
        endog_grid_refined, value_expected[0], value_expected[1]
    )
    aaae(value_refined, value_expected_interp)
