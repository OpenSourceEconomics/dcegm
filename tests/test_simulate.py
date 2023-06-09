import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sns
from dcegm.interpolation import linear_interpolation_with_extrapolation
from dcegm.solve import solve_dcegm
from numpy.testing import assert_array_almost_equal as aaae
from scipy.stats import norm
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.exogenous_processes import (
    get_transition_matrix_by_state,
)
from toy_models.consumption_retirement_model.final_period_solution import (
    solve_final_period_scalar,
)
from toy_models.consumption_retirement_model.simulate import simulate
from toy_models.consumption_retirement_model.simulate_stacked import simulate_stacked
from toy_models.consumption_retirement_model.state_space_objects import (
    create_state_space,
)
from toy_models.consumption_retirement_model.state_space_objects import (
    get_state_specific_choice_set,
)
from toy_models.consumption_retirement_model.utility_functions import (
    inverse_marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra
from matplotlib.backends.backend_pdf import PdfPages


from jax.config import config

config.update("jax_enable_x64", True)

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent

# Directory with additional resources for the testing harness
TEST_RESOURCES_DIR = TEST_DIR / "resources"


@pytest.fixture()
def utility_functions():
    """Return dict with utility functions."""
    return {
        "utility": utility_func_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
        "marginal_utility": marginal_utility_crra,
    }


@pytest.fixture()
def state_space_functions():
    """Return dict with utility functions."""
    return {
        "create_state_space": create_state_space,
        "get_state_specific_choice_set": get_state_specific_choice_set,
    }


def test_simulate(utility_functions, state_space_functions, load_example_model):
    # Model parameter definitions

    # Number of periods (fist period is t=1)
    num_periods = 25

    # Number of grid points over assets
    num_grid = 500

    # Maximum level of assets
    max_wealth = 50

    # Number of quadrature points used in calculation of expectations
    n_quad_points = 5

    # Number of simulations
    num_sims = 50

    # Interval of the initial wealth
    init = [10, 30]

    # Interest rate
    r = 0.05

    # Discount factor
    beta = 0.95

    # Standard deviation of log-normally distributed income shocks
    sigma = 0.00

    # Disutility of work
    cost_work = 0.35

    # CRRA coefficient (log utility if ==1)
    theta = 1.95

    # Careful with the coefficients here -- original code had the polynomial
    # Coded as a + b * x - c * x**2 ... (note the crazy minus)
    coeffs_age_poly = np.array([0.75, 0.04, -0.0002])

    # Consumption floor (safety net in retirement)
    cfloor = 0.001

    # Scale of the EV taste shocks
    lambda_ = 2.2204e-16

    model = "retirement_no_taste_shocks"

    params, options = load_example_model(f"{model}")
    options["n_exog_processes"] = 1

    state_space, indexer = create_state_space(options)

    endog_grid, policy, value = solve_dcegm(
        params,
        options,
        utility_functions,
        budget_constraint=budget_constraint,
        final_period_solution=solve_final_period_scalar,
        state_space_functions=state_space_functions,
        transition_function=get_transition_matrix_by_state,
    )

    policy_stacked = np.stack([endog_grid[::2], policy[::2]], axis=2)
    value_stacked = np.stack([endog_grid[::2], value[::2]], axis=2)

    df = simulate_stacked(
        # endog_grid=endog_grid,
        policy=policy_stacked,
        value=value_stacked,
        num_periods=num_periods,
        # cost_work,
        # theta,
        # beta,
        lambda_=lambda_,
        sigma=sigma,
        r=r,
        coeffs_age_poly=coeffs_age_poly,
        options=options,
        params=params,
        state_space=state_space,
        indexer=indexer,
        init=init,
        num_sims=100,
        seed=7134,
    )

    dens_plot = sns.kdeplot(df["retirement_age"].loc[:, 0], fill=True, color="r")
    fig = dens_plot.get_figure()
    fig.savefig("retirement_age.png")

    with PdfPages("wealth0.pdf") as pdf_pages:
        fig = plt.figure()
        sns.kdeplot(df["wealth0"], fill=True, color="r")
        pdf_pages.savefig(fig)

    with PdfPages("wealth1.pdf") as pdf_pages:
        fig = plt.figure()
        sns.kdeplot(df["wealth1"], fill=True, color="r")
        pdf_pages.savefig(fig)

    with PdfPages("consumption.pdf") as pdf_pages:
        fig = plt.figure()
        sns.kdeplot(df["consumption"], fill=True, color="r")
        pdf_pages.savefig(fig)

    with PdfPages("working.pdf") as pdf_pages:
        fig = plt.figure()
        sns.kdeplot(df["working"], fill=True, color="r")
        pdf_pages.savefig(fig)

    breakpoint()
