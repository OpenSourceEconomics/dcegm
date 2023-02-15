import pickle
from pathlib import Path

import pandas as pd
import yaml
from dcegm.marg_utilities_and_exp_value import (
    marginal_util_and_exp_max_value_states_period,
)
from dcegm.pre_processing import get_partial_functions
from dcegm.pre_processing import params_todict
from numpy.testing import assert_array_almost_equal as aaae
from toy_models.consumption_retirement_model.budget_functions import budget_constraint
from toy_models.consumption_retirement_model.exogenous_processes import (
    get_transition_matrix_by_state,
)
from toy_models.consumption_retirement_model.utility_functions import (
    inverse_marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import (
    marginal_utility_crra,
)
from toy_models.consumption_retirement_model.utility_functions import utility_func_crra

# Obtain the test directory of the package.
TEST_DIR = Path(__file__).parent
TEST_RESOURCES_DIR = TEST_DIR / "resources"

# inputs for function marginal_util_and_exp_max_value_states_period

utility_functions = {
    "utility": utility_func_crra,
    "inverse_marginal_utility": inverse_marginal_utility_crra,
    "marginal_utility": marginal_utility_crra,
}
params = pd.read_csv(TEST_RESOURCES_DIR / "deaton.csv", index_col=["category", "name"])
options = yaml.safe_load((TEST_RESOURCES_DIR / "deaton.yaml").read_text())
params_dict = params_todict(params)
(
    compute_utility,
    compute_marginal_utility,
    compute_inverse_marginal_utility,
    compute_value,
    compute_next_period_wealth,
    transition_function,
) = get_partial_functions(
    params_dict,
    options,
    utility_functions,
    budget_constraint,
    get_transition_matrix_by_state,
)
taste_shock_scale = params_dict["lambda"]
exogenous_savings_grid = pickle.load(
    open(TEST_RESOURCES_DIR / "exogenous_savings_grid.pkl", "rb")
)
income_shock_draws = pickle.load(
    open(TEST_RESOURCES_DIR / "income_shock_draws.pkl", "rb")
)
income_shock_weights = pickle.load(
    open(TEST_RESOURCES_DIR / "income_shock_weights.pkl", "rb")
)
possible_child_states = pickle.load(
    open(TEST_RESOURCES_DIR / "possible_child_states.pkl", "rb")
)
choices_child_states = pickle.load(
    open(TEST_RESOURCES_DIR / "choices_child_states.pkl", "rb")
)
policies_child_states = pickle.load(
    open(TEST_RESOURCES_DIR / "policies_child_states.pkl", "rb")
)
values_child_states = pickle.load(
    open(TEST_RESOURCES_DIR / "values_child_states.pkl", "rb")
)

marginal_utilities_expected = pickle.load(
    open(TEST_RESOURCES_DIR / "marginal_utilities.pkl", "rb")
)
max_expected_values_expected = pickle.load(
    open(TEST_RESOURCES_DIR / "max_expected_values.pkl", "rb")
)

marginal_utilities, max_expected_values = marginal_util_and_exp_max_value_states_period(
    compute_next_period_wealth,
    compute_marginal_utility,
    compute_value,
    taste_shock_scale,
    exogenous_savings_grid,
    income_shock_draws,
    income_shock_weights,
    possible_child_states,
    choices_child_states,
    policies_child_states,
    values_child_states,
)


def test_marginal_utilities_states_period():
    aaae(marginal_utilities, marginal_utilities_expected)


# does not pass yet
def test_max_expected_values_states_period():
    aaae(max_expected_values, max_expected_values_expected)
