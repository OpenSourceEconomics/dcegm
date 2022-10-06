from functools import partial

from dcegm.aggregate_policy_value import calc_current_period_policy
from dcegm.aggregate_policy_value import calc_expected_value
from dcegm.aggregate_policy_value import calc_next_period_choice_probs
from dcegm.aggregate_policy_value import calc_value_constrained
from dcegm.egm_step import _store_current_period_policy_and_value
from scipy.special import roots_sh_legendre
from scipy.stats import norm


def partial_functions(
    params,
    options,
    exogenous_savings_grid,
    user_utility_func,
    user_marginal_utility_func,
    user_inverse_marginal_utility_func,
    user_budget_constraint,
    user_marginal_next_period_wealth,
):
    sigma = params.loc[("shocks", "sigma"), "value"]
    n_quad_points = options["quadrature_points_stochastic"]
    # Gauss-Legendre (shifted) quadrature over the interval [0,1].
    quad_points, quad_weights = roots_sh_legendre(n_quad_points)
    quad_points_normal = norm.ppf(quad_points)
    compute_utility = partial(
        user_utility_func,
        params=params,
    )
    compute_marginal_utility = partial(
        user_marginal_utility_func,
        params=params,
    )
    compute_inverse_marginal_utility = partial(
        user_inverse_marginal_utility_func,
        params=params,
    )
    compute_current_policy = partial(
        calc_current_period_policy,
        quad_weights=quad_weights,
        compute_inverse_marginal_utility=compute_inverse_marginal_utility,
    )
    compute_value_constrained = partial(
        calc_value_constrained,
        beta=params.loc[("beta", "beta"), "value"],
        compute_utility=compute_utility,
    )
    compute_expected_value = partial(
        calc_expected_value,
        params=params,
        quad_weights=quad_weights,
    )
    compute_next_choice_probs = partial(
        calc_next_period_choice_probs, params=params, options=options
    )
    compute_next_wealth_matrices = partial(
        user_budget_constraint,
        savings=exogenous_savings_grid,
        params=params,
        options=options,
        income_shocks=quad_points_normal * sigma,
    )
    compute_next_marginal_wealth = partial(
        user_marginal_next_period_wealth,
        params=params,
        options=options,
    )
    store_current_policy_and_value = partial(
        _store_current_period_policy_and_value,
        savings=exogenous_savings_grid,
        params=params,
        options=options,
        compute_utility=compute_utility,
    )
    return (
        compute_utility,
        compute_marginal_utility,
        compute_current_policy,
        compute_value_constrained,
        compute_expected_value,
        compute_next_choice_probs,
        compute_next_wealth_matrices,
        compute_next_marginal_wealth,
        store_current_policy_and_value,
    )
