"""Independent hand-rolled reference solver for the divorce toy model.

Plain, explicit backward-induction EGM -- no upper envelope, no dcegm
machinery -- so it can serve as ground truth to check the dcegm-based model
in `dcegm_functions.py` against. State is always *individual* wealth;
resources are only rescaled at an actual period-to-period partner
transition (halved on divorce, doubled on marriage, unchanged otherwise),
and utility has no partner-status multiplier at all. See the dcegm guide
"Implementing a divorce/marriage transition without a lagged partner state"
(`docs/source/guides/`) for why the dcegm-side model uses a *different*,
per-period convention, and why the two are nonetheless the same underlying
economics.

Every function here takes the model's numeric parametrization (`params`, a
dict) explicitly -- nothing is hardcoded at module level. The actual numbers
used for testing live in `tests/test_divorce_toy_model.py`, not here.

Uses dcegm's timing convention throughout: income earned under a given
`choice` at period t is paid at the start of period t+1 (i.e. it enters the
budget via `lagged_choice`).

"""

import numpy as np


def partner_transition_np(partner_state, params):
    """Returns [P(single next), P(married next)]."""
    prob_married_next = (
        params["persistence_married"] if partner_state == 1 else params["prob_marry"]
    )
    return np.array([1 - prob_married_next, prob_married_next])


def resources_after_transition(a, partner_state_0, partner_state_1, own_income, params):
    """Plain resources, with the individual's asset stock rescaled once for
    the period-to-period partner transition: halved on divorce (lose the
    ex-partner's share), doubled on marriage (new partner matches wealth),
    unchanged otherwise. No scaling anywhere else -- not on staying
    partnered, not on income, not in utility. Partner income is added
    whenever partnered next period (partner_state_1), unscaled, exactly
    like own_income.
    """
    if partner_state_0 == 1 and partner_state_1 == 0:
        a = a / 2  # divorce
    elif partner_state_0 == 0 and partner_state_1 == 1:
        a = a * 2  # marriage

    partner_income = params["y_partner"] if partner_state_1 == 1 else 0.0
    wealth = a * (1 + params["interest_rate"]) + own_income + partner_income
    return max(wealth, params["consumption_floor"])


def consumption_utility(consumption, choice, params):
    mu, delta = params["rho"], params["delta"]
    u = (
        np.log(consumption)
        if abs(mu - 1) < 1e-12
        else (consumption ** (1 - mu) - 1) / (1 - mu)
    )
    return u - (1 - choice) * delta


def marginal_utility_np(consumption, params):
    mu = params["rho"]
    return 1 / consumption if abs(mu - 1) < 1e-12 else consumption ** (-mu)


def inverse_marginal_utility_np(marg_util, params):
    mu = params["rho"]
    return 1 / marg_util if abs(mu - 1) < 1e-12 else marg_util ** (-1 / mu)


def solve_reference_at_point(a0_end, partner_state_0, work0, params):
    """Manual EGM backward induction for a single exogenous end-of-period asset point,
    two periods only: period 1 (analytic terminal period) then the period-0 Euler
    equation.

    No grid needed for two periods -- the n-period version below (`solve_reference`)
    generalizes this by interpolating a stored grid instead of evaluating period 1
    analytically.

    """
    beta, r = params["discount_factor"], params["interest_rate"]
    taste_shock_scale = params["taste_shock_scale"]
    transition_probs = partner_transition_np(partner_state_0, params)

    # work0's income arrives at the start of period 1, i.e. lagged_choice =
    # work0 there.
    income_period1 = params["y_work"] * (work0 == 0)

    expected_marg_util = 0.0
    expected_value = 0.0
    for partner_state_1, prob_partner in enumerate(transition_probs):
        if prob_partner == 0.0:
            continue
        wealth_1 = resources_after_transition(
            a0_end, partner_state_0, partner_state_1, income_period1, params
        )

        # Terminal period: consume everything regardless of choice1, only
        # the disutility of choice1 differs.
        choice_values = np.array(
            [consumption_utility(wealth_1, c1, params) for c1 in (0, 1)]
        )
        choice_marg_utils = np.array(
            [marginal_utility_np(wealth_1, params) for _ in (0, 1)]
        )

        max_v = choice_values.max()
        weights = np.exp((choice_values - max_v) / taste_shock_scale)
        choice_probs = weights / weights.sum()
        ev_choice = max_v + taste_shock_scale * np.log(weights.sum())

        expected_marg_util += prob_partner * np.sum(choice_probs * choice_marg_utils)
        expected_value += prob_partner * ev_choice

    rhs_euler = beta * (1 + r) * expected_marg_util
    c0 = inverse_marginal_utility_np(rhs_euler, params)

    return {
        "endog_grid": c0 + a0_end,
        "policy": c0,
        "value": consumption_utility(c0, work0, params) + beta * expected_value,
    }


def solve_reference_backward_induction(params, a_grid):
    """Two-period manual EGM backward induction over the whole exogenous grid.

    Returns a dict keyed by (partner_state_0, work0), each holding parallel arrays
    `endog_grid`, `policy`, `value` -- one entry per point of `a_grid`, built by calling
    `solve_reference_at_point` at each point.

    """
    solved = {}

    for partner_state_0 in (0, 1):
        for work0 in (0, 1):
            points = [
                solve_reference_at_point(a0_end, partner_state_0, work0, params)
                for a0_end in a_grid
            ]
            solved[(partner_state_0, work0)] = {
                key: np.array([p[key] for p in points])
                for key in ("endog_grid", "policy", "value")
            }

    return solved


def solve_reference(n_periods, params, a_grid):
    """General n-period manual EGM backward induction, no upper envelope.

    Only the terminal period is analytic (consume everything). Every earlier
    period is solved on `a_grid` and stored; the period before it reads
    (interpolates) that stored grid for its own continuation value and
    marginal utility, exactly as dcegm does internally.

    Returns solved[period][(partner_state, choice)] = dict with parallel
    arrays "endog_grid", "policy", "value", for period in
    range(n_periods - 1). The terminal period (n_periods - 1) is not stored
    -- it's cheaper and exact to just evaluate `consumption_utility` /
    `marginal_utility_np` directly on any wealth level.
    """
    beta, r = params["discount_factor"], params["interest_rate"]
    taste_shock_scale = params["taste_shock_scale"]
    last_period = n_periods - 1
    solved = {}

    def continuation_value_and_marg_util(period, wealth, partner_state, choice):
        """Value and marginal utility of consumption in `period`, at a given
        wealth level, either analytically (terminal period) or by
        interpolating the already-solved grid for `period`.

        The endogenous grid for a solved period only starts at its own
        a_end=0 point's implied wealth (`endog_grid[0] == policy[0]`, since
        saving nothing means consuming everything) -- it does not cover
        [0, endog_grid[0]). Querying a lower wealth there is *not* an
        extrapolation edge case to clamp away: the true optimal policy is
        the borrowing constraint binding, i.e. consume everything
        (`policy = wealth`) and keep the same continuation choice of
        a_end=0. Handling this explicitly (rather than letting `np.interp`
        silently clamp to `policy[0]`) is exactly what dcegm's own natural
        borrowing-constraint point does internally.

        Symmetrically, a wealth level *above* the grid's own top point can
        arise here too (e.g. a marriage transition doubling wealth): there
        is no exact closed form there (unlike the borrowing constraint), so
        this uses standard linear extrapolation from the top two grid
        points -- accurate since `policy(wealth)` is close to linear for
        large wealth in a CRRA problem, and the alternative (letting
        `np.interp` clamp to a constant) is not.
        """
        if period == last_period:
            return consumption_utility(wealth, choice, params), marginal_utility_np(
                wealth, params
            )
        arrs = solved[period][(partner_state, choice)]
        endog_grid, policy_grid, value_grid = (
            arrs["endog_grid"],
            arrs["policy"],
            arrs["value"],
        )
        if wealth <= endog_grid[0]:
            policy = wealth
            # Continuation value at a_end=0, backed out from the stored
            # value at the grid's own lower endpoint (where policy ==
            # endog_grid[0] already, i.e. a_end=0 there too).
            ev_at_zero_savings = (
                value_grid[0] - consumption_utility(policy_grid[0], choice, params)
            ) / beta
            value = (
                consumption_utility(policy, choice, params) + beta * ev_at_zero_savings
            )
        elif wealth > endog_grid[-1]:
            slope_c = (policy_grid[-1] - policy_grid[-2]) / (
                endog_grid[-1] - endog_grid[-2]
            )
            slope_v = (value_grid[-1] - value_grid[-2]) / (
                endog_grid[-1] - endog_grid[-2]
            )
            policy = policy_grid[-1] + slope_c * (wealth - endog_grid[-1])
            value = value_grid[-1] + slope_v * (wealth - endog_grid[-1])
        else:
            policy = np.interp(wealth, endog_grid, policy_grid)
            value = np.interp(wealth, endog_grid, value_grid)
        return value, marginal_utility_np(policy, params)

    for period in range(last_period - 1, -1, -1):
        solved[period] = {}
        for partner_state_now in (0, 1):
            transition_probs = partner_transition_np(partner_state_now, params)

            for choice_now in (0, 1):
                endog_grid = np.empty_like(a_grid)
                policy = np.empty_like(a_grid)
                value = np.empty_like(a_grid)

                # Income from choice_now (work vs not) arrives at the start
                # of period + 1, dcegm-style.
                income_next = params["y_work"] * (choice_now == 0)

                for i, a_end in enumerate(a_grid):
                    expected_marg_util = 0.0
                    expected_value = 0.0
                    for partner_state_next, prob_partner in enumerate(transition_probs):
                        if prob_partner == 0.0:
                            continue
                        wealth_next = resources_after_transition(
                            a_end,
                            partner_state_now,
                            partner_state_next,
                            income_next,
                            params,
                        )

                        choice_values = np.empty(2)
                        choice_marg_utils = np.empty(2)
                        for choice_next in (0, 1):
                            v, mu = continuation_value_and_marg_util(
                                period + 1,
                                wealth_next,
                                partner_state_next,
                                choice_next,
                            )
                            choice_values[choice_next] = v
                            choice_marg_utils[choice_next] = mu

                        max_v = choice_values.max()
                        weights = np.exp((choice_values - max_v) / taste_shock_scale)
                        choice_probs = weights / weights.sum()
                        ev_choice = max_v + taste_shock_scale * np.log(weights.sum())

                        expected_marg_util += prob_partner * np.sum(
                            choice_probs * choice_marg_utils
                        )
                        expected_value += prob_partner * ev_choice

                    rhs_euler = beta * (1 + r) * expected_marg_util
                    c = inverse_marginal_utility_np(rhs_euler, params)

                    endog_grid[i] = c + a_end
                    policy[i] = c
                    value[i] = (
                        consumption_utility(c, choice_now, params)
                        + beta * expected_value
                    )

                solved[period][(partner_state_now, choice_now)] = {
                    "endog_grid": endog_grid,
                    "policy": policy,
                    "value": value,
                }

    return solved
