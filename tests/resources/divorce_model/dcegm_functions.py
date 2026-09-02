"""dcegm-facing model functions for the divorce toy model.

`partner_state` is a genuine stochastic state (0 = single, 1 = married),
with per-period wealth and utility scaling keyed off the *current* period's
own `partner_state` only. See the dcegm guide "Implementing a
divorce/marriage transition without a lagged partner state"
(`docs/source/guides/`) for why the per-period convention used below is
nonetheless exactly equivalent to a transition-based ("halve on divorce,
double on marriage") rule.

As in `reference.py`, nothing here is hardcoded: `params` and the asset
grid are passed in by the caller (the test file owns the actual numbers).
"""

import jax
import jax.numpy as jnp
import numpy as np

import dcegm


def utility_func(consumption, choice, partner_state, params):
    # The 2 only applies if there is a partner: consumption is drawn from a
    # jointly funded (single, pooled) account, so a given dollar of recorded
    # spending only cost this individual half of it.
    scale = jnp.where(partner_state == 1, 2.0, 1.0)
    x = scale * consumption
    felicity = jax.lax.select(
        jnp.allclose(params["rho"], 1),
        jnp.log(x),
        (x ** (1 - params["rho"]) - 1) / (1 - params["rho"]),
    )
    return felicity - (1 - choice) * params["delta"]


def marginal_utility_func(consumption, partner_state, params):
    scale = jnp.where(partner_state == 1, 2.0, 1.0)
    x = scale * consumption
    du_dx = jax.lax.select(jnp.allclose(params["rho"], 1), 1 / x, x ** (-params["rho"]))
    return du_dx


def inverse_marginal_utility_func(marginal_utility, partner_state, params):
    # Inverts marginal_utility_func(c) = (scale*c)**(-rho): c = m**(-1/rho) / scale.
    scale = jnp.where(partner_state == 1, 2.0, 1.0)
    c_rho1 = (1 / scale) * (1 / marginal_utility)  # log case: scale cancels, as always
    c_general = marginal_utility ** (-1 / params["rho"]) / scale
    return jax.lax.select(jnp.allclose(params["rho"], 1), c_rho1, c_general)


def utility_final(wealth, choice, partner_state, params):
    return utility_func(wealth, choice, partner_state, params)


def marginal_utility_final(wealth, choice, partner_state, params):
    return marginal_utility_func(wealth, partner_state, params)


def budget_constraint(
    period,
    lagged_choice,
    partner_state,
    asset_end_of_previous_period,
    income_shock_previous_period,
    params,
    model_specs,
):
    multiplier = jnp.where(partner_state == 1, 2.0, 1.0)
    own_income = params["y_work"] * (lagged_choice == 0)
    # Partner income depends on *this* period's own partner_state directly
    # (no lagged_choice involved -- we don't model the partner's own labor
    # supply), added unscaled just like own_income.
    partner_income = params["y_partner"] * (partner_state == 1)
    wealth = (
        multiplier * asset_end_of_previous_period * (1 + params["interest_rate"])
        + own_income
        + partner_income
    )
    return jnp.maximum(wealth, params["consumption_floor"]) / multiplier


def feasible_choice_set(lagged_choice, model_specs):
    return np.arange(model_specs["n_choices"])


def partner_transition(partner_state, params):
    """Returns [P(single next), P(married next)]."""
    prob_married_next = jnp.where(
        partner_state == 1, params["persistence_married"], params["prob_marry"]
    )
    return jnp.array([1 - prob_married_next, prob_married_next])


def build_and_solve(params, n_periods, a_grid):
    model_specs = {"n_periods": n_periods, "n_choices": 2}
    model_config = {
        "n_periods": n_periods,
        "choices": np.arange(2),
        "stochastic_states": {"partner_state": np.arange(2)},
        "continuous_states": {"assets_end_of_period": a_grid},
        "n_quad_points": 5,
    }
    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        state_space_functions={"state_specific_choice_set": feasible_choice_set},
        stochastic_states_transitions={"partner_state": partner_transition},
        utility_functions={
            "utility": utility_func,
            "marginal_utility": marginal_utility_func,
            "inverse_marginal_utility": inverse_marginal_utility_func,
        },
        utility_functions_final_period={
            "utility": utility_final,
            "marginal_utility": marginal_utility_final,
        },
        budget_constraint=budget_constraint,
    )
    return model, model.solve(params)
