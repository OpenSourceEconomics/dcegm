"""Regression tests for adjust_observed_assets.

PR #198 renamed the keys in the processed continuous_states_info
(second_continuous_exists -> has_additional_continuous_state, etc.) and removed
the continuous_state= kwarg alias that used to be injected into every processed
user function. asset_correction.py was not migrated:

- Wealth-only models raised KeyError: 'second_continuous_exists' (the config
  lookup throws regardless of which branch would be taken).
- Models with an additional continuous state passed the state under the dead
  "continuous_state" alias, so budget constraints declaring the real state name
  (e.g. experience) raised KeyError at trace time.

Expected values are computed by calling the raw toy budget constraints directly
per observation with asset_end_of_previous_period = observed_wealth / (1 + r)
and a zero income shock, independent of the vmap plumbing under test.

"""

import jax.numpy as jnp
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models
from dcegm.asset_correction import adjust_observed_assets
from dcegm.toy_models.cons_ret_model_dcegm_paper.budget_constraint import (
    budget_constraint,
)
from dcegm.toy_models.cons_ret_model_with_cont_exp.budget_constraint import (
    budget_constraint_cont_exp,
)


def test_adjust_observed_assets_wealth_only():
    """Wealth-only model: config lookup must use the renamed key."""
    model_funcs = toy_models.load_example_model_functions("dcegm_paper")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("dcegm_paper")
    )

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_funcs,
    )

    observed_states = {
        "period": jnp.array([0, 1, 2]),
        "lagged_choice": jnp.array([0, 0, 1]),
        "assets_begin_of_period": jnp.array([10.0, 20.0, 30.0]),
    }

    adjusted_assets = adjust_observed_assets(
        observed_states_dict=observed_states,
        params=params,
        model_class=model,
    )

    assets_end_of_last_period = observed_states["assets_begin_of_period"] / (
        1 + params["interest_rate"]
    )
    expected = jnp.array(
        [
            budget_constraint(
                period=period,
                lagged_choice=lagged_choice,
                asset_end_of_previous_period=asset_end,
                income_shock_previous_period=0.0,
                model_specs=model_specs,
                params=params,
            )
            for period, lagged_choice, asset_end in zip(
                observed_states["period"],
                observed_states["lagged_choice"],
                assets_end_of_last_period,
            )
        ]
    )

    aaae(adjusted_assets, expected)


def test_adjust_observed_assets_second_continuous_state():
    """Second continuous state: the budget constraint declares the state by its real
    name (experience), so it must not be passed under the dead alias."""
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_funcs,
    )

    observed_states = {
        "period": jnp.array([1, 2, 3]),
        "lagged_choice": jnp.array([0, 0, 1]),
        "experience": jnp.array([0.5, 0.25, 1.0]),
        "assets_begin_of_period": jnp.array([10.0, 20.0, 30.0]),
    }

    adjusted_assets = adjust_observed_assets(
        observed_states_dict=observed_states,
        params=params,
        model_class=model,
    )

    assets_end_of_last_period = observed_states["assets_begin_of_period"] / (
        1 + params["interest_rate"]
    )
    expected = jnp.array(
        [
            budget_constraint_cont_exp(
                period=period,
                lagged_choice=lagged_choice,
                experience=experience,
                asset_end_of_previous_period=asset_end,
                income_shock_previous_period=0.0,
                params=params,
                model_specs=model_specs,
            )
            for period, lagged_choice, experience, asset_end in zip(
                observed_states["period"],
                observed_states["lagged_choice"],
                observed_states["experience"],
                assets_end_of_last_period,
            )
        ]
    )

    aaae(adjusted_assets, expected)
