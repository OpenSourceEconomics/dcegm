"""Test that a state-dependent discount factor matches looping over scalar solves.

We add a `type` deterministic state to the simple dcegm-paper retirement model that
affects *only* the discount factor (nothing else -- not utility, not the budget
constraint, not the choice set). We then compare two ways of solving it:

(i)  one joint solve, with `discount_factor_per_state` reading `beta_by_type[type]` from
`params`, and (ii) looping over each type value and solving the (type-less) base model
with the      corresponding scalar `discount_factor`.

Since `type` has no effect other than through the discount factor, and it never
transitions, the two approaches should be exactly equivalent: solving jointly with a
per-type discount factor cannot differ from solving each type separately with the
matching scalar. We check this by comparing interpolated policy and value functions on a
wealth grid, for every period/lagged_choice/type combination.

"""

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal as aaae

import dcegm
import dcegm.toy_models as toy_models

BETA_BY_TYPE = jnp.array([0.90, 0.98])


def discount_factor_per_type(type, params):
    return params["beta_by_type"][type]


def next_period_deterministic_state(period, choice, type):
    return {"period": period + 1, "lagged_choice": choice, "type": type}


@pytest.fixture()
def base_ingredients():
    model_funcs = toy_models.load_example_model_functions("dcegm_paper")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_retirement_with_shocks"
        )
    )
    # A handful of periods and a coarse wealth grid keep this test fast; the
    # equivalence being tested does not depend on their size.
    model_config["n_periods"] = 6
    return model_funcs, params, model_specs, model_config


def test_discount_factor_per_state_matches_scalar_loop(base_ingredients):
    model_funcs, params_base, model_specs, model_config = base_ingredients

    # --- (i) one joint solve, type as a state, discount factor read per state ---
    model_config_joint = {
        **model_config,
        "deterministic_states": {"type": np.arange(2, dtype=int)},
    }
    state_space_functions_joint = {
        **model_funcs["state_space_functions"],
        "next_period_deterministic_state": next_period_deterministic_state,
    }
    params_joint = {**params_base, "beta_by_type": BETA_BY_TYPE}
    del params_joint["discount_factor"]

    model_joint = dcegm.setup_model(
        model_config=model_config_joint,
        model_specs=model_specs,
        utility_functions=model_funcs["utility_functions"],
        utility_functions_final_period=model_funcs["utility_functions_final_period"],
        budget_constraint=model_funcs["budget_constraint"],
        state_space_functions=state_space_functions_joint,
        shock_functions={"discount_factor_per_state": discount_factor_per_type},
    )
    model_joint_solved = model_joint.solve(params_joint)

    wealth_grid = jnp.linspace(1.0, 40.0, 50)

    for type_value in (0, 1):
        # --- (ii) loop: plain (type-less) model, scalar discount factor ---
        params_loop = {
            **params_base,
            "discount_factor": float(BETA_BY_TYPE[type_value]),
        }
        model_loop = dcegm.setup_model(
            model_config=model_config,
            model_specs=model_specs,
            **model_funcs,
        )
        model_loop_solved = model_loop.solve(params_loop)

        for period in range(model_config["n_periods"] - 1):
            for lagged_choice in (0, 1):
                for choice in (0, 1):
                    if lagged_choice == 1 and choice == 0:
                        # retirement is absorbing; this state-choice does not exist
                        continue

                    states_joint = {
                        "period": jnp.full_like(wealth_grid, period, dtype=int),
                        "lagged_choice": jnp.full_like(
                            wealth_grid, lagged_choice, dtype=int
                        ),
                        "type": jnp.full_like(wealth_grid, type_value, dtype=int),
                        "assets_begin_of_period": wealth_grid,
                    }
                    states_loop = {
                        "period": jnp.full_like(wealth_grid, period, dtype=int),
                        "lagged_choice": jnp.full_like(
                            wealth_grid, lagged_choice, dtype=int
                        ),
                        "assets_begin_of_period": wealth_grid,
                    }
                    choices = jnp.full_like(wealth_grid, choice, dtype=int)

                    policy_joint, value_joint = (
                        model_joint_solved.policy_and_value_for_states_and_choices(
                            states=states_joint, choices=choices
                        )
                    )
                    policy_loop, value_loop = (
                        model_loop_solved.policy_and_value_for_states_and_choices(
                            states=states_loop, choices=choices
                        )
                    )

                    aaae(policy_joint, policy_loop, decimal=6)
                    aaae(value_joint, value_loop, decimal=6)
