import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap

import dcegm
import dcegm.toy_models as toy_models


@pytest.mark.parametrize(
    "model_name",
    [
        ("retirement_no_shocks"),
        ("retirement_with_shocks"),
        ("deaton"),
    ],
)
def test_load_and_save_model(
    model_name,
):
    _params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_" + model_name
        )
    )
    model_funcs = toy_models.load_example_model_functions("dcegm_paper_" + model_name)

    model_setup = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )

    model_after_saving = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        model_save_path="model.pkl",
        **model_funcs,
    )

    model_after_loading = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        model_load_path="model.pkl",
        **model_funcs,
    )

    # Get list of attributes
    attr_list = [
        "model_structure",
        "model_config",
        "batch_info",
    ]

    for key in attr_list:
        key_attr = getattr(model_setup, key)
        if isinstance(key_attr, np.ndarray):
            # Request attributes from model classes
            np.testing.assert_allclose(key_attr, getattr(model_after_saving, key))
            np.testing.assert_allclose(key_attr, getattr(model_after_loading, key))
        elif isinstance(key_attr, dict):
            for k in key_attr.keys():
                if isinstance(key_attr[k], np.ndarray):
                    np.testing.assert_allclose(
                        key_attr[k], getattr(model_after_loading, key)[k]
                    )
                    np.testing.assert_allclose(
                        key_attr[k], getattr(model_after_loading, key)[k]
                    )
                else:
                    pass
        else:
            pass

    import os

    os.remove("model.pkl")


@pytest.mark.parametrize(
    "model_name",
    [
        ("retirement_no_shocks"),
        ("retirement_with_shocks"),
        ("deaton"),
    ],
)
def test_load_and_save_solution(
    model_name,
):
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config(
            "dcegm_paper_" + model_name
        )
    )
    model_funcs = toy_models.load_example_model_functions("dcegm_paper_" + model_name)

    model = dcegm.setup_model(
        model_config=model_config, model_specs=model_specs, **model_funcs
    )
    model_solved = model.solve(params)
    model_solved_after_saving = model.solve(params, save_sol_path="sol.pkl")
    model_solved_after_loading = model.solve(params, load_sol_path="sol.pkl")

    # Check if
    for key in ["value", "policy", "endog_grid"]:
        key_attr = getattr(model_solved, key)
        key_attr_after_loading = getattr(model_solved_after_loading, key)
        key_attr_after_saving = getattr(model_solved_after_saving, key)

        np.testing.assert_allclose(key_attr, key_attr_after_loading)
        np.testing.assert_allclose(key_attr, key_attr_after_saving)

    n_agents = 1_000

    states_initial = {
        "period": jnp.zeros(n_agents, dtype=int),
        "lagged_choice": jnp.zeros(n_agents, dtype=int),
        "assets_begin_of_period": jnp.ones(n_agents, dtype=float) * 10,
    }
    seed = 132

    df = model.solve_and_simulate(
        params=params,
        states_initial=states_initial,
        seed=seed,
    )

    df_after_saving = model.solve_and_simulate(
        params=params, states_initial=states_initial, seed=seed, save_sol_path="sol.pkl"
    )

    df_after_loading = model.solve_and_simulate(
        params=params, states_initial=states_initial, seed=seed, load_sol_path="sol.pkl"
    )
    df.equals(df_after_saving)
    df_after_loading.equals(df_after_saving)

    import os

    os.remove("sol.pkl")
