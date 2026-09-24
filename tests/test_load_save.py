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


# =====================================================================================
# Continuous grids declared as None in model_config
#
# Their size is only knowable once a real state-choice exists to evaluate the grid
# function against, so check_model_config.py leaves it unresolved. On the load path
# model_config is rebuilt from the raw user config while the state-choice space comes
# back from the pickle, so the size has to be pinned again there.
# =====================================================================================


def _state_specific_experience_model():
    """``experience`` -- an additional continuous state -- supplied per state-choice."""
    model_funcs = toy_models.load_example_model_functions("with_cont_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_cont_exp")
    )
    model_config = dict(model_config)
    model_config["continuous_states"] = dict(model_config["continuous_states"])
    grid = jnp.asarray(model_config["continuous_states"]["experience"])
    model_config["continuous_states"]["experience"] = None

    def experience_grid(period):
        return grid

    return (
        model_funcs,
        params,
        model_specs,
        model_config,
        {"experience": experience_grid},
    )


def _state_specific_wealth_grid_model():
    """``assets_begin_of_period`` -- the Druedahl-Jorgensen wealth grid."""
    model_funcs = toy_models.load_example_model_functions("with_exp")
    params, model_specs, model_config = (
        toy_models.load_example_params_model_specs_and_config("with_exp")
    )
    model_config = dict(model_config)
    model_config["continuous_states"] = dict(model_config["continuous_states"])
    model_config["continuous_states"]["assets_begin_of_period"] = None
    model_config["upper_envelope"] = {"method": "druedahl_jorgensen"}
    grid = jnp.linspace(0, 50, 50)

    def wealth_grid(period):
        return grid

    return (
        model_funcs,
        params,
        model_specs,
        model_config,
        {"assets_begin_of_period": wealth_grid},
    )


@pytest.mark.parametrize(
    "model_loader",
    [_state_specific_experience_model, _state_specific_wealth_grid_model],
)
def test_loaded_model_pins_deferred_continuous_grid_sizes(model_loader, tmp_path):
    model_funcs, params, model_specs, model_config, continuous_grid_functions = (
        model_loader()
    )
    path = str(tmp_path / "model.pkl")

    model_saved = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        continuous_grid_functions=continuous_grid_functions,
        model_save_path=path,
        **model_funcs,
    )
    model_loaded = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        continuous_grid_functions=continuous_grid_functions,
        model_load_path=path,
        **model_funcs,
    )

    for key in ["n_total_wealth_grid"]:
        assert model_loaded.model_config[key] == model_saved.model_config[key]
        assert model_loaded.model_config[key] is not None

    combinations_key = "n_continuous_state_combinations"
    assert (
        model_loaded.model_config["continuous_states_info"][combinations_key]
        == model_saved.model_config["continuous_states_info"][combinations_key]
    )
    assert (
        model_loaded.model_config["continuous_states_info"][combinations_key]
        is not None
    )

    np.testing.assert_array_equal(
        np.asarray(model_loaded.solve(params).value),
        np.asarray(model_saved.solve(params).value),
    )
