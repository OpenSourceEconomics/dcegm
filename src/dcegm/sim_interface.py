import jax

from dcegm.simulation.simulate import simulate_all_periods
from dcegm.solve import get_solve_func_for_model


def get_sol_and_sim_func_for_model(
    model, states_initial, wealth_initial, n_periods, seed, alt_model_funcs_sim=None
):

    solve_func = get_solve_func_for_model(model)

    sim_func = lambda params, value, policy, endog_gid: simulate_all_periods(
        states_initial=states_initial,
        wealth_initial=wealth_initial,
        n_periods=n_periods,
        params=params,
        seed=seed,
        endog_grid_solved=endog_gid,
        policy_solved=policy,
        value_solved=value,
        model=model,
        alt_model_funcs_sim=alt_model_funcs_sim,
    )

    def solve_sim_func(params):

        (value_solved, policy_solved, endog_grid_solved) = solve_func(params=params)

        sim_dict = sim_func(
            params=params,
            value=value_solved,
            policy=policy_solved,
            endog_gid=endog_grid_solved,
        )

        out = {
            "sim_dict": sim_dict,
            "value": value_solved,
            "policy": policy_solved,
            "endog_grid": endog_grid_solved,
        }

        return out

    return jax.jit(solve_sim_func)
