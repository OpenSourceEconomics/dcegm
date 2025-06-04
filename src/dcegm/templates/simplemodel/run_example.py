# import pdb
# import traceback


def main():

    # packages needed
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from jax import config

    # import model_funcs
    from model_funcs import (
        budget_constraint,
        create_final_period_utility_function_dict,
        create_state_space_function_dict,
        create_utility_function_dict,
    )

    import dcegm

    config.update("jax_enable_x64", True)

    line_styles = [
        {
            "color": "black",
            "marker": "o",
            "linestyle": "None",
            "markersize": 5,
        },  # Black dots
        {"color": "red", "linestyle": "-"},  # Red solid line
        {"color": "black", "linestyle": "--"},  # Black dashed line
        {"color": "black", "linestyle": "-."},  # Black dash-dot line
        {"color": "red", "linestyle": ":"},  # Red dotted line
    ]

    ## set up
    params = {
        "discount_factor": 0.98,
        # disutility of work
        "delta": 1.0,
        # CRRA coefficient
        "rho": 1.0,
        # labor income coefficients
        "constant": np.log(20.0),
        "exp": 0.0,
        "exp_squared": 0.0,
        # Shock parameters of income
        "income_shock_std": 0,
        "income_shock_mean": 0,
        "taste_shock_scale": 0.1,
        "interest_rate": 0.0,
        "consumption_floor": 0.001,
    }
    # set the income uncertainty parameters analogous to the Figure 4 (b) in the paper
    params["income_shock_std"] = np.sqrt(0.005)
    params["income_shock_mean"] = -0.005 / 2
    model_specs = {
        "min_age": 20,
        "n_choices": 2,
    }

    model_config = {
        "n_periods": 10,  # number of periods in the model (e.g. 43 for t = 0, 1, ..., 42)
        "choices": [0, 1],  # choices for the model
        "continuous_states": {
            "assets_end_of_period": np.linspace(
                0,
                150,
                500,  # why if i set this on 100 does upper envelope go into a infinite loop?
            )
        },
        "n_quad_points": 5,  # number of quadrature points for the income shock
    }

    model_functions = {
        "utility_functions": create_utility_function_dict(),
        "utility_functions_final_period": create_final_period_utility_function_dict(),
        "state_space_functions": create_state_space_function_dict(),
        "budget_constraint": budget_constraint,
    }

    model = dcegm.setup_model(
        model_config=model_config,
        model_specs=model_specs,
        **model_functions,
    )

    # Define state and choice for the plot
    state_dict = {
        "period": 4,  # Perioid T-5
        "lagged_choice": 0,  # working in the previous period
    }
    choice = 0  # continuing to work

    # solve the model for different taste shock scales with income uncertainty and plot both
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    for k, taste_shock_scale in enumerate(
        [0.05, 0.10, 0.15]
    ):  # enumerate([2e-16, 0.01, 0.05, 0.10, 0.15]):
        params["taste_shock_scale"] = taste_shock_scale
        model_solved = model.solve(params)
        endog_grid, policy, value = model_solved.get_solution_for_discrete_state_choice(
            state_dict, choice
        )

        # Plot consumption policy
        axes[0].plot(
            endog_grid,
            policy,
            label=f"λ={taste_shock_scale:.2f}",
            **line_styles[k],
        )

        # Plot value function
        axes[1].plot(
            endog_grid,
            value,
            label=f"λ={taste_shock_scale:.2f}",
            **line_styles[k],
        )

    # Consumption policy plot settings
    # axes[0].set_xlim([15, 120])
    # axes[0].set_ylim([15, 25])
    # axes[0].set_yticks(np.arange(15, 25, 1))
    axes[0].set_xlabel("Wealth")
    axes[0].set_ylabel("Consumption")
    axes[0].set_title(
        "Consumption Policy Function\nPeriod T-5, Varying Taste Shock, Continues Employment",
        fontsize=11,
        fontweight="bold",
    )
    axes[0].legend()

    # Value function plot settings
    # axes[1].set_xlim([15, 120])
    axes[1].set_xlabel("Wealth")
    axes[1].set_ylabel("Value")
    axes[1].set_title(
        "Value Function\nPeriod T-5, Varying Taste Shock, Continues Employment",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].legend()

    plt.tight_layout()
    plt.show()

    # # Simulate the model
    # n_agents = 1_000
    # states_initial = {
    #     "period": jnp.zeros(n_agents),
    #     "lagged_choice": jnp.zeros(n_agents),  # all agents start as workers
    #     "experience": jnp.ones(n_agents),
    #     "assets_begin_of_period": jnp.ones(n_agents) * 10,
    # }

    # model_solved.simulate(states_initial=states_initial, seed=42)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        pdb.post_mortem()  # Start the debugger in post-mortem mode
