import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from jax import config
from model_funcs import (
    budget_constraint,
    create_final_period_utility_function_dict,
    create_state_space_function_dict,
    create_utility_function_dict,
)

import dcegm

config.update("jax_enable_x64", True)

# Set professional style
plt.style.use("seaborn-v0_8-white")
plt.rcParams.update(
    {
        "font.family": "Helvetica",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.0,
        "lines.markersize": 8,
    }
)

# Refined colorblind-friendly palette
colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

# Model parameters
params = {
    "discount_factor": 0.95,
    "delta": 0.5,
    "rho": 1.0,
    "constant": np.log(20.0),
    "exp": 0.0,
    "exp_squared": 0.0,
    "income_shock_std": np.sqrt(0.005),
    "income_shock_mean": -0.005 / 2,
    "taste_shock_scale": 2e-16,
    "interest_rate": (1 - 0.95) / 0.95,
    "consumption_floor": 0.001,
}

model_specs = {
    "min_age": 20,
    "n_choices": 2,
}

# Define quadrature points to test
quad_points_list = [5, 15, 50, 100, 500]
n_quad_points_true = 500
# New list for 50 quadrature points (log-spaced for better distribution)
quad_points_extended = np.unique(np.logspace(0, 3, 50, dtype=int))


# State and choice for the plot
state_dict = {
    "period": 0,  # Period T-5 since we start from 0
    "lagged_choice": 0,
}
choice = 0

# Set up model functions
model_functions = {
    "utility_functions": create_utility_function_dict(),
    "utility_functions_final_period": create_final_period_utility_function_dict(),
    "state_space_functions": create_state_space_function_dict(),
    "budget_constraint": budget_constraint,
}

# Solve model for the "truth"
model_config_true = {
    "n_periods": 5,
    "choices": [0, 1],
    "continuous_states": {"assets_end_of_period": np.linspace(1, 200, 10000)},
    "n_quad_points": n_quad_points_true,
}
model_config_test = {
    "n_periods": 10,
    "choices": [0, 1],
    "continuous_states": {"assets_end_of_period": np.linspace(0, 150, 500)},
    "n_quad_points": n_quad_points_true,
}

model_true = dcegm.setup_model(
    model_config=model_config_true,
    model_specs=model_specs,
    **model_functions,
)

model_solved_true = model_true.solve(params)
endog_grid_true, policy_true, value_true = (
    model_solved_true.get_solution_for_discrete_state_choice(state_dict, choice)
)

# Wealth grid for interpolation
wealth_grid_to_test = np.linspace(
    np.nanmin(endog_grid_true), np.nanmax(endog_grid_true), 500
)
true_policy_interp = np.interp(wealth_grid_to_test, endog_grid_true, policy_true)
true_value_interp = np.interp(wealth_grid_to_test, endog_grid_true, value_true)

# --- Original Plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
# Plot consumption policy
axes[0].scatter(
    endog_grid_true,
    policy_true,
    s=1,
    color=colors[0],
    label="True Policy (N=1000)",
)

# Plot value function
axes[1].scatter(
    endog_grid_true,
    value_true,
    s=1,
    color=colors[0],
    label="True Value (N=1000)",
)

# Consumption policy plot settings
# axes[0].set_xlim([15, 170])
# axes[0].set_ylim([15, 25])
# axes[0].set_yticks(np.arange(15, 25, 1))
axes[0].set_xlabel("Wealth")
axes[0].set_ylabel("Consumption")
axes[0].set_title(
    "Policy Function, Period T-9, Varying Taste Shock,\n Continues Employment with Income Uncertainty",
    fontsize=11,
    fontweight="bold",
)
axes[0].legend()

# Value function plot settings
# axes[1].set_xlim([15, 170])
axes[1].set_xlabel("Wealth")
axes[1].set_ylabel("Value")
axes[1].set_title(
    "Value Function, Period T-9, Varying Taste Shock,\nContinues Employment with Income Uncertainty",
    fontsize=11,
    fontweight="bold",
)
axes[1].legend()

plt.tight_layout()
plt.show()

# # --- Original Relative Error Plots ---
# fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# for idx, n_quad_points in enumerate(quad_points_list):
#     model_config = model_config_test.copy()
#     model_config["n_quad_points"] = n_quad_points

#     model = dcegm.setup_model(
#         model_config=model_config,
#         model_specs=model_specs,
#         **model_functions,
#     )

#     model_solved = model.solve(params)
#     endog_grid, policy, value = model_solved.get_solution_for_discrete_state_choice(
#         state_dict, choice
#     )

#     policy_interp = np.interp(wealth_grid_to_test, endog_grid, policy)
#     value_interp = np.interp(wealth_grid_to_test, endog_grid, value)

#     rel_error_policy = np.abs((true_policy_interp - policy_interp) / true_policy_interp)
#     rel_error_value = np.abs((true_value_interp - value_interp) / true_value_interp)

#     # Plot with larger markers and thin lines
#     axes[0].plot(
#         wealth_grid_to_test,
#         rel_error_policy,
#         color=colors[idx],
#         alpha=0.3,
#         linewidth=0.8,
#     )
#     axes[0].scatter(
#         wealth_grid_to_test,
#         rel_error_policy,
#         label=f"N = {n_quad_points}",
#         color=colors[idx],
#         s=7,
#         alpha=0.9,
#         edgecolor="none",
#     )
#     axes[1].plot(
#         wealth_grid_to_test,
#         rel_error_value,
#         color=colors[idx],
#         alpha=0.3,
#         linewidth=0.8,
#     )
#     axes[1].scatter(
#         wealth_grid_to_test,
#         rel_error_value,
#         label=f"N = {n_quad_points}",
#         color=colors[idx],
#         s=7,
#         alpha=0.9,
#         edgecolor="none",
#     )

# # Format subplots
# for ax, title in zip(
#     axes, ["Policy Function Relative Error", "Value Function Relative Error"]
# ):
#     ax.set_yscale("log")
#     ax.set_xlim([15, 170])
#     ax.set_ylim([1e-6, 1e0])
#     ax.set_xlabel("Wealth", labelpad=8)
#     ax.set_ylabel("Relative Error", labelpad=8)
#     ax.set_title(f"{title}\nPeriod T-9, λ=0", pad=12, fontweight="bold")
#     ax.grid(True, which="both", linestyle="--", alpha=0.6)
#     ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
#     ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs="auto", numticks=10))
#     ax.yaxis.set_minor_formatter(ticker.NullFormatter())
#     ax.tick_params(axis="both", which="major", labelsize=12, direction="in")

# # Single legend for both subplots
# handles, labels = axes[0].get_legend_handles_labels()
# fig.legend(
#     handles, labels, loc="center right", bbox_to_anchor=(0.98, 0.5), frameon=False
# )

# plt.tight_layout(rect=[0, 0, 0.85, 1])
# plt.show()

# # --- New Plots for Mean and Max Relative Errors ---
# # Compute errors for extended quadrature points
# mean_error_policy = []
# max_error_policy = []
# mean_error_value = []
# max_error_value = []
# quad_points_used = []

# for n_quad_points in quad_points_extended:
#     if n_quad_points > 0:  # Skip zero quadrature points
#         model_config = model_config_test.copy()
#         model_config["n_quad_points"] = int(n_quad_points)

#         model = dcegm.setup_model(
#             model_config=model_config,
#             model_specs=model_specs,
#             **model_functions,
#         )

#         model_solved = model.solve(params)
#         endog_grid, policy, value = model_solved.get_solution_for_discrete_state_choice(
#             state_dict, choice
#         )

#         policy_interp = np.interp(wealth_grid_to_test, endog_grid, policy)
#         value_interp = np.interp(wealth_grid_to_test, endog_grid, value)

#         rel_error_policy = np.abs((true_policy_interp - policy_interp) / true_policy_interp)
#         rel_error_value = np.abs((true_value_interp - value_interp) / true_value_interp)

#         # Compute mean and max errors
#         mean_error_policy.append(np.nanmean(rel_error_policy))
#         max_error_policy.append(np.nanmax(rel_error_policy))
#         mean_error_value.append(np.nanmean(rel_error_value))
#         max_error_value.append(np.nanmax(rel_error_value))
#         quad_points_used.append(n_quad_points)

# # Create new subplots for mean and max errors
# fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

# # Plot mean relative error for policy function
# axes[0, 0].scatter(
#     quad_points_used,
#     mean_error_policy,
#     color=colors[0],
#     s=7,
#     alpha=0.9,
#     edgecolor="none",
#     label="Mean Relative Error",
# )
# axes[0, 0].plot(
#     quad_points_used,
#     mean_error_policy,
#     color=colors[0],
#     alpha=0.3,
#     linewidth=0.8,
# )

# # Plot max relative error for policy function
# axes[0, 1].scatter(
#     quad_points_used,
#     max_error_policy,
#     color=colors[1],
#     s=7,
#     alpha=0.9,
#     edgecolor="none",
#     label="Max Relative Error",
# )
# axes[0, 1].plot(
#     quad_points_used,
#     max_error_policy,
#     color=colors[1],
#     alpha=0.3,
#     linewidth=0.8,
# )

# # Plot mean relative error for value function
# axes[1, 0].scatter(
#     quad_points_used,
#     mean_error_value,
#     color=colors[0],
#     s=7,
#     alpha=0.9,
#     edgecolor="none",
#     label="Mean Relative Error",
# )
# axes[1, 0].plot(
#     quad_points_used,
#     mean_error_value,
#     color=colors[0],
#     alpha=0.3,
#     linewidth=0.8,
# )

# # Plot max relative error for value function
# axes[1, 1].scatter(
#     quad_points_used,
#     max_error_value,
#     color=colors[1],
#     s=7,
#     alpha=0.9,
#     edgecolor="none",
#     label="Max Relative Error",
# )
# axes[1, 1].plot(
#     quad_points_used,
#     max_error_value,
#     color=colors[1],
#     alpha=0.3,
#     linewidth=0.8,
# )

# # Format subplots
# titles = [
#     "Policy Function Mean Relative Error",
#     "Policy Function Max Relative Error",
#     "Value Function Mean Relative Error",
#     "Value Function Max Relative Error",
# ]
# for ax, title in zip(axes.flatten(), titles):
#     ax.set_xscale("log")
#     ax.set_yscale("log")
#     ax.set_xlim([1, 1000])
#     ax.set_ylim([1e-6, 1e0])
#     ax.set_xlabel("Number of Quadrature Points", labelpad=8)
#     ax.set_ylabel("Relative Error", labelpad=8)
#     ax.set_title(f"{title}\nPeriod T-5, λ=0", pad=12, fontweight="bold")
#     ax.grid(True, which="both", linestyle="--", alpha=0.6)
#     ax.xaxis.set_major_locator(ticker.LogLocator(base=10.0))
#     ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))
#     ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs="auto", numticks=10))
#     ax.yaxis.set_minor_formatter(ticker.NullFormatter())
#     ax.tick_params(axis="both", which="major", labelsize=12, direction="in")
#     ax.legend(frameon=False)

# plt.tight_layout()
# plt.show()
