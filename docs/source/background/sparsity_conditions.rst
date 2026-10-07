.. _sparsity_conditions:

Defining Sparsity Conditions
============================

Defining valid transitions between states is crucial for both correctness and computational efficiency. *dcegm* handles these restrictions via **sparsity conditions** which should be defined by the user as a function which is passed on to ``dcegm.setup_model()`` via the ``state_space_functions`` dictionary.

Why Sparsity Conditions Matter
------------------------------

DCEGM constructs the full model state space using the cross-product of all possible values for the discrete state variables (e.g., period, job offer, retirement status, health, (discrete) experience, etc.). Without constraints, this would result in a very large number of implausible or impossible states being included in the model solution (for instance a larger number of experience than completed periods in the model).

To ensure that only **feasible** and **logically consistent** states are evaluated, *dcegm* allows users to define a `sparsity_condition` function. This function serves two purposes:

1. **Filters** out invalid state combinations that violate model logic (e.g., working while dead).
2. **Returns** cleaned or adjusted versions of the state variables to account for absorbing states (e.g., setting `job_offer = 0` when dead).

Note that sparsity conditions can only be defined for discrete state variables.

Basic Structure of a Sparsity Condition
---------------------------------------

A sparsity condition function takes as input all current state variables and the model specification dictionary. It returns either:

- `False` to discard the current state as invalid,
- or a dictionary with valid (possibly adjusted) state values to retain.

The logic typically includes:

- **Age- or period-based constraints** (e.g., too young to retire),
- **Absorbing states** (e.g., once retired, always retired),
- **Consistency checks** (e.g., cannot have work experience without past employment),
- **Implied adjustments** (e.g., dead agents do not receive job offers).

Example: Sparsity Condition for a Retirement Model
--------------------------------------------------

Below is an example sparsity condition from a stylized retirement model (*full_model*) which can be found in the templates.

.. code-block:: python

    def sparsity_condition(
        period, lagged_choice, job_offer, already_retired, survival, model_specs
    ):
        last_period = model_specs["n_periods"] - 1
        min_ret = model_specs["min_ret_period"]
        max_ret = model_specs["max_ret_period"]

        c1 = (period <= min_ret) & (lagged_choice == 0)
        c2 = (lagged_choice != 0) & (already_retired == 1)
        c3 = (period <= min_ret + 1) & (already_retired == 1)
        c4 = (period > max_ret + 1) & (already_retired != 1) & (survival != 0)
        c5 = (period > max_ret) & (lagged_choice != 0) & (survival != 0)

        if c1 or c2 or c3 or c4 or c5:
            return False

        job_offer_out = 0 if (survival == 0 or lagged_choice == 0) else job_offer
        period_out = last_period if survival == 0 else period

        return {
            "period": period_out,
            "lagged_choice": lagged_choice,
            "already_retired": already_retired,
            "survival": survival,
            "job_offer": job_offer_out,
        }

Explanation of Logic:

- **c1**: Rules out working before retirement is legally possible.
- **c2/c3**: Ensures logical consistency around retirement being an absorbing state.
- **c4/c5**: Forces retirement if agent is above maximum retirement age.
- **Adjustments**: If the agent is *dead* (`survival == 0`), the job offer is set to 0, and the period is fixed at the final period to make the state absorbing.

Utility Functions at Proxied Terminal States
--------------------------------------------

Proxying an absorbing terminal state has a consequence for your **utility functions** that is easy to miss. In the example above, every dead state (``survival == 0``) is redirected to a single slot in the final period, where its value is stored once and reused. But that stored value is not the whole story: during the EGM step, whenever a *living* parent integrates over its (possibly dead) children, *dcegm* re-evaluates your periodic ``utility`` and ``marginal_utility`` at the child's own identity -- and for a dead child that identity still carries ``survival == 0``.

So your periodic utility functions must branch on the terminal state and return the terminal (e.g. consume-all / bequest) value there, not the ordinary living value:

.. code-block:: python

    def utility(consumption, survival, ..., params):
        alive = utility_alive(consumption, ..., params)
        dead = utility_final_consume_all(wealth=consumption, ..., params)
        return jnp.where(survival == 0, dead, alive)

Two things matter:

- **It must match** ``utility_functions_final_period``. The dead child's value was stored with the final-period utility; the branch above is what the parent re-evaluates for that same child during EGM. If the two disagree, the two representations of the same terminal state are inconsistent and the Euler equation of every parent with a dead child is wrong.
- **The same applies to** ``marginal_utility``, and more strongly: the child's marginal utility is *always* recomputed at the proxied identity in the EGM step (it is never read back from storage), so a missing terminal branch there breaks the bequest term directly.

When this branch is exercised
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The terminal branch of the *periodic* functions fires for every period solved by the main backward-induction scan -- that is, all periods **except the last two**:

- the **last period** is solved entirely with ``utility_functions_final_period`` and never calls the periodic ``utility``;
- the **second-to-last period** is handed the final period's values and marginal utilities directly, so it too bypasses the periodic terminal branch;
- **every earlier period** reads its terminal children through the proxy and re-evaluates the periodic ``utility`` / ``marginal_utility`` at the ``survival == 0`` identity.

Best Practices
--------------

- Make sparsity conditions **strict**: only allow logically valid state combinations.
- Handle **absorbing states** like death or permanent retirement carefully.
- When a sparsity condition proxies an absorbing terminal state, make sure your periodic ``utility`` and ``marginal_utility`` branch on that state and agree with ``utility_functions_final_period`` (see `Utility Functions at Proxied Terminal States`_ above).
- Ensure that any state created in the deterministic transition function also satisfies the sparsity condition.

Failing to correctly define sparsity conditions will result in `ValueError` exceptions and warnings during model setup, as *dcegm* verifies that every `(state, choice)` pair leads to a valid next state.

If your restriction depends on a *choice* -- either the one made last period or which choices a state offers at all -- see :ref:`choice_dependent_sparsity`, which covers how the two mechanisms (``sparsity_condition`` and ``state_specific_choice_set``) divide that work between them.

To help you setup the correct sparsity conditions for your model, a debug mode is available which returns a ``pandas.DataFrame`` over the unfiltered state space, showing ``is_valid``/``is_proxied`` per candidate state:

.. code-block:: python

    from dcegm.pre_processing.setup_model import create_model_dict

    state_space_df = create_model_dict(..., debug_info="state_space_df")

Note this must be called on ``create_model_dict``, not on ``setup_model``: the latter is a class that immediately indexes ``model_dict["model_config"]``, so the debug path (which returns a DataFrame) raises ``KeyError: 'model_config'`` there.
