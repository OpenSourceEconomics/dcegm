.. _practitioner_guide:

Practitioner's Guide
=====================

This guide explains how to specify, solve, simulate and potentially estimate structural life cycle models using the `dc-egm` framework. First, the main interface functions of the package are explained, before we turn to explaining the inputs of these functions. This guide requires you to have installed the dcegm packase as outlined in the installation guide.

.. dropdown:: Overview of main interface (specify, solve, simulate)

    The logic order of the main interface is inspired by the e.g. the OLS specification in `statsmodels`. In a first step,
    we specify a model, then we can solve it (in parallel to fitting an OLS model), and finally we can simulate it (in parallel to the predict method of an OLS model). Lets walk through the three steps:

    .. code-block:: python

        import dcgem

        model = dcegm.setup_model(
            model_config=model_config,
            model_specs=model_specs,
            utility_functions=utility_functions,
            utility_functions_final_period=utility_functions_final_period,
            budget_constraint=budget_constraint,
            state_space_functions=state_space_functions,
            stochastic_states_transitions=stochastic_states_transitions,
            shock_functions=shock_functions,
        )

    The `setup_model` functions takes several inputs, which are explained in the dropdowns below. It returns the model object. It is specified as a class and has several attributes and methods. You can find extensive documentation on these in the API section. Most relevant the class has a `solve` method, which takes the parameters `params` as input. The difference between the parameter container `model_specs` and `params` is explained in the parametrization dropdown.

    .. code-block:: python

        solved_model = model.solve(params=params)

    The solve method returns a solved model object, which contains the solved value functions and policy functions. This class now has the method of simulation, which returns the simulated data containing life cycle profile for all agents.

    .. code-block:: python

        simulated_data = solved_model.simulate(
            states_initial=states_initial,
            seed=111,
        )


.. dropdown:: Parameterization (params and model_specs)

    `dcegm` uses two distinct dictionaries to store model parameters.

    - `params`
    - `model_specs`

    The difference between the two is that `params` contains parameters that are subject to be changed frequently. Most naturrally these would be parameters to be estimated. Parameters which determine shapes of arrays or the number of computational steps have to be set in the `model_specs` dictionary. The distinction arises from the functionality of the `jax` library, which allows just in time compiling. More on this in the background section.

    Every user function can access both of these dictionaries, by including it in the signature. The five core parameters of the model, can be stored in either of the two objects. These are:

    - `discount_factor`: The discount factor
    - `interest_rate`: The interest rate
    - `taste_shock_scale`: The scale of the the taste shock.
    - `income_shock_std`: The incomme shock standard deviation of the assumed normal distribution.
    - `income_shock_mean`: The mean of the income shock distribution of the assumed normal distribution.

    An example for a model, where one estimates the disutility of work and the taste shock scale and fixes the income parameters, would be:

    .. code-block:: python

        params = {
            "disutil_of_work": 2,
            "taste_shock_scale": 1,
        }

        model_specs = {
            "discount_factor": 0.98,
            "interest_rate": 0.02,
            "income_shock_std": 0.5,
            "income_shock_mean": 0

        }

.. dropdown:: Model Configuration

    The `model_config` dictionary specifies all structural elements of your dynamic model. It includes required and optional elements.
