.. _practitioner_guide:

Practitioner's Guide
=====================

This guide explains how to specify, solve, simulate and potentially estimate structural life cycle models using the `dcegm` framework. First, the main interface functions of the package are explained, before we turn to explaining the inputs of these functions. This guide requires you to have installed the dcegm packase as outlined in the installation guide.

.. dropdown:: Overview of main interface (specify, solve, simulate)

    The logic of the main interface is inspired by e.g. the OLS interface in `statsmodels`. In a first step,
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

    The difference between the two is that `params` contains parameters that are subject to be changed frequently. Most naturally these would be parameters to be estimated. Parameters which determine shapes of arrays or the number of computational steps have to be set in the `model_specs` dictionary. The distinction arises from the functionality of the `jax` library, which allows just in time compiling. More on this in the background section.

    Every user function can access both of these dictionaries, by including it in the signature. The five core parameters of the model, can be stored in either of the two objects. It is required to specify them in one of the two. The five core parameters are:

    - `discount_factor`: The discount factor
    - `interest_rate`: The interest rate
    - `taste_shock_scale`: The scale of the the taste shock.
    - `income_shock_std`: The income shock standard deviation of the assumed normal distribution.
    - `income_shock_mean`: The mean of the income shock distribution of the assumed normal distribution.

    An example for a model, where one estimates the disutility of work, the taste shock scale and fixes the income parameters, would be:

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

.. dropdown:: Model Configuration (model_config)

    The `model_config` dictionary specifies the structure of the model. It is processed internally by the software and creates the decision tree of the dynamic programming model. We will now document the mandatory keys of the `model_config` dictionary before we turn to the optional ones. The mandatory keys are:

    - `n_periods`: Number of period. Needs to be an integer larger than 1.
    - `choices`: Discrete choices of the model. Consecutive integers starting from 0. Either provided as a list or as a integer, which then is converted to a list with consecutive integers starting from 0 and to the integer minus 1.
    - `continuous_states`: Dictionary containing the grids for continuous variable. The dictionary requires
        - `assets_end_of_period`: The grid for the end of period assets, which is required for the egm step. It is expected as a numpy array and with monotonic increasing values.
    - `n_quad_points`: Number of quadrature points used for the integration over the income shock distribution. The quadrature points are used to approximate the integral of the value function over the income shock distribution. The number of quadrature points should be a positive integer.

    An example for a model configuration with the mandatory keys is:

    .. code-block:: python

        model_config = {
            "n_periods": 20,
            "choices": 2,
            "continuous_states": {
                "assets_end_of_period": numpy.linspace(0, 10, 100),
            },
            "n_quad_points": 5,
        }

    This is enough to specify the simplest model. The following keys can be used to specify more complex models. They are optional and can be used in any combination. The optional keys are:

    - `deterministic_states`: Dictionary containing the name of deterministic state variables of the model as keys. For a given key the corresponding value has to be a numpy array or python list with the possible values of the respective deterministic state variable. The values have to be consecutive integers starting from 0.
    - `stochastic_states`: Dictionary containing the name of stochastic state variables of the model as keys. For a given key the corresponding value has to be a numpy array or python list with the possible values of the respective stochastic state variable.The values have to be consecutive integers starting from 0. The transition probabilities of the stochastic states are specified in the stochastic_state_transitions which is explained below.


    Additionally, one can define a second continuous state variable. This can be done, by adding the state name as a key in `continuous_states` and a monotone increasing grid.

    An example for a model configuration with all optional keys is:
    .. code-block:: python

        model_config = {
            "n_periods": 30,
            "choices": np.arange(3, dtype=int),
            "deterministic_states": {
                "already_retired": np.arange(2, dtype=int),
            },
            "continuous_states": {
                "assets_end_of_period": np.arange(0, 100, 5, dtype=float),
                "experience": np.linspace(0, 1, 7, dtype=float),
            },
            "stochastic_states": {
                "job_offer": [0, 1],
                "survival": [0, 1],
            },
            "n_quad_points": 5,
        }


.. dropdown:: Utility Function

    The utility function, its derivative the marginal utility function, as well as the inverse marginal utility function have to be supplied to the `setup_model` function. This is done via the utility functions dictionary, which has to consist of three keys. An example would be:

    .. code-block:: python

        utility_functions = {
            "utility": utility_function,
            "marginal_utility": marginal_utility_function,
            "inverse_marginal_utility": inverse_marginal_utility_function,
        }

    The user is responsible to ensure, that the functions are the derivative of the utility function and its inverse. Here is an example for a utility function from the dcegm paper (you can find this function in the toy models of the package):

    .. code-block:: python

        def utility_crra(consumption, choice, params):

            rho_equal_one = jnp.allclose(params["rho"], 1)

            log_utility = jnp.log(consumption)

            utility_rho_not_one = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])

            utility_consumption = jax.lax.select(rho_equal_one, log_utility, utility_rho_not_one)

            utility = utility_consumption - (1 - choice) * params["delta"]

        return utility

    Note, that the utility function has to be written jax jit compatible. There we can not use any if conditions (except for arguments in model_specs, as these are fixed before evaluating). In this case, `rho` is a part of params. So in order to write the function, such that it can be evaluated for all possible values of `rho` including 1, we need to check if `rho` is equal to 1, calculate the utility for either case and select the correct one. Note, that instead of jax.lax.select, one could also use jnp.where.

    The utility function is evaluated for each state and choice separately. Besides the standard arguments of `params` and `model_specs`, the following state-choice variables can be used in the signature:
        - consumption
        - choice
        - period
        - lagged_choice
        - assets_begin_of_period
        - state_name (any key of `deterministic_states`, `stochastic_states`, `continuous_states`)

    The interfaces of `marginal_utility` and `inverse_marginal_utility` are accept the same inputs, except `inverse_marginal_utility` where naturally consumption is not accepted, but instead `marginal_utility`.
