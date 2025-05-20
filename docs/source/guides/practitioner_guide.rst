.. _practitioner_guide:

Practitioner's Guide
=====================

This guide explains how to specify, solve, simulate and potentially estimate structural life cycle models using the `dc-egm` framework. First, the main interface functions of the package are explained, before we turn to explaining the inputs of These functions.

.. dropdown:: Overview of main interface

    The logic order of the main interface is inspired by the e.g. the OLS specification in `statsmodels`. In a first step,
    we specify a model,

.. dropdown:: Parameterization (params and model_specs)

    `dc-egm` uses two distinct dictionaries to store model parameters:

    - `params`: parameters estimated *within* the model
    - `model_specs`: parameters calibrated or estimated *outside* the model

    These should be Python dictionaries mapping string labels to numeric values.

    **`params`**

    `params` stores parameters that are subject to estimation (e.g., return to experience in a labor supply model).

    **`model_specs`**

    These contain external calibrations, such as interest rates or deterministic tax rules.

    Example:

    .. code-block:: python

        params = {
            "beta": 0.96,
            "gamma": 2.0
        }

        model_specs = {
            "r": 0.03,
            "tax_rate": 0.2
        }

    These objects are used to construct value functions and solve the model.

.. dropdown:: Model Configuration

    The `model_config` dictionary specifies all structural elements of your dynamic model. It includes required and optional elements.
