.. _practitioner_guide:

Practitioner's Guide
=====================

This guide explains how to specify and calibrate models using the `dc-egm` framework. It combines configuration and parameterization instructions in a modular format for practitioners.

.. dropdown:: Parameterization Guide

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

.. dropdown:: Model Configuration Guide

    The `model_config` dictionary specifies all structural elements of your dynamic model. It includes required and optional elements.

    ### Required

    - `n_periods`: Number of periods
    - `choices`: Discrete choices (e.g., work, retire)
    - `state_vars`: List of state variables (e.g., age, assets)

    .. code-block:: python

        model_config = {}

    ### Optional but usually needed

    - Endogenous discrete states

    Example:

    .. code-block:: python

        model_config = {}

    These specifications are passed to `setup_model()` which creates the full state-choice space and value function objects.
