# Specifying Exogenous Processes in the DCEGM Module

## Overview

Exogenous processes in the ```dcegm``` module define how state variables evolve over time based on transition probabilities. These transition probabilities must be properly specified to ensure consistency with the model structure. This document outlines how to define exogenous processes and specify them within the model's ```options``` dictionary.

## Defining an Exogenous Process Function

An exogenous process function takes relevant state variables and model parameters as inputs and returns an ordered list of transition probabilities for each possible next state. Each transition probability must meet the following requirements:

- It must be a float (dtype `float64`).
- It must be non-negative.
- It must sum to 1 across all possible next states.

### Example 1: Health Transition Process

```python
import jax.numpy as jnp

def prob_exog_health_mother(health_father, params):
    prob_good_health = (health_father == 0) * 0.7 + (health_father == 1) * 0.3
    prob_medium_health = (health_father == 0) * 0.2 + (health_father == 1) * 0.5
    prob_bad_health = (health_father == 0) * 0.1 + (health_father == 1) * 0.2

    return jnp.array([prob_good_health, prob_medium_health, prob_bad_health])
```

## Specifying Exogenous Processes in Model Options

Each exogenous process should be defined in the `state_space` section of the model ```options```, mapping process names to transition functions and defining possible next states.

### Example 2: Specifying Exogenous Processes in `options`

```python
options = {
    "state_space": {
        "exogenous_processes": {
            "health_mother": {
                "transition": prob_exog_health_mother,
                "states": [0, 1, 2],
            },
            "job_offer": {
                "transition": job_offer_process_transition,
                "states": [0, 1],
            },
        },
    },
}
```

## Validating Exogenous Processes

To ensure correctness, the validation function can be used as implemented in the ```dcegm``` module:

```python
from dcegm.interface import validate_exogenous_processes

validate_exogenous_processes(model, params)
```

This function checks for compliance with probability constraints and verifies that transition matrices align with the model's state space. Possible errors and their respective messages include:

- **Non-float transition probabilities**: *"Exogenous process {exog_name} does not return float transition probabilities."*
- **Negative transition probabilities**: *"Exogenous process {exog_name} returns one or more negative transition probabilities."*
- **Probabilities greater than 1**: *"Exogenous process {exog_name} returns one or more transition probabilities > 1."*
- **Incorrect number of transitions**: *"Exogenous process {exog_name} does not return the correct number of transitions. Expected {expected}, got {actual}."*
- **Probabilities not summing to 1**: *"Exogenous process {exog_name} transition probabilities do not sum to 1."*

By following this specification, exogenous processes can be correctly integrated into the DCEGM framework, ensuring a consistent and reliable dynamic model setup.
