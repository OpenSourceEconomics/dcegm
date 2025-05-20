from jax import numpy as jnp


def check_budget_equation_and_return_wealth_plus_optional_aux(out, optional_aux=False):
    if isinstance(out, tuple):
        if len(out) == 2:
            # Check validity of first element
            if not check_valid_jnp_array(out[0]):
                raise ValueError(
                    "The first output of the budget equation must be a jnp.ndarray with one element."
                )

            check_valid_auxiliary_dict(out[1], "budget equation")
            if optional_aux:
                return out
            else:
                return out[0]

    elif isinstance(out, jnp.ndarray):
        # Check validity of jnp.ndarray
        if not check_valid_jnp_array(out):
            raise ValueError(
                "The output of the budget equation is an jnp.ndarray, "
                "but does contain more than one element."
            )
        if optional_aux:
            return out, {}
        else:
            return out
    else:
        raise ValueError(
            "The output of the budget equation must be either a jnp.ndarray or a tuple."
        )


def check_valid_auxiliary_dict(aux, func_name):
    """The function checks weather the auxiliary dictionary contains one value per
    key."""
    if isinstance(aux, dict):
        for key, value in aux.items():
            if not check_valid_jnp_array(value):
                raise ValueError(
                    f"The value for key '{key}' in the auxiliary dictionary of {func_name} is not a "
                    f"jnp.ndarray containing one element."
                )
    else:
        raise ValueError(f"The second output of {func_name} is not a dictionary.")


def check_valid_jnp_array(arr):
    if isinstance(arr, jnp.ndarray):
        # Check if array is 0 dimensional or 1 dimension with 1 element
        if arr.ndim == 0:
            return True
        elif arr.ndim == 1 and arr.shape[0] == 1:
            return True
        else:
            return False
    return False
