import jax
from jax import numpy as jnp

from dcegm.pre_processing.model_structure.state_choice_space import (
    create_state_choice_space_and_child_state_mapping,
)
from dcegm.pre_processing.model_structure.state_space import create_state_space
from dcegm.pre_processing.shared import create_array_with_smallest_int_dtype


def create_model_structure(
    model_config,
    model_funcs,
):
    """Create dictionary of discrete state and state-choice objects for each period.

    Args:
        options (Dict[str, int]): Options dictionary.

    Returns:
        dict of np.ndarray: Dictionary containing period-specific
            state and state-choice objects, with the following keys:
            - "state_choice_mat" (np.ndarray)
            - "idx_state_of_state_choice" (np.ndarray)
            - "reshape_state_choice_vec_to_mat" (callable)
            - "transform_between_state_and_state_choice_vec" (callable)

    """
    print("Starting state space creation")
    state_space_objects = create_state_space(
        model_config=model_config,
        sparsity_condition=model_funcs["sparsity_condition"],
        debugging=False,
    )
    print("State space created.\n")
    print("Starting state-choice space creation and child state mapping.")

    state_choice_and_child_state_objects = (
        create_state_choice_space_and_child_state_mapping(
            model_config=model_config,
            state_specific_choice_set=model_funcs["state_specific_choice_set"],
            next_period_deterministic_state=model_funcs[
                "next_period_deterministic_state"
            ],
            state_space_arrays=state_space_objects,
        )
    )
    state_space_objects.pop("map_state_to_index_with_proxy")
    state_space_objects.pop("state_space_incl_proxies")

    model_structure = {
        **state_space_objects,
        **state_choice_and_child_state_objects,
        "choice_range": jnp.asarray(model_config["choices"]),
    }
    return jax.tree.map(create_array_with_smallest_int_dtype, model_structure)
