def split_structure_and_batch_info(model_structure, batch_info):
    """Splits the model structure and batch info into static parts, which we can not jit
    compile and (large) arrays that we want to include in the function call for
    jitting."""

    struct_keys_not_for_jit = [
        "discrete_states_names",
        "state_names_without_stochastic",
        "stochastic_states_names",
    ]
    model_structure_non_jit = {
        key: model_structure[key] for key in struct_keys_not_for_jit
    }
    model_structure_jit = model_structure.copy()
    # Remove non-jittable items
    for key in struct_keys_not_for_jit:
        model_structure_jit.pop(key, None)

    # Remove non-jittable items from batch_info
    batch_info_jit = batch_info.copy()
    batch_info_non_jit = {
        "two_period_model": batch_info["two_period_model"],
    }
    batch_info_jit.pop("two_period_model", None)
    # If it is not a two period model, there is more
    if not batch_info["two_period_model"]:
        batch_info_non_jit["n_segments"] = batch_info["n_segments"]
        batch_info_jit.pop("n_segments", None)
        for batch_id in range(batch_info_non_jit["n_segments"]):
            batch_key = f"batches_info_segment_{batch_id}"
            batch_info_non_jit[batch_key] = {}
            batch_info_non_jit[batch_key]["batches_cover_all"] = batch_info[batch_key][
                "batches_cover_all"
            ]
            batch_info_jit[batch_key].pop("batches_cover_all", None)

    return (
        model_structure_jit,
        batch_info_jit,
        model_structure_non_jit,
        batch_info_non_jit,
    )


def merge_non_jit_and_jit_model_structure(model_structure_jit, model_structure_non_jit):
    """Generate one model_structure to handle inside the package functions."""
    model_structure = {
        **model_structure_jit,
        **model_structure_non_jit,
    }
    return model_structure


def merg_non_jit_batch_info_and_jit_batch_info(batch_info_jit, batch_info_non_jit):
    batch_info = {
        **batch_info_jit,
        "two_period_model": batch_info_non_jit["two_period_model"],
    }
    if not batch_info_non_jit["two_period_model"]:
        batch_info["n_segments"] = batch_info_non_jit["n_segments"]
        for batch_id in range(batch_info_non_jit["n_segments"]):
            batch_key = f"batches_info_segment_{batch_id}"
            batch_info[batch_key]["batches_cover_all"] = batch_info_non_jit[batch_key][
                "batches_cover_all"
            ]
    return batch_info
