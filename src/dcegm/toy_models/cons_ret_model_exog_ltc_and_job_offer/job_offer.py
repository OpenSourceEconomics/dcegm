from jax import numpy as jnp


def prob_exog_job_offer(
    job_offer,
    params,
):
    prob_job_offer = (job_offer == 0) * params["job_offer_constant"] + (
        job_offer == 1
    ) * (params["job_offer_constant"] + params["job_offer_type_two"])
    prob_no_job_offer = 1 - prob_job_offer

    return jnp.array([prob_no_job_offer, prob_job_offer])
