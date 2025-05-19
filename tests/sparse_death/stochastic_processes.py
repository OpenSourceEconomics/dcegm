import jax.numpy as jnp


def prob_survival():
    return jnp.array([0.1, 0.9])


def job_offer(choice):
    job_offer_probs = jnp.array([0.4, 0.6])
    job_destruction_probs = jnp.array([0.02, 0.98])

    job_offer_next = choice * job_destruction_probs + (1 - choice) * job_offer_probs
    return job_offer_next
