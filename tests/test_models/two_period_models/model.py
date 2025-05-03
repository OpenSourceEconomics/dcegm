from jax import numpy as jnp


def prob_exog_health(health, params):
    prob_good_health = (health == 0) * 0.7 + (health == 1) * 0.3 + (health == 2) * 0.2
    prob_medium_health = (health == 0) * 0.2 + (health == 1) * 0.5 + (health == 2) * 0.2
    prob_bad_health = (health == 0) * 0.1 + (health == 1) * 0.2 + (health == 2) * 0.6

    return jnp.array([prob_good_health, prob_medium_health, prob_bad_health])
