def utility_function(
    consumption,
    choice,
    death,
    params,
):

    utility_consumption = (consumption ** (1 - params["rho"]) - 1) / (1 - params["rho"])

    working = choice == 1
    dead = death == 1
    working_and_not_dead = working * (1 - dead)

    utility = utility_consumption - working_and_not_dead * params["delta"]

    return utility
