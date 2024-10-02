from toy_models.cons_ret_model_dcegm_paper.utility_functions import (
    inverse_marginal_utility_crra,
    marginal_utility_crra,
    marginal_utility_final_consume_all,
    utility_crra,
    utility_final_consume_all,
)


def create_utility_function_dict():
    """Create dictionary with utility functions from dcegm paper.

    Returns:
        utility_functions (dict): Dictionary with utility functions.

    """
    return {
        "utility": utility_crra,
        "marginal_utility": marginal_utility_crra,
        "inverse_marginal_utility": inverse_marginal_utility_crra,
    }


def create_final_period_utility_function_dict():
    """Create dictionary with utility functions for the final period.

    Returns:
        utility_functions_final_period (dict): Dictionary with utility functions
            for the final period.

    """
    return {
        "utility": utility_final_consume_all,
        "marginal_utility": marginal_utility_final_consume_all,
    }
