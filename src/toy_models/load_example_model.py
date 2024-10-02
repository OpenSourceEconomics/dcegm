import toy_models.cons_ret_model_dcegm_paper as crm_paper
import toy_models.cons_ret_model_with_cont_exp as crm_cont_exp
import toy_models.cons_ret_model_with_exp as crm_exp


def load_example_models(model_name):

    if model_name == "dcegm_paper":
        model_functions = {
            "state_space_functions": crm_paper.create_state_space_function_dict(),
            "utility_functions": crm_paper.create_utility_function_dict(),
            "final_period_utility_functions": crm_paper.create_final_period_utility_function_dict(),
            "budget_constraint": crm_paper.budget_constraint,
        }

    elif model_name == "with_exp":
        model_functions = {
            "state_space_functions": crm_exp.create_state_space_function_dict(),
            "utility_functions": crm_paper.create_utility_function_dict(),
            "final_period_utility_functions": crm_paper.create_final_period_utility_function_dict(),
            "budget_constraint": crm_exp.budget_constraint_exp,
        }

    elif model_name == "with_cont_exp":
        model_functions = {
            "state_space_functions": crm_cont_exp.create_state_space_function_dict(),
            "utility_functions": crm_paper.create_utility_function_dict(),
            "final_period_utility_functions": crm_paper.create_final_period_utility_function_dict(),
            "budget_constraint": crm_cont_exp.budget_constraint_cont_exp,
        }
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return model_functions
