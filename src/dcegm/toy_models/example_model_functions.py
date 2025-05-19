import dcegm.toy_models.cons_ret_model_dcegm_paper as crm_paper
import dcegm.toy_models.cons_ret_model_stochastic_ltc as crm_exog_ltc
import dcegm.toy_models.cons_ret_model_stochastic_ltc_and_job_offer as crm_exog_ltc_and_job_offer
import dcegm.toy_models.cons_ret_model_with_cont_exp as crm_cont_exp
import dcegm.toy_models.cons_ret_model_with_exp as crm_exp


def load_example_model_functions(model_name):

    if "dcegm_paper" in model_name:
        # We choose the model with retirement as the standard model and return the model functions
        # if just "dcegm_paper" is requested
        model_functions = {
            "utility_functions": crm_paper.create_utility_function_dict(),
            "state_space_functions": crm_paper.create_state_space_function_dict(),
            "utility_functions_final_period": crm_paper.create_final_period_utility_function_dict(),
            "budget_constraint": crm_paper.budget_constraint,
        }
        if model_name == "dcegm_paper_deaton":
            model_functions.pop("state_space_functions")

    elif model_name == "with_exp":
        model_functions = {
            "utility_functions": crm_paper.create_utility_function_dict(),
            "state_space_functions": crm_exp.create_state_space_function_dict(),
            "utility_functions_final_period": crm_paper.create_final_period_utility_function_dict(),
            "budget_constraint": crm_exp.budget_constraint_exp,
        }

    elif model_name == "with_cont_exp":
        model_functions = {
            "utility_functions": crm_paper.create_utility_function_dict(),
            "state_space_functions": crm_cont_exp.create_state_space_function_dict(),
            "utility_functions_final_period": crm_paper.create_final_period_utility_function_dict(),
            "budget_constraint": crm_cont_exp.budget_constraint_cont_exp,
        }
    elif model_name == "with_stochastic_ltc":
        model_functions = {
            "utility_functions": crm_paper.create_utility_function_dict(),
            "state_space_functions": crm_paper.create_state_space_function_dict(),
            "utility_functions_final_period": crm_paper.create_final_period_utility_function_dict(),
            "budget_constraint": crm_exog_ltc.budget_equation_with_ltc,
            "stochastic_states_transitions": crm_exog_ltc.create_stochastic_states_transition(),
        }
    elif model_name == "with_stochastic_ltc_and_job_offer":
        model_functions = {
            "utility_functions": crm_paper.create_utility_function_dict(),
            "state_space_functions": crm_paper.create_state_space_function_dict(),
            "utility_functions_final_period": crm_paper.create_final_period_utility_function_dict(),
            "budget_constraint": crm_exog_ltc.budget_equation_with_ltc,
            "stochastic_states_transitions": crm_exog_ltc_and_job_offer.create_stochastic_states_transition(),
        }
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return model_functions
