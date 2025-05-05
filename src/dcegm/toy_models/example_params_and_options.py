import dcegm.toy_models.cons_ret_model_dcegm_paper as crm_paper
import dcegm.toy_models.cons_ret_model_exog_ltc as crm_exog_ltc
import dcegm.toy_models.cons_ret_model_exog_ltc_and_job_offer as crm_exog_ltc_job_offer
import dcegm.toy_models.cons_ret_model_with_cont_exp as crm_cont_exp
import dcegm.toy_models.cons_ret_model_with_exp as crm_exp


def load_example_params_and_options(model_name):

    if model_name == "dcegm_paper":
        model_functions = {
            "utility_functions": crm_paper.create_utility_function_dict(),
            "state_space_functions": crm_paper.create_state_space_function_dict(),
            "utility_functions_final_period": crm_paper.create_final_period_utility_function_dict(),
            "budget_constraint": crm_paper.budget_constraint,
        }

    elif model_name == "with_exp":
        model_functions = {
            "utility_functions": crm_paper.create_utility_function_dict(),
            "state_space_functions": crm_exp.create_state_space_function_dict(),
            "utility_functions_final_period": crm_paper.create_final_period_utility_function_dict(),
            "budget_constraint": crm_exp.budget_constraint_exp,
        }

    elif model_name == "with_cont_exp":
        pass
    elif model_name == "with_exog_ltc":
        params = crm_exog_ltc.example_params()
        options = crm_exog_ltc.example_options()
    elif model_name == "with_exog_ltc_and_job_offer":
        params = crm_exog_ltc_job_offer.example_params()
        options = crm_exog_ltc_job_offer.example_options()
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return params, options
