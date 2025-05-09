import dcegm.toy_models.cons_ret_model_dcegm_paper as crm_paper
import dcegm.toy_models.cons_ret_model_exog_ltc as crm_exog_ltc
import dcegm.toy_models.cons_ret_model_exog_ltc_and_job_offer as crm_exog_ltc_job_offer
import dcegm.toy_models.cons_ret_model_with_cont_exp as crm_cont_exp
import dcegm.toy_models.cons_ret_model_with_exp as crm_exp


def load_example_params_and_options(model_name):

    if (model_name == "dcegm_paper") or (
        model_name == "dcegm_paper_retirement_with_shocks"
    ):
        params = crm_paper.example_params_ret_model_with_shocks()
        options = crm_paper.example_options_ret_model_with_shocks()
    elif model_name == "dcegm_paper_retirement_no_shocks":
        params = crm_paper.example_params_retirement_no_shocks()
        options = crm_paper.example_options_retirement_no_shocks()
    elif model_name == "dcegm_paper_deaton":
        params = crm_paper.example_params_deaton()
        options = crm_paper.example_options_deaton()
    elif model_name == "with_exp":
        params = crm_exp.example_params()
        options = crm_exp.example_options()
    elif model_name == "with_cont_exp":
        params = crm_cont_exp.example_params()
        options = crm_cont_exp.example_options()
    elif model_name == "with_exog_ltc":
        params = crm_exog_ltc.example_params()
        options = crm_exog_ltc.example_options()
    elif model_name == "with_exog_ltc_and_job_offer":
        params = crm_exog_ltc_job_offer.example_params()
        options = crm_exog_ltc_job_offer.example_options()
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return params, options
