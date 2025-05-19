import dcegm.toy_models.cons_ret_model_dcegm_paper as crm_paper
import dcegm.toy_models.cons_ret_model_stochastic_ltc as crm_stochastic_ltc
import dcegm.toy_models.cons_ret_model_stochastic_ltc_and_job_offer as crm_stochastic_ltc_job_offer
import dcegm.toy_models.cons_ret_model_with_cont_exp as crm_cont_exp
import dcegm.toy_models.cons_ret_model_with_exp as crm_exp


def load_example_params_model_specs_and_config(model_name):

    if (model_name == "dcegm_paper") or (
        model_name == "dcegm_paper_retirement_with_shocks"
    ):
        params = crm_paper.example_params_ret_model_with_shocks()
        model_specs = crm_paper.example_model_specs_ret_model_with_shocks()
        model_config = crm_paper.example_model_config_ret_model_with_shocks()
    elif model_name == "dcegm_paper_retirement_no_shocks":
        params = crm_paper.example_params_retirement_no_shocks()
        model_specs = crm_paper.example_model_specs_retirement_no_shocks()
        model_config = crm_paper.example_model_config_retirement_no_shocks()
    elif model_name == "dcegm_paper_deaton":
        params = crm_paper.example_params_deaton()
        model_specs = crm_paper.example_model_specs_deaton()
        model_config = crm_paper.example_model_config_deaton()
    elif model_name == "with_exp":
        params = crm_exp.example_params()
        model_specs = crm_exp.example_model_specs()
        model_config = crm_exp.example_model_config()
    elif model_name == "with_cont_exp":
        params = crm_cont_exp.example_params()
        model_specs = crm_cont_exp.example_model_specs()
        model_config = crm_cont_exp.example_model_config()
    elif model_name == "with_stochastic_ltc":
        params = crm_stochastic_ltc.example_params()
        model_specs = crm_stochastic_ltc.example_model_specs()
        model_config = crm_stochastic_ltc.example_model_config()
    elif model_name == "with_stochastic_ltc_and_job_offer":
        params = crm_stochastic_ltc_job_offer.example_params()
        model_specs = crm_stochastic_ltc_job_offer.example_model_specs()
        model_config = crm_stochastic_ltc_job_offer.example_model_config()
    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return params, model_specs, model_config
