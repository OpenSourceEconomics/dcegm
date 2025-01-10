from dcegm.pre_processing.setup_model import setup_model


def test_n_periods():

    test_model = setup_model(
        options={"state_space": {"n_periods": 1}},
        utility_functions={},
        utility_functions_final_period={},
        budget_constraint=lambda x: x,
    )
