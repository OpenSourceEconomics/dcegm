.. _limitations:

Limitations
===========

`dc-egm` is a versatile Python package capable of solving and simulating a large class of economic models flexibly and efficiently..

However, there are limitations to what kind of features can be implemented and how complex models can become to be feasible to solve.

Below we discuss what kind of models `dc-egm` is designed for and importantly what limitations to be aware of when implementing a model.

What can dc-egm do?
---------------------

The package follows Ishakov et al. (2017) and implements the discrete-continuous endogenous grid method (DC-EGM) for solving dynamic stochastic optimization problems with both discrete and continuous choices. Our code originated as a Python replication of their Matlab code and has since been extended to include additional features and improvements (such as used in Iskhakov and Keane (2021)).

The class of economic models suitable for implementation with `dc-egm` are dynamic discrete-continuous choice models where agents make decisions that include both continuous controls (e.g., consumption or savings) and discrete choices (e.g., labor supply, retirement, or occupational states), potentially influenced by idiosyncratic taste shocks.

These models are defined by recursive Bellman equations, where the optimal policy must be computed over a joint choice set. Traditional solution methods, such as nested fixed-point or brute-force grid search, are computationally intensive for these settings. The DC-EGM adapts the endogenous grid method to this structure by efficiently solving the Euler equation for the continuous choice conditional on each discrete alternative, thus avoiding costly root-finding or interpolation on the value function.

This makes DC-EGM particularly well-suited for life-cycle models with a modest number of discrete alternatives, especially when taste shocks (e.g., Type I extreme value) allow the use of smooth choice probabilities. In contrast to purely discrete dynamic choice models (e.g., Rust-style models), or models with only continuous controls, DC-EGM addresses hybrid choice problems with substantial gains in speed and accuracy.


What can dc-egm not be used for?
---------------------------------

- Purely discrete choice models (see e.g. Keane and Wolpin; 1997)
- Models with more than two continuous state variables.
- Models with more than one continuous state variable with a normally distributed idiosyncratic shock.


**References**

- Iskhakov, Jørgensen, Rust, & Schjerning (2017). `The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks <http://onlinelibrary.wiley.com/doi/10.3982/QE643/full>`_. *Quantitative Economics*

- Iskhakov, F., & Keane, M. (2021). Effects of taxes and safety net pensions on life-cycle labor supply, savings and human capital: The case of Australia. *Journal of Econometrics*, 223(2), 401–432.

- Keane, M. P., & Wolpin, K. I. (1997). The career decisions of young men. *Journal of Political Economy*, 105(3), 473–522.
