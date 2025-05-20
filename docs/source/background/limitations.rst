.. _limitations:

Limitations
===========

`dc-egm` is a versatile Python package capable of solving and simulating a large class of economic models flexibly and efficiently..

However, there are limitations to what kind of features can be implemented and how complex models can become to be feasible to solve.

Below we discuss what kind of models `dc-egm` is designed for and importantly what limitations to be aware of when implementing a model.

What can dc-egm do?
---------------------

The class of economic models suitable for the DC-EGM (Discrete-Continuous Endogenous Grid Method) framework are dynamic discrete-continuous choice models where agents make decisions that include both continuous controls (e.g., consumption or savings) and discrete choices (e.g., labor supply, retirement, or occupational states), potentially influenced by idiosyncratic taste shocks. These models are defined by recursive Bellman equations, where the optimal policy must be computed over a joint choice set. Traditional solution methods, such as nested fixed-point or brute-force grid search, are computationally intensive for these settings. The DC-EGM adapts the endogenous grid method to this structure by efficiently solving the Euler equation for the continuous choice conditional on each discrete alternative, thus avoiding costly root-finding or interpolation on the value function. This makes DC-EGM particularly well-suited for life-cycle models with a modest number of discrete alternatives, especially when taste shocks (e.g., Type I extreme value) allow the use of smooth choice probabilities. In contrast to purely discrete dynamic choice models (e.g., Rust-style models), or models with only continuous controls, DC-EGM addresses hybrid choice problems with substantial gains in speed and accuracy.
