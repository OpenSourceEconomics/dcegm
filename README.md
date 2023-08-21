# DC-EGM

[![Continuous Integration Workflow](https://github.com/segsell/dc-egm/actions/workflows/main.yml/badge.svg)](https://github.com/segsell/dc-egm/actions/workflows/main.yml)
[![Codecov](https://codecov.io/gh/segsell/dc-egm/branch/main/graph/badge.svg)](https://codecov.io/gh/segsell/dc-egm)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Python implementation of the Discrete-Continuous Endogenous Grid Method (DC-EGM) for
solving dynamic stochastic lifecycle models of continuous (e.g. consumption-savings) and
additional discrete choices.

The solution algorithm employs an extension to the Fast Upper-Envelope Scan (FUES),
based on Dobrescu & Shanker (2022).

## References

1. Iskhakov, Jorgensen, Rust, & Schjerning (2017).
   [The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks](http://onlinelibrary.wiley.com/doi/10.3982/QE643/full).
   *Quantitative Economics*
1. Christopher D. Carroll (2006).
   [The method of endogenous gridpoints for solving dynamic stochastic optimization problems](http://www.sciencedirect.com/science/article/pii/S0165176505003368).
   *Economics Letters*
1. Loretti I. Dobrescu & Akshay Shanker (2022).
   [Fast Upper-Envelope Scan for Discrete-Continuous Dynamic Programming](https://dx.doi.org/10.2139/ssrn.4181302).
