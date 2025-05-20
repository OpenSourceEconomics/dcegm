# DC-EGM

[![Continuous Integration Workflow](https://github.com/OpenSourceEconomics/dcegm/actions/workflows/main.yml/badge.svg)](https://github.com/OpenSourceEconomics/dcegm/actions/workflows/main.yml)
[![Codecov](https://codecov.io/gh/OpenSourceEconomics/dcegm/branch/main/graph/badge.svg)](https://app.codecov.io/gh/OpenSourceEconomics/dcegm)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/OpenSourceEconomics/dcegm/main.svg)](https://results.pre-commit.ci/latest/github/OpenSourceEconomics/dcegm/main)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!--
Python implementation of the Discrete-Continuous Endogenous Grid Method (DC-EGM) for
solving dynamic stochastic lifecycle models of continuous (e.g. consumption-savings) and
additional discrete choices. -->

Python package for the solution and simulation of finite-horizon discrete-continuous
dynamic choice models based on the DC-EGM algorithm from Iskhakov, Jorgensen, Rust, and
Schjerning (QE, 2017).

The solution algorithm employs an extension of the Fast Upper-Envelope Scan (FUES) from
Dobrescu & Shanker (2022).

## References

1. Iskhakov, Jorgensen, Rust, & Schjerning (2017).
   [The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks](http://onlinelibrary.wiley.com/doi/10.3982/QE643/full).
   *Quantitative Economics*
1. Christopher D. Carroll (2006).
   [The method of endogenous gridpoints for solving dynamic stochastic optimization problems](http://www.sciencedirect.com/science/article/pii/S0165176505003368).
   *Economics Letters*
1. Loretti I. Dobrescu & Akshay Shanker (2022).
   [Fast Upper-Envelope Scan for Discrete-Continuous Dynamic Programming](https://dx.doi.org/10.2139/ssrn.4181302).
