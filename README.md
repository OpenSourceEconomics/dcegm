# dc-egm

[![Continuous Integration Workflow](https://github.com/OpenSourceEconomics/dcegm/actions/workflows/main.yml/badge.svg)](https://github.com/OpenSourceEconomics/dcegm/actions/workflows/main.yml)
[![image](https://readthedocs.org/projects/dcegm/badge/?version=latest)](https://dcegm.readthedocs.io/en/latest)
[![Codecov](https://codecov.io/gh/OpenSourceEconomics/dcegm/branch/main/graph/badge.svg)](https://app.codecov.io/gh/OpenSourceEconomics/dcegm)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/OpenSourceEconomics/dcegm/main.svg)](https://results.pre-commit.ci/latest/github/OpenSourceEconomics/dcegm/main)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!--
Python implementation of the Discrete-Continuous Endogenous Grid Method (DC-EGM) for
solving dynamic stochastic lifecycle models of continuous (e.g. consumption-savings) and
additional discrete choices. -->

**Note:** This is a pre-release version of the package. While the core features are in
place, the interface and functionality may still evolve. Feedback and contributions are
welcome.

`dcegm` is a Python package for solving and simulating finite-horizon stochastic
discrete-continuous dynamic choice models using the DC-EGM algorithm from Iskhakov,
Jørgensen, Rust, and Schjerning (QE, 2017).

The solution algorithm employs an extension of the Fast Upper-Envelope Scan (FUES) from
Dobrescu & Shanker (2022).

## Installation

You can install `dcegm` via PyPI or directly from GitHub. In your terminal, run:

```console
$ pip install dcegm
```

To install the latest development version directly from the GitHub repository, run:

```console
$ pip install git+https://github.com/OpenSourceEconomics/dcegm.git
```

## Documentation

The documentation is hosted at https://dcegm.readthedocs.io

## References

1. Christopher D. Carroll (2006).
   [The method of endogenous gridpoints for solving dynamic stochastic optimization problems](http://www.sciencedirect.com/science/article/pii/S0165176505003368).
   *Economics Letters*
1. Iskhakov, Jorgensen, Rust, & Schjerning (2017).
   [The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks](http://onlinelibrary.wiley.com/doi/10.3982/QE643/full).
   *Quantitative Economics*
1. Loretti I. Dobrescu & Akshay Shanker (2022).
   [Fast Upper-Envelope Scan for Discrete-Continuous Dynamic Programming](https://dx.doi.org/10.2139/ssrn.4181302).

<!-- ## Citation

If you use dcegm for your research, please do not forget to cite it.

```bibtex
@Unpublished{BleschGsell2025,
   Title = {dcegm: A GPU-accelerated python implementation of the Discrete-Contiunous Endogenous Grid Method},
   Author = {Maximilian Blesch & Sebastian Gsell},
   Year = {2025},
   Url = {https://github.com/OpenSourceEconomics/dcegm} }
``` -->
