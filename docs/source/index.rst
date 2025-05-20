.. dc-egm documentation master file, created by
   sphinx-quickstart on Fri Sep 30 17:22:05 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

dc-egm`s Documentation
======================

Python implementation of the *Endogenous Grid Method* (EGM) and Discrete-Continuous EGM
(DC-EGM) algorithms for solving dynamic stochastic lifecycle models of consumption and
savings, including additional discrete choices.

**References**

1. Christopher D. Carroll (2006). `The method of endogenous gridpoints for solving dynamic stochastic optimization problems <http://www.sciencedirect.com/science/article/pii/S0165176505003368>`__. *Economics Letters*
2. Iskhakov, Jorgensen, Rust, and Schjerning (2017). `The Endogenous Grid Method for Discrete-Continuous Dynamic Choice Models with (or without) Taste Shocks <http://onlinelibrary.wiley.com/doi/10.3982/QE643/full>`__. *Quantitative Economics*

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   guides/installation
   guides/practitioner_guide
   genindex
   autoapi/index


.. toctree::
   :maxdepth: 2
   :caption: Background
   :hidden:

   background/limitations
   background/literature
   background/interface_plots.ipynb
   background/specify_exogenous_processes.ipynb
   background/specify_exogenous_processes.md
   background/timing_benchmarks.ipynb


.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   development/team
   development/changes
   development/roadmap
