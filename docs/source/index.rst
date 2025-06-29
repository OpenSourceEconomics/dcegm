dc-egm
======

**Note:** This is a pre-release version of the package. While the core features are in
place, the interface and functionality may still evolve. Feedback and contributions are
welcome.

`dcegm` is a Python package for solving and simulating finite-horizon stochastic
discrete-continuous dynamic choice models using the DC-EGM algorithm from Iskhakov,
Jørgensen, Rust, and Schjerning (QE, 2017).

The solution algorithm employs an extension of the Fast Upper-Envelope Scan (FUES) from
Dobrescu & Shanker (2022).


# Installation

You can install `dcegm` via PyPI or directly from GitHub. In your terminal, run:

```console
$ pip install dcegm
```

To install the latest development version directly from the GitHub repository, run:

```console
$ pip install git+https://github.com/OpenSourceEconomics/dcegm.git
```


.. toctree::
   :maxdepth: 1
   :hidden:

   installation



.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   guides/practitioner_guide
   guides/templates
   guides/minimal_example.ipynb



.. toctree::
   :maxdepth: 2
   :caption: Background
   :hidden:

   background/limitations
   background/literature
   background/interface_plots.ipynb
   background/specify_exogenous_processes.md
   background/sparsity_conditions


.. toctree::
   :maxdepth: 2
   :caption: Development
   :hidden:

   development/team
   development/changes
   development/roadmap


.. toctree::
   :maxdepth: 2
   :caption: dcegm API
   :hidden:

   genindex
   autoapi/index
