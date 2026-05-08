.. _roadmap:

Roadmap
=======

Below is a list of features that are currently planned for `dcegm` as well as associated issues and pull requests.

- **Alternative batching** (`#200 <https://github.com/OpenSourceEconomics/dcegm/issues/200>`_)
  The current solver relies on batching to handle large state spaces efficiently. We can add an alternative execution path that avoids batching, which may be more suitable for certain model specifications and/or and option to fill in dummy states for larger batches.

- **Generalized income timing** (`#199 <https://github.com/OpenSourceEconomics/dcegm/issues/199>`_)
  The classic `dcegm` implementation applies the income shock at the beginning of the period so that income is a function of last period's choice. Consequently, wealth determining the consumption choice is the same across all discrete choices. We plan to make income timing more flexible, allowing users to specify when income is received relative to the continuous (consumption) decisions.

- **Improve documentation and interface for user specified functions** (`#94 <https://github.com/OpenSourceEconomics/dcegm/issues/94>`_)
  The current interface already includes various error messages and warnings for misspecified user functions, but the interface could be improved to further help the user specify their model. In particular we want to add information that documents which inputs user functions can and cannot handle and some common sources of errors in user functions.


Recent Developments
===================

April 2026:  **Multiple deterministic continuous state variables**

Previously, `dcegm` only supported a single deterministic continuous state variable. We extended the solver to handle multiple deterministic continuous states, including support for the Druedahl–Jørgensen upper envelope algorithm. See PR `#197 <https://github.com/OpenSourceEconomics/dcegm/pull/197>`_ & `#198 <https://github.com/OpenSourceEconomics/dcegm/pull/198>`_ for more information.
