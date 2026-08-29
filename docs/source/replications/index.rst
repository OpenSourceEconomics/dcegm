.. _replications:

Replications
============

This section walks through published studies re-implemented with `dcegm`. Unlike
the :doc:`guides <../guides/practitioner_guide>`, which teach the interface
through small worked examples, replications show the package applied to a
full-scale, published life-cycle model.

A replication in these docs is a **calibrated structural replication**: we
implement the paper's model faithfully and use its published parameter
estimates, but we do not re-run the paper's own estimation procedure (which
typically requires restricted-access microdata we don't have). The goal is
to demonstrate that `dcegm` reproduces the paper's *mechanism* and
*qualitative* implications, not to match its estimated moments point for
point. Each notebook states its simplifications explicitly.

.. toctree::
   :maxdepth: 1

   iskhakov_keane_2021.ipynb
