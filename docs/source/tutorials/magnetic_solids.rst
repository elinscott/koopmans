#################
 Magnetic solids
#################

The band structure of a ferromagnetic solid. The whole chain fans out over the two spin
channels, so the input carries a set of Wannier projections for each of them alongside
the total magnetization, and you get a band structure per channel at the end.

Ferromagnets only, for the moment: an antiferromagnet needs its two magnetic sublattices
started antiparallel, and a structure whose sublattices are inequivalent cannot yet be
written in an input file — `issue #74
<https://github.com/elinscott/koopmans/issues/74>`_ tracks that gap.

.. note::

    This tutorial has not been written yet. It is tracked by `issue #76
    <https://github.com/elinscott/koopmans/issues/76>`_.
