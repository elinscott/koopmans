########
 Theory
########

The Kohn-Sham eigenvalues of an approximate density functional are not electron removal
and addition energies. They are the ingredients of a fictitious non-interacting system,
and with the semilocal functionals in common use they are also contaminated by
self-interaction: the energy of an orbital drifts as you change its occupancy, when it
should not. This is why band gaps computed from Kohn-Sham eigenvalues are too small, and
why those eigenvalues are unreliable as a spectrum.

Koopmans functionals repair this :cite:`Dabo2010,Borghi2014`. They are corrective
functionals, built on top of a base density functional,

.. math::

    E^\text{Koopmans} = E^\text{DFT} + \sum_i \alpha_i \Pi^u_i

where the correction :math:`\Pi^u_i` is constructed so that the energy of orbital
:math:`i` is linear in its occupancy :math:`f_i`. Imposing that condition on every
orbital in the system — the generalized piecewise linearity condition — makes each
orbital energy equal to the total-energy difference for adding or removing an electron
from that orbital. The orbital energies become quantities you can compare with an
experimental spectrum.

The :math:`\alpha_i` are screening parameters, and they account for the relaxation of
everything else in the system when the occupancy of orbital :math:`i` changes. They are
computed from first principles, rather than being fitted to experiment or to some other
level of theory. This is why a Koopmans calculation is a workflow rather than a single
run: we need to determine the :math:`\alpha_i`, either from a series of constrained
total-energy calculations :cite:`Nguyen2018`, or from linear response
:cite:`Colonna2018,Colonna2019`. The tutorials show examples for both.

The result is spectral accuracy comparable to many-body perturbation theory at a much
lower computational cost, while staying within a functional formulation with a
well-defined total energy.

**************
 Going deeper
**************

For the theory in full — the derivation of the functionals, the different flavors (KI,
KIPZ), the role of the variational orbitals, the algorithms, and more, see :cite:`Linscott2023`.

The :doc:`references page <references>` also lists many papers that you might be interested to read.
