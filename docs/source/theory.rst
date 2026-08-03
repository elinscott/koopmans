########
 Theory
########

The Kohn-Sham eigenvalues of an approximate density functional are not electron removal
and addition energies. They are the ingredients of a fictitious non-interacting system,
and with the semilocal functionals in common use they are also contaminated by
self-interaction: the energy of an orbital drifts as you change its occupancy, when it
should not. This is why band gaps computed from Kohn-Sham eigenvalues are too small, and
why those eigenvalues are unreliable as a spectrum.

Koopmans functionals repair this. They are corrective functionals, built on top of a
base density functional,

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
computed from first principles, one per orbital, not fitted. This is why a Koopmans
calculation is a workflow rather than a single run: most of the work is determining the
:math:`\alpha_i`, either from a series of constrained total-energy calculations, or from
linear response. The tutorials show both.

The result is spectral accuracy comparable to many-body perturbation theory at a cost
much closer to that of density-functional theory, while staying within a functional
formulation with a well-defined total energy.

**************
 Going deeper
**************

For the theory in full — the derivation of the functionals, the different flavors (KI,
KIPZ), the role of the variational orbitals, the algorithms, and benchmarks against
experiment — see

    E. B. Linscott, N. Colonna, R. De Gennaro, N. L. Nguyen, G. Borghi, A. Ferretti, I.
    Dabo and N. Marzari, *koopmans: An Open-Source Package for Accurately and
    Efficiently Predicting Spectral Properties with Koopmans Functionals*, J. Chem.
    Theory Comput. **19**, 7097 (2023), `doi:10.1021/acs.jctc.3c00652
    <https://doi.org/10.1021/acs.jctc.3c00652>`_.
