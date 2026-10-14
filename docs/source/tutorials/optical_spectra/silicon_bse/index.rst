##################################################################
 Optical absorption of silicon, from the Bethe-Salpeter equation
##################################################################

This tutorial computes the optical absorption spectrum of bulk silicon with the
Bethe-Salpeter equation (BSE), on top of quasiparticle energies from a Koopmans
calculation. It assumes you already know how a periodic Koopmans calculation is built
from Wannier functions, taught in the :doc:`finite-differences tutorial
<../../band_structures/silicon_finite_differences/index>`. The screening parameters
here come from density-functional perturbation theory instead of total-energy
differences — the same linear-response route as the :doc:`sibling tutorial
<../../band_structures/silicon_linear_response/index>`, computed in the primitive
cell rather than a supercell, and this tutorial feeds the resulting quasiparticle
energies into `yambo <http://www.yambo-code.org/>`_.

**************************************
 What the Bethe-Salpeter equation adds
**************************************

The simplest way to compute an absorption spectrum is to promote an electron from an
occupied band to an empty one and add up every such transition allowed by the light's
polarization — the independent-particle approximation. It ignores that the promoted
electron and the hole it leaves behind are charged particles that attract one another.
The Bethe-Salpeter equation adds that electron-hole interaction back in, on top of a
screened Coulomb kernel, and diagonalizes the resulting two-particle Hamiltonian. Two
things change: transitions close in energy mix into new combinations (some brighter,
some dimmer than any single transition alone), and bound electron-hole pairs — excitons
— can appear below the lowest unbound transition, red-shifting the spectrum's onset.

The single-particle energies that go into either calculation matter as much as the
kernel: an absorption spectrum built from bare DFT eigenvalues inherits DFT's
underestimated gap, shifting the whole spectrum to the wrong energy before the
electron-hole interaction ever enters. yambo BSE calculations usually source their
input quasiparticle energies from a GW calculation, but this route sources them from a
Koopmans calculation instead.

***************
 Prerequisites
***************

Have ``p2y`` and ``yambo`` on your ``PATH`` when you run ``koopmans install``, as
described on the :doc:`installation page <../../../installation>`.

The ``bse`` task runs the DFPT chain from the previous tutorial and then a yambo
calculation on its eigenvalues. Three rules for the input follow from that, and
``koopmans`` checks each before it submits anything:

- ``workflow.screening_method`` is ``dfpt``, the only screening method that produces
  the Koopmans eigenvalues yambo reads.
- ``kpoints.path`` is not set, since a spectrum has no band path.
- ``calculator_parameters.yambo.BSEBands`` lies inside the Wannierized manifold, since
  a Koopmans correction exists only for the bands the DFPT chain Wannierizes.

****************
 The input file
****************

Download :download:`si-bse.yaml <si-bse.yaml>` and place it in an empty directory. Here
it is in full:

.. literalinclude:: si-bse.yaml
    :language: yaml

Most of this you have seen before: the ``atoms``, ``kpoints`` and ``wannier90``
projections are identical to the linear-response tutorial's silicon input, and
``workflow.task`` is the only new keyword in that block.

The new block is ``calculator_parameters.yambo``, which sets yambo's own runcard
variables under their own names. Six have no default:

.. literalinclude:: si-bse.yaml
    :language: yaml
    :start-at: yambo:
    :end-before: parallelization:

- ``BndsRnXs`` is the band range that builds the screened Coulomb interaction. This is a
  convergence parameter: raise it until the spectrum stops changing.
- ``NGsBlkXs`` is the G-vector cutoff of that interaction, in Ry. This is also a
  convergence parameter.
- ``BSEBands`` is the band range of the two-particle kernel. ``[1, 8]`` is the whole
  manifold the DFPT chain Wannierizes, four filled and four empty bands.
- ``BEnRange`` is the energy window of the spectrum, in eV.
- ``LongDrXs`` is the light polarization used for the screening.
- ``BLongDir`` is the light polarization used for the spectrum.

``BEnSteps`` sets the number of energy points in the window and ``BDmRange`` the
broadening that turns discrete transitions into a smooth curve, in eV. Both have
defaults, as do the other yambo variables listed in the :doc:`input file documentation
<../../../input_schema>`.

The ``parallelization`` block gives yambo 4 MPI ranks. Its ``bethe_salpeter`` entry
splits them as 2 over k-points and 2 over electron-hole pairs, and the two counts
multiply to the 4 ranks.

*************
 Running it
*************

.. code-block:: console

    $ koopmans run si-bse.yaml

The progress table walks through the same DFPT chain as the linear-response tutorial —
scf, nscf, a Wannierization per manifold, then wann2kc, screen and ham — followed by a
fresh yambo chain: an scf and nscf for yambo's own ground state, ``p2y`` to build its
SAVE directory, a step that turns the DFPT chain's kcw.x eigenvalues into yambo's own
quasiparticle database, and finally the BSE calculation itself.

.. question:: Which step brings the Koopmans correction into the yambo calculation?

    The step named "Quasiparticle database". It writes the ``ham`` step's Koopmans
    eigenvalues into a yambo ``ndb.QP`` database, in place of the GW database a BSE
    calculation more commonly reads, so every later yambo step applies the Koopmans
    correction.

*******************************
 The quasiparticle correction
*******************************

yambo's report on the BSE step states the gap before and after applying that database.
On the :math:`2\times2\times2` grid the indirect gap is 0.62 eV in LDA and 1.39 eV in
KI@LDA, a correction of 0.77 eV.

.. question:: What is the smallest *direct* transition, and where does it sit?

    3.38 eV, at :math:`\Gamma`. Light carries too little momentum to bridge the indirect
    gap, so absorption sets in at the direct transitions, and the exciton spectrum below
    is compared against this 3.38 eV.

**********************
 The optical spectrum
**********************

.. code-block:: console

    $ koopmans plot spectrum si-bse --label "KI@LDA" --xlim 2 6 -o spectrum.png

This draws the imaginary part of the dielectric function against energy, with the
independent-particle spectrum of the same run as a dashed curve:

.. image:: spectrum.png
    :align: center
    :width: 80%

.. question:: What does the electron-hole interaction do to the spectrum?

    It moves the main peak from 3.52 eV in the independent-particle spectrum down to
    3.12 eV and raises its height by a factor of 1.9. The electron and hole attract, so
    their bound pair lies below the 3.38 eV direct transition it is built from, and
    it takes oscillator strength from the transitions near it.

.. question:: What is the exciton's binding energy?

    yambo reports the lowest bound state at 3.119 eV. Measured from the 3.385 eV direct
    transition, the binding energy is 3.385 - 3.119 = 0.27 eV. The :math:`2\times2\times2`
    grid samples the pair's relative motion too coarsely to converge this value.

*********
 Up next
*********

To converge the spectrum, raise ``BndsRnXs``, ``NGsBlkXs`` and ``kpoints.grid`` in
``si-bse.yaml`` and rerun until the main peak stops moving.
