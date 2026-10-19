##################################################################
 Optical absorption of silicon, from the Bethe-Salpeter equation
##################################################################

This tutorial computes the optical absorption spectrum of bulk silicon with the
Bethe-Salpeter equation (BSE), on top of quasiparticle energies from a Koopmans
calculation, using `yambo <http://www.yambo-code.org/>`_.

****************
 The input file
****************

Download :download:`si-bse.yaml <si-bse.yaml>` and place it in an empty directory. Here
it is in full:

.. literalinclude:: si-bse.yaml
    :language: yaml

The ``workflow`` block asks for a KI calculation of the ``atoms`` below, built from
maximally localized Wannier functions, followed by a BSE spectrum on its eigenvalues.

The BSE calculation is primarily controlled by the ``yambo`` block, which sets yambo's own runcard variables:

.. literalinclude:: si-bse.yaml
    :language: yaml
    :start-at: yambo:
    :end-before: parallelization:

- ``BndsRnXs`` is the band range that builds the screened Coulomb interaction. This is a
  convergence parameter: raise it until the spectrum stops changing.
- ``NGsBlkXs`` is the G-vector cutoff of that interaction, in Ry. This is also a
  convergence parameter.
- ``BSEBands`` is the band range of the two-particle kernel. It must lie inside the
  Wannierized manifold, since a Koopmans correction exists only for those bands. ``[1, 8]``
  is the whole manifold, four filled and four empty bands.
- ``BEnRange`` is the energy window of the spectrum, in eV.
- ``BEnSteps`` is the number of energy points in that window.
- ``BDmRange`` is the broadening that turns discrete transitions into a smooth curve, in eV.
- ``LongDrXs`` is the light polarization used for the screening.
- ``BLongDir`` is the light polarization used for the spectrum.

For more details, see the :doc:`input file documentation
<../../../input_schema>`, under :class:`~koopmans.input_file.YamboBseParameters`.

The ``parallelization`` block gives yambo 4 MPI ranks.

*************
 Running it
*************

.. code-block:: console

    $ koopmans run si-bse.yaml

The progress table first walks through the Koopmans calculation: a self-consistent
calculation, a non-self-consistent one that adds the empty bands, a Wannierization of
each manifold, and then three kcw.x steps that convert the Wannier functions to its own
format, compute the screening parameters, and build and diagonalize the Koopmans
Hamiltonian. The :doc:`linear-response tutorial
<../../band_structures/silicon_linear_response/index>` covers this part. A fresh yambo
chain follows. It starts with a self-consistent and a non-self-consistent calculation for
yambo's own ground state, then converts that ground state to yambo's own format, writes
the eigenvalues of the Koopmans Hamiltonian into yambo's quasiparticle database, and
finally runs the BSE calculation itself.

The quasiparticle database is how the Koopmans correction enters yambo. It stores the
Koopmans eigenvalues in the format that a BSE calculation expects from a GW
calculation. The BSE step reads it, so the Koopmans eigenvalues set the transition
energies and therefore the spectrum.

*******************************
 The quasiparticle correction
*******************************

yambo's report on the BSE step records the band gaps before and after the Koopmans
eigenvalues are applied. Open it with

.. code-block:: console

    $ less si-bse/03-bse/03-bse/01-yambo/outputs/r-aiida.out_rim_cut_optics_dipoles_bss_bse_em1s

and find the two ``Gaps and Widths`` blocks. The first, marked ``[X]``, is built on the
LDA eigenvalues:

.. code-block:: text

    [X] Direct Gap                                    :  2.535045 [eV]
    [X] Direct Gap localized at k                     :  1
    [X] Indirect Gap                                  :  0.618534 [eV]
    [X] Indirect Gap between kpts                     :  1  3

and the second, marked ``[K+QP]``, on the Koopmans eigenvalues:

.. code-block:: text

    [K+QP] Direct Gap                                 :  3.384599 [eV]
    [K+QP] Direct Gap localized at k                  :  1
    [K+QP] Indirect Gap                               :  1.392000 [eV]
    [K+QP] Indirect Gap between kpts                  :  1  3

On the :math:`2\times2\times2` grid the indirect gap is 0.62 eV in LDA and 1.39 eV in
KI@LDA, a correction of 0.77 eV.

.. question:: What is the smallest *direct* transition, and where does it sit?

    3.38 eV, at k-point 1. The quasiparticle energies the report lists a little further
    down, under ``Eqp @ K [1]``, show the three highest filled states degenerate at 0 eV
    and the three lowest empty states degenerate at 3.38 eV. Those threefold degeneracies
    are the signature of :math:`\Gamma` in the diamond structure, so the smallest direct
    transition sits at :math:`\Gamma`. This calculation leaves out phonons, so absorption
    across the indirect gap, which needs a phonon to supply the missing momentum, is
    absent. The computed absorption therefore sets in at the lowest direct transition.

**********************
 The optical spectrum
**********************

.. code-block:: console

    $ koopmans plot spectrum si-bse --label "BSE (KI@LDA)" --xlim 2 5 -o spectrum.png

This draws the imaginary part of the dielectric function against energy. The solid curve
is the BSE spectrum. The dashed curve is the independent-particle spectrum, in which each
filled-to-empty transition absorbs on its own, computed from the same KI@LDA energies:

.. image:: spectrum.png
    :width: 480
    :align: center
    :alt: The imaginary part of the dielectric function of silicon between 2 and 5 eV;
        the BSE curve peaks near 3.1 eV, the independent-particle curve near 3.5 eV

.. question:: What does the electron-hole interaction do to the spectrum?

    It moves the main peak from about 3.5 eV in the independent-particle spectrum down
    to about 3.1 eV and roughly doubles its height. Both curves use the same KI@LDA
    energies, so this shift and this gain are the effect of the BSE kernel,
    :math:`2\bar{v} - W` :cite:`Onida2002`, alone. The kernel adds two terms to the
    independent-particle picture. :math:`W` is the screened attraction between the
    electron and the hole, so it lowers the energy of a pair and therefore shifts the
    peak down. :math:`\bar{v}` is the repulsive exchange, or local-field, term, and it
    mixes transitions close in energy so that some gain and some lose oscillator strength.

.. question:: What is the exciton's binding energy?

    The lowest direct transition lies at 3.385 eV, from the yambo report. The lowest
    exciton lies at 3.119 eV, from the excitonic states in yambo's BSE database as the
    aiida-yambo parser reads them. The difference, 0.27 eV, is how far the lowest exciton
    lies below the lowest direct transition. The
    :math:`2\times2\times2` grid samples the pair's relative motion too coarsely to
    converge this value.

*********
 Up next
*********

To converge the spectrum, raise ``BndsRnXs``, ``NGsBlkXs`` and ``kpoints.grid`` in
``si-bse.yaml`` and rerun until the main peak stops moving.
