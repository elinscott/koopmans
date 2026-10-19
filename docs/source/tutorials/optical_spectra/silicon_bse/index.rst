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
each manifold, and then the ``wann2kc``, ``screen`` and ``ham`` steps of kcw.x, which the
:doc:`linear-response tutorial <../../band_structures/silicon_linear_response/index>`
covers. A fresh yambo chain follows. It starts with a self-consistent and a
non-self-consistent calculation for yambo's own ground state, then ``p2y`` builds yambo's
SAVE directory, a further step turns the ``ham`` step's eigenvalues into yambo's
quasiparticle database, and finally the BSE calculation itself runs.

The quasiparticle database is where the Koopmans correction enters yambo. It holds the
``ham`` step's Koopmans eigenvalues in the ``ndb.QP`` format that a BSE calculation more
commonly takes from a GW calculation. The BSE step reads it, so the Koopmans eigenvalues
set the transition energies and therefore the spectrum. The static screening is built
earlier, on the LDA energies, and the database covers bands 1 to 8 only while
``BndsRnXs`` runs to band 100.

*******************************
 The quasiparticle correction
*******************************

yambo's report on the BSE step states the gap before and after applying that database.
On the :math:`2\times2\times2` grid the indirect gap is 0.62 eV in LDA and 1.39 eV in
KI@LDA, a correction of 0.77 eV.

.. question:: What is the smallest *direct* transition, and where does it sit?

    3.38 eV, at :math:`\Gamma`. This calculation leaves out phonons, so absorption across
    the indirect gap, which needs a phonon to supply the missing momentum, is absent. The
    computed absorption therefore sets in at the lowest direct transition.

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
