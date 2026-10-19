##########################################
 Band structures with spin-orbit coupling
##########################################

This tutorial computes the KI@LDA band structure of bulk gallium arsenide with spin-orbit
coupling, with the screening parameters obtained by linear response as in the
:doc:`linear-response tutorial <../silicon_linear_response/index>`. Compared with that
tutorial, three things change:

- Spin-orbit coupling is included, which means the calculations run on spinor
  wavefunctions.
- The pseudopotentials must include spin-orbit coupling, and no automatically installable
  family does, so you install one by hand first.
- The screening parameters are supplied in the input file rather than computed, because
  each linear-response calculation takes about a day on four MPI ranks once the
  wavefunctions are spinors.

******************
 Pseudopotentials
******************

Spin-orbit coupling enters a plane-wave calculation through the pseudopotentials, so they
must be generated fully relativistically. ``kcw.x`` runs its spin-orbit chain for the
local-density approximation only, so the pseudopotentials must also be LDA. ``aiida-pseudo``
installs `PseudoDojo <http://www.pseudo-dojo.org/>`_'s fully relativistic tables for PBE and
PBEsol but not for LDA, so this is the case the :doc:`installation page
<../../../installation>` has in mind when it says some pseudopotentials are installed by
hand.

PseudoDojo does publish an LDA fully relativistic table, in a repository of its own with one
directory per element. Download gallium and arsenic:

.. code-block:: console

    $ mkdir LDA-FR-GaAs
    $ curl -Lo LDA-FR-GaAs/Ga-d_r.upf https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-LDA-FR-PDv0.4/main/Ga/Ga-d_r.upf
    $ curl -Lo LDA-FR-GaAs/As-d_r.upf https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-LDA-FR-PDv0.4/main/As/As-d_r.upf

and install the directory as a family:

.. code-block:: console

    $ aiida-pseudo install family LDA-FR-GaAs LDA-FR-GaAs -P pseudo.upf

.. note::

    The repository's README says this LDA table reuses the generation inputs of the PBE
    table with only the functional changed, and asks that it be used with care. It is
    adequate for learning the workflow; check it before production work.

****************
 The input file
****************

Download :download:`gaas.yaml <gaas.yaml>` and place it in an empty directory. Here it is in
full:

.. literalinclude:: gaas.yaml
    :language: yaml

Most of it reads as a silicon input would. What is specific to spin-orbit coupling:

.. literalinclude:: gaas.yaml
    :language: yaml
    :start-at: spin: spin_orbit
    :end-at: spin: spin_orbit

makes the wavefunctions spinors, two-component objects in which spin up and spin down mix.
Spin is then no longer a label on a band, so each band holds one electron rather than two.
Gallium and arsenic bring 28 valence electrons between them, so there are 28 filled bands,
and ``nbnd: 60`` leaves empty bands to disentangle among.

.. literalinclude:: gaas.yaml
    :language: yaml
    :start-at: pseudo_library
    :end-at: pseudo_library

names the family installed above. A family installed by hand publishes no recommended
cutoffs, so ``ecutwfc`` is stated explicitly.

The ``projections`` list four blocks of bands: the arsenic and gallium :math:`3d` shells,
the four valence bands, and the four lowest conduction bands. Each projection yields twice
the Wannier functions it would without spin-orbit coupling, one per spinor state, so the
four blocks produce 10, 10, 8 and 8 Wannier functions, 36 in all. The projections and the
disentanglement window are written exactly as they would be for a scalar-relativistic
calculation.

.. literalinclude:: gaas.yaml
    :language: yaml
    :start-at: calculate_alpha
    :end-at: calculate_alpha

skips the screening calculations, and the 36 values under ``alpha_guess`` are used in
their place, one per Wannier function in the order the projection blocks define. They were
computed by linear response from this same input with ``calculate_alpha: true`` and
``group_orbitals_by: spread``, under which Wannier functions whose spreads agree within
0.05 Å\ :sup:`2` share one parameter, which is why the values repeat in runs. Because spin
is not a separate label, there is one set of screening parameters, not one per spin channel
as in the :doc:`magnetic solids tutorial <../magnetic/index>`.

*************************
 Running the calculation
*************************

.. code-block:: console

    $ koopmans run gaas.yaml

The progress table shows a self-consistent calculation, a non-self-consistent one that
adds the empty bands, one Wannierization per projection block, and then two kcw.x steps:
one that rotates the Hamiltonian into the Wannier gauge and one that builds and
diagonalizes the KI Hamiltonian from the supplied screening parameters. There is no
screening step, since the screening parameters were given.

*************
 The results
*************

Plot the LDA and KI@LDA bands together, with the KI@LDA gap annotated:

.. code-block:: console

    $ koopmans plot bandstructure \
        gaas/02-wannierize/01-bands --label LDA --style k-- \
        gaas/03-dfpt/03-ham --label "KI@LDA" --style - --gap \
        --ylim -8 10 \
        -o gaas_bandstructure.svg

.. figure:: gaas_bandstructure.svg
    :width: 480
    :align: center

    KI@LDA band structure of GaAs along L-:math:`\Gamma`-X-W-L-K-:math:`\Gamma`, against
    the LDA base functional, with the direct gap at :math:`\Gamma` annotated.

.. question:: How does the band gap compare with experiment?

    The direct gap at :math:`\Gamma` is 1.41 eV in KI@LDA, against 1.52 eV in experiment
    :cite:`Marrazzo2024`. In contrast, LDA gives 0.16 eV, so the KI correction removes
    more than ninety percent of LDA's error. The same reference obtains 1.51 eV by the same
    route with a denser k-point grid, a higher cutoff, and pseudopotentials of its own, so
    the remaining 0.1 eV is a matter of convergence rather than of the functional.

.. question:: Where is spin-orbit coupling visible in the band structure?

    At the top of the valence band at :math:`\Gamma`. Without spin-orbit coupling the three
    highest valence bands are degenerate there. Spin-orbit coupling splits one of them off
    and pushes it down: the heavy-hole and light-hole bands stay degenerate at
    :math:`\Gamma`, and the split-off band lies :math:`\Delta_\text{SO}` below them. The
    KI eigenvalues kcw.x prints for :math:`\Gamma` in ``gaas/03-dfpt/03-ham/outputs/aiida.kho``
    show it directly:

    .. code-block:: text

        k = 0.0000 0.0000 0.0000     band energies (ev):
        ...
        KI     -10.0461   -10.0460   -10.0301   -10.0300    -5.1469    -5.1468     7.4170     7.4171
        KI       7.7639     7.7641     7.7641     7.7643     9.1761     9.1761    12.4146    12.4148

    Bands 25 to 28, the heavy and light holes, sit at 7.76 eV; bands 23 and 24, the
    split-off band, sit at 7.42 eV. The difference, 0.35 eV, is the spin-orbit splitting,
    and experiment gives 0.35 eV :cite:`Marrazzo2024`. The LDA lines above it in the same
    block give the same splitting, so the KI correction moves the valence bands together
    and leaves the splitting to the pseudopotentials.

.. question:: Why do the gallium :math:`3d` orbitals carry two screening parameters while the arsenic ones carry one?

    The run that computed them grouped Wannier functions whose spreads agree within
    0.05 Å\ :sup:`2`, the default tolerance of ``group_orbitals_by: spread``. Each
    Wannierization step writes its spreads to its
    ``outputs/output_parameters.json``. The ten arsenic :math:`3d` functions all have a
    spread of 0.24 Å\ :sup:`2`, so they share one parameter. The crystal field splits the
    gallium :math:`3d` shell into four functions of spread 0.32 Å\ :sup:`2` and six of
    0.37 Å\ :sup:`2`, which differ by more than the tolerance, so they receive two. The
    grouping therefore depends on ``group_orbitals_tol`` as much as on the physics: a
    tolerance above that difference would give all ten one parameter.

*********
 Up next
*********

To compute the screening parameters yourself, set ``calculate_alpha: true``, remove
``alpha_guess``, and add ``group_orbitals_by: spread``. One linear-response calculation
then runs per group of equivalent Wannier functions, five in all, each taking about a day
on four MPI ranks.
