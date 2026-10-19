##########################################
 Band structures with spin-orbit coupling
##########################################

This tutorial computes the KI@LDA band structure of bulk gallium arsenide with spin-orbit
coupling. The screening parameters come from linear response, as in the
:doc:`linear-response tutorial <../silicon_linear_response/index>`. Three things differ
from that tutorial:

- Spin-orbit coupling is included, so the calculations run on spinor wavefunctions.
- Spin-orbit coupling has to be built into the pseudopotentials, and no family that
  ``koopmans`` installs automatically has it together with the LDA, so you install one by
  hand first.
- The screening parameters are given in the input file rather than computed, because with
  spinor wavefunctions each linear-response calculation takes about a day on four MPI
  ranks.

******************
 Pseudopotentials
******************

A calculation with spin-orbit coupling needs fully relativistic pseudopotentials.
``kcw.x`` supports only the LDA when spin-orbit coupling is enabled, so the
pseudopotentials must also be LDA. ``koopmans`` installs `PseudoDojo
<http://www.pseudo-dojo.org/>`_'s fully relativistic families automatically for PBE and
PBEsol, but not for LDA, so this family has to be installed by hand.

Thankfully, PseudoDojo publishes fully relativistic LDA pseudopotentials,
and ``aiida-pseudo`` can install any directory of pseudopotentials as a family. First
download gallium and arsenic from PseudoDojo:

.. code-block:: console

    $ mkdir LDA-FR-GaAs
    $ curl -Lo LDA-FR-GaAs/Ga-d_r.upf https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-LDA-FR-PDv0.4/main/Ga/Ga-d_r.upf
    $ curl -Lo LDA-FR-GaAs/As-d_r.upf https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-LDA-FR-PDv0.4/main/As/As-d_r.upf

Then install the directory as a family, as described :ref:`on the installation page
<installing-pseudopotentials>`:

.. code-block:: console

    $ aiida-pseudo install family LDA-FR-GaAs LDA-FR-GaAs -P pseudo.upf

.. warning::

    The repository's README says this LDA table reuses the generation inputs of the PBE
    table with only the functional changed, and asks that it be used with care. It is
    adequate for a tutorial such as this; check it before using it in production.

****************
 The input file
****************

Download :download:`gaas.yaml <gaas.yaml>` and place it in an empty directory. Here it is in
full:

.. literalinclude:: gaas.yaml
    :language: yaml

Most of it reads as a silicon input would. The differences all follow from spin-orbit
coupling. Spin-orbit coupling couples an electron's spin to its orbital motion. Spin up and
spin down are then no longer conserved separately, so the wavefunctions become spinors:
two-component objects in which the two mix. This is enabled with

.. literalinclude:: gaas.yaml
    :language: yaml
    :start-at: spin: spin_orbit
    :end-at: spin: spin_orbit

Spin is then no longer a label on a band, so each band holds one electron rather than two.

.. question:: How many filled bands does GaAs have in this calculation?

    The gallium and arsenic pseudopotentials have pseudoatomic charges of 13
    (:math:`3d^{10} 4s^2 4p^1`) and 15 (:math:`3d^{10} 4s^2 4p^3`) respectively, so the cell
    holds 28 valence electrons. Each band holds one electron, so 28 bands are filled.
    Without spin-orbit coupling the same electrons would fill 14.

.. literalinclude:: gaas.yaml
    :language: yaml
    :start-at: pseudo_library
    :end-at: pseudo_library

names the family installed above.

The ``projections`` list four blocks of bands: the arsenic and gallium :math:`3d` shells,
the four valence bands, and the four lowest conduction bands. Each projection now yields
two Wannier functions where it would yield one without spin-orbit coupling, one per spinor
state. So the four blocks produce 10, 10, 8 and 8 Wannier functions, 36 in all. The
projections and the disentanglement window themselves are written as for a
scalar-relativistic calculation.

.. literalinclude:: gaas.yaml
    :language: yaml
    :start-at: calculate_alpha
    :end-at: calculate_alpha

skips the screening calculations. In their place, the 36 values under ``alpha_guess`` are
used, one per Wannier function, in the order the projection blocks define. They were
computed by linear response from this same input with ``calculate_alpha: true`` and
``group_orbitals_by: spread``. Under that setting, Wannier functions whose spreads agree
within 0.05 Å\ :sup:`2` share one parameter, which is why the values repeat in runs. Spin
is not a separate label, so there is one set of screening parameters rather than one per
spin channel as in the :doc:`magnetic solids tutorial <../magnetic/index>`.

*************************
 Running the calculation
*************************

.. code-block:: console

    $ koopmans run gaas.yaml

The progress table shows a self-consistent calculation, a non-self-consistent one that
adds the empty bands, and one Wannierization per projection block. Then come two kcw.x
steps: one rotates the Hamiltonian into the Wannier gauge, and one builds and
diagonalizes the KI Hamiltonian from the supplied screening parameters. There is no
screening step, because the screening parameters were given.

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

    The KI@LDA band structure of GaAs with spin-orbit coupling, against LDA.

.. question:: How does the band gap compare with experiment?

    The direct gap at :math:`\Gamma` is 1.41 eV in KI@LDA. Experiment gives 1.52 eV
    :cite:`Marrazzo2024`. In contrast, LDA gives 0.16 eV, so the KI correction removes
    more than ninety percent of LDA's error. The same reference obtains 1.51 eV by the same
    route, with a denser k-point grid, a higher cutoff and pseudopotentials of its own. So
    the remaining 0.1 eV comes from convergence rather than from the functional.

.. question:: Where is spin-orbit coupling visible in the band structure?

    At the top of the valence band at :math:`\Gamma`. Without spin-orbit coupling the three
    highest valence bands are degenerate there, each holding two electrons. With it, two of
    them, the heavy-hole and light-hole bands, stay degenerate at :math:`\Gamma`, and the
    third, the split-off band, is pushed down by :math:`\Delta_\text{SO}`. The KI
    eigenvalues that kcw.x prints for :math:`\Gamma` in
    ``gaas/03-dfpt/03-ham/outputs/aiida.kho`` show this directly:

    .. code-block:: text

        k = 0.0000 0.0000 0.0000     band energies (ev):
        ...
        KI     -10.0461   -10.0460   -10.0301   -10.0300    -5.1469    -5.1468     7.4170     7.4171
        KI       7.7639     7.7641     7.7641     7.7643     9.1761     9.1761    12.4146    12.4148

    In this calculation each band holds one electron, so the heavy and light holes are
    bands 25 to 28, at 7.76 eV, and the split-off band is bands 23 and 24, at 7.42 eV.
    Their difference is the spin-orbit splitting, 0.35 eV, and experiment gives 0.35 eV
    :cite:`Marrazzo2024`. The LDA lines in the same block give the same 0.35 eV, so the KI
    correction shifts these bands together and leaves the splitting unchanged.

.. question:: Why do the gallium :math:`3d` orbitals carry two screening parameters while the arsenic ones carry one?

    The run that computed them grouped Wannier functions by spread: functions whose
    spreads agree within 0.05 Å\ :sup:`2`, the default tolerance of ``group_orbitals_by:
    spread``, share one parameter. Each Wannierization step writes its spreads to its
    ``outputs/output_parameters.json``. The ten arsenic :math:`3d` functions all have a
    spread of 0.24 Å\ :sup:`2`, so they share one parameter. The crystal field splits the
    gallium :math:`3d` shell into four functions of spread 0.32 Å\ :sup:`2` and six of
    0.37 Å\ :sup:`2`. These differ by more than the tolerance, so they receive two
    parameters. The grouping therefore depends on ``group_orbitals_tol`` as much as on the
    physics: a tolerance above that difference would give all ten gallium functions one
    parameter.

*********
 Up next
*********

To compute the screening parameters yourself, set ``calculate_alpha: true``, remove
``alpha_guess``, and add ``group_orbitals_by: spread``. One linear-response calculation
then runs per group of equivalent Wannier functions, five in all, each taking about a day
on four MPI ranks.
