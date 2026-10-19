##########################################
 Band structures with spin-orbit coupling
##########################################

The KI@LDA band structure of bulk GaAs, including spin-orbit coupling. GaAs is a textbook
case: spin-orbit coupling splits the top of its valence band at :math:`\Gamma` into a
heavy-hole/light-hole pair and a split-off band, a splitting large enough
(:math:`\Delta_\text{SO} = 0.35` eV in experiment :cite:`Marrazzo2024`) that a
scalar-relativistic calculation misses it entirely.

Spin-orbit coupling changes three things about the calculation. The wavefunctions become
spinors, mixing what would otherwise be independent spin-up and spin-down states into a
single two-component object. Each spinless Wannier function therefore becomes two spinor
Wannier functions, each with two spin components, so a block that Wannierizes :math:`n`
bands without spin-orbit now Wannierizes :math:`2n`.
Because spin is no longer a separate label, there is one set of screening parameters rather
than one per channel, unlike the collinear case in the :doc:`magnetic solids tutorial
<../../orbital_energies/magnetic/index>`. And ``kcw.x`` runs its noncollinear chain, which
the KCW paper on noncollinear Koopmans functionals derives in full :cite:`Marrazzo2024`.

This tutorial ships its screening parameters precomputed rather than computing them from
scratch. Linear-response screening solves one constrained problem per group of equivalent
orbitals, and here that is five problems, each taking on the order of a day on four MPI
ranks — reproducing them is a production calculation, not something to run while reading a
tutorial.

******************
 Pseudopotentials
******************

``kcw.x``'s noncollinear implementation only supports the local-density approximation as its
base functional, and a spin-orbit calculation needs pseudopotentials generated with full
relativistic effects included. No installable ``aiida-pseudo`` family satisfies both:
`PseudoDojo <http://www.pseudo-dojo.org/>`_ publishes full-relativistic sets through
``aiida-pseudo`` only for PBE and PBEsol, and every LDA configuration that ``aiida-pseudo
install pseudo-dojo`` exposes is scalar-relativistic. This is the case the :doc:`installation page
<../../../installation>` has in mind when it says some pseudopotentials have to be installed
by hand.

PseudoDojo also publishes LDA full-relativistic pseudopotentials, which ``aiida-pseudo``'s
automatic installer does not offer. They live in a separate repository, one directory per
element:

.. code-block:: console

    $ mkdir LDA-FR-GaAs
    $ curl -Lo LDA-FR-GaAs/Ga-d_r.upf https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-LDA-FR-PDv0.4/main/Ga/Ga-d_r.upf
    $ curl -Lo LDA-FR-GaAs/As-d_r.upf https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-LDA-FR-PDv0.4/main/As/As-d_r.upf

Install the directory as a family, naming the pseudopotential type explicitly so that the
element is read from each file's own header rather than guessed from its filename (which
here does not follow the ``ELEMENT.EXTENSION`` convention ``aiida-pseudo`` otherwise
requires):

.. code-block:: console

    $ aiida-pseudo install family LDA-FR-GaAs LDA-FR-GaAs -P pseudo.upf

A family installed this way publishes no recommended cutoffs, so the input file states
``ecutwfc`` explicitly, as :doc:`the installation page <../../../installation>` describes.

.. warning::

    The repository's own README notes that this LDA table reuses the PBE full-relativistic
    generation inputs with only the exchange-correlation functional changed, and asks that
    it be used carefully. Treat it as adequate for learning the workflow, not as a citable
    pseudopotential for production spin-orbit work.

****************
 The input file
****************

Download :download:`gaas.yaml <gaas.yaml>` and place it in an empty directory. Here it is in
full:

.. literalinclude:: gaas.yaml
    :language: yaml

.. literalinclude:: gaas.yaml
    :language: yaml
    :start-at: spin: spin_orbit
    :end-at: spin: spin_orbit

selects the spinor, spin-orbit-coupled formulation described above.

.. literalinclude:: gaas.yaml
    :language: yaml
    :start-at: pseudo_library
    :end-at: pseudo_library

names the family installed above.

The ``calculator_parameters.wannier90.projections`` block lists four sets of Wannier
functions: gallium and arsenic's semicore :math:`3d` shells, the four valence bands
(arsenic-centered :math:`sp^3` bonds), and the four lowest conduction bands
(gallium-centered :math:`sp^3` antibonds). ``dis_win_max`` and ``dis_froz_max`` bound the
disentanglement for the last, empty block, exactly as they would without spin-orbit coupling
— the doubled orbital count changes how many Wannier functions each block produces, not how
the projections or the disentanglement window are specified.

.. literalinclude:: gaas.yaml
    :language: yaml
    :start-at: calculate_alpha
    :end-at: calculate_alpha

skips the screening calculations,

and feeds the final Hamiltonian the thirty-six screening parameters listed under
``alpha_guess`` instead, one per spinor Wannier function, in the order the projections
above define. A linear-response calculation on the same input produced them, with orbitals
whose wannier90 spreads agree within 0.05 Å\ :sup:`2` sharing one parameter. The values
therefore repeat within each such group: all ten As :math:`3d` orbitals share one value,
the ten Ga :math:`3d` orbitals split into a group of four and a group of six, and the
eight orbitals of each :math:`sp^3` block share one value apiece.

.. question:: Why do the As :math:`3d` orbitals form one group while the Ga :math:`3d` orbitals split in two?

    The crystal field splits the Ga :math:`3d` shell into two sets of Wannier functions
    with spreads of 0.32 and 0.37 Å\ :sup:`2`, which differ by slightly more than the
    0.05 Å\ :sup:`2` tolerance, so they receive separate screening parameters. In contrast,
    the As :math:`3d` spreads agree to within 0.001 Å\ :sup:`2`, so all ten share one.
    The grouping therefore depends on the tolerance as much as on the physics: a tolerance
    above the Ga :math:`3d` spread difference would put all ten Ga :math:`3d` orbitals in
    one group.

*************************
 Running the calculation
*************************

.. warning::

    Of all the ``Quantum ESPRESSO`` codes, this workflow needs ``pw.x``, ``wannier90.x``,
    ``pw2wannier90.x`` and ``kcw.x``.

Run it with:

.. code-block:: console

    $ koopmans run gaas.yaml

With the screening parameters supplied, the workflow skips straight from Wannierization to
the final Hamiltonian: one shared SCF and NSCF calculation, one Wannierization per
projection block, ``wann2kc`` to rotate the DFT Hamiltonian into the Wannier gauge, and
``ham`` to diagonalize the KI Hamiltonian these screening parameters define. There is no
``screen`` step at all — that step only exists when ``koopmans`` has to solve for the
screening parameters itself.

*************
 The results
*************

Plot the LDA and KI@LDA bands together, with the gap annotated, using ``koopmans plot
bandstructure``:

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

The final Hamiltonian step reports the KI@LDA bands at every k-point on the path. At
:math:`\Gamma` the valence band splits into the heavy-hole/light-hole pair and the
split-off band that spin-orbit coupling produces.

.. question:: How does the gap compare with experiment?

    The direct gap at :math:`\Gamma` that the figure annotates lies within 0.1 eV of the
    experimental 1.52 eV :cite:`Marrazzo2024`. In contrast, the LDA gap of 0.16 eV is too
    small by 1.36 eV, so KI@LDA removes more than ninety percent of the error. The spin-orbit
    splitting :math:`\Delta_\text{SO}` between the top of the valence band and the split-off
    band comes out at 0.35 eV, equal at two decimals to the experimental 0.35 eV.

    The same reference reports a gap of 1.51 eV from the same linear-response
    Wannier-function route :cite:`Marrazzo2024`. Its setup differs from this tutorial's in
    four ways:

    - a 12×12×12 k-point and 6×6×6 q-point grid, against 4×4×4 here
    - a wavefunction cutoff of 80 Ry, against 70 Ry here
    - a renormalized screening coefficient that includes a correction beyond second order
    - full-relativistic pseudopotentials generated by the authors from the scalar-relativistic
      PseudoDojo library

.. question:: How do the gap and the cost change from a scalar-relativistic KI@LDA calculation to a spin-orbit one?

    Spin-orbit coupling lowers the gap by about :math:`\Delta_\text{SO}/3 = 0.12` eV. The
    scalar-relativistic calculation, with its own screening parameters and the same structure
    and pseudopotential family, gives a KI@LDA gap of 1.52 eV and an LDA gap of 0.27 eV. The
    spin-orbit values are 1.41 eV and 0.16 eV, lower by 0.11 eV and 0.12 eV.

    The cause is the textbook shift of the valence-band maximum: spin-orbit coupling raises
    the :math:`j = 3/2` states at the top of the valence band by :math:`\Delta_\text{SO}/3`,
    so the gap shrinks by that amount. The ratio of :math:`\Delta_\text{SO}` to the gap plays
    no part, and the scalar-relativistic KI@LDA gap matches the experimental 1.52 eV only
    by accident, because it omits a reduction that the experimental gap includes.

    The cost grows too. On four MPI ranks each screening calculation takes about 4 to 5
    hours without spin-orbit coupling and 22 to 26 hours with it.
