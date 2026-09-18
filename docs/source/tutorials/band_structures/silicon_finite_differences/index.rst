#################################################
 The band structure of silicon, from a supercell
#################################################

This tutorial applies the KI functional to a crystal. Two things change once the system
becomes periodic:

- instead of using Kohn-Sham states as the variational orbitals, we use Wannier functions
- when computing screening parameters from total-energy differences, the calculations
  require a supercell to avoid the spurious interaction of charged periodic images

Everything else is the workflow you already know from
:doc:`the previous tutorials on molecules <../../orbital_energies/index>`: initialize the
variational orbitals, compute one screening parameter per orbital from constrained
calculations, then evaluate the corrected functional.

***********************************
 Variational orbitals in a crystal
***********************************

A Koopmans correction acts orbital by orbital, and for bulk systems the orbitals it acts
on must be localized. Bloch states are not: a Bloch state is spread evenly over every unit
cell of the crystal, so removing "one electron from a Bloch state" removes an
infinitesimal amount of charge from each cell and the Koopmans condition becomes trivially
satisfied by Janak's theorem. Applying the Koopmans correction to localized orbitals fixes
this :cite:`Nguyen2018`.

Wannier functions :math:`w_{n\mathbf{R}}(\mathbf{r})` are a unitary transformation of the
Bloch states :math:`\psi_{n\mathbf{k}}(\mathbf{r})` into a localized basis:

.. math::

    w_{n \mathbf{R}}(\mathbf{r})
    = \frac{V}{(2 \pi)^{3}}
    \int_{\mathrm{BZ}}
    \left[ \sum_{m} U_{m n}^{(\mathbf{k})} \psi_{m \mathbf{k}}(\mathbf{r}) \right]
    e^{-i \mathbf{k}\cdot \mathbf{R}}
    \, \mathrm{d} \mathbf{k}

Each one belongs to a lattice site :math:`\mathbf{R}`, :math:`V` is the cell volume, and
:math:`U^{(\mathbf{k})}_{mn}` mixes the Bloch states at each :math:`\mathbf{k}`. The
transformation is exact for any unitary :math:`U^{(\mathbf{k})}_{mn}`, so the choice is
free — and we spend that freedom on localization, minimizing the spread

.. math::

    \Omega = \sum_{n}
    \left[
    \left\langle w_{n \mathbf{0}} \right| r^{2} \left| w_{n \mathbf{0}} \right\rangle
    - \left| \left\langle w_{n \mathbf{0}} \right| \mathbf{r} \left| w_{n \mathbf{0}} \right\rangle \right|^{2}
    \right]

The Wannier functions that minimize this metric are the maximally localized Wannier
functions (MLWFs) :cite:`Marzari2012`, and they are what ``koopmans`` uses as variational
orbitals for a periodic system. They are constructed by
`Wannier90 <http://www.wannier.org/>`_ together with ``pw.x`` and ``pw2wannier90.x``.

.. note::

    This tutorial does not teach Wannierization itself — the
    `Wannier90 tutorials <http://www.wannier.org/support/>`_ do that already. One point is
    worth flagging, though: most Wannier90 tutorials Wannierize the occupied and empty
    states together, but a Koopmans calculation requires separate representations of the
    occupied and empty manifolds. We therefore Wannierize the two manifolds separately.


****************
 The input file
****************

Download :download:`si.yaml <si.yaml>` and place it in an empty directory. Here it is in
full:

.. literalinclude:: si.yaml
    :language: yaml

.. warning::

    The cutoff and the k-point grid in this file are deliberately coarse, so that the
    workflow finishes in a reasonable time on a desktop. They are not converged.

    Likewise, turning off ``mp_correction`` keeps this tutorial short and its numbers
    comparable with published ones, but a charged periodic supercell really does need the
    correction. For production work leave it on and give the material's ``eps_inf``.

Most of the ``workflow`` block you have met before. One entry is new:

.. literalinclude:: si.yaml
    :language: yaml
    :start-at: init_orbitals
    :end-at: init_orbitals

replaces ozone's ``kohn-sham`` with maximally localized Wannier functions.

Periodic systems also require ``kpoints``:

.. literalinclude:: si.yaml
    :language: yaml
    :start-at: kpoints:
    :end-at: grid:

sets the Brillouin-zone sampling, and with it the size of the supercell the screening
calculations will run in — a :math:`2\times2\times2` grid means the
:math:`N \pm 1`-electron calculations run in an 8-cell supercell. Refining the sampling
therefore costs a great deal more here than it would in a plain DFT calculation.

Finally, the ``wannier90`` sub-block says which Wannier functions we want:

.. literalinclude:: si.yaml
    :language: yaml
    :start-at: projections:
    :end-before: dis_win_max

Each inner list is one block, and each block gets its own Wannierization. Both blocks
ask for the same four :math:`sp^3` hybrids on a silicon atom, pointing along its four
bonds. What distinguishes them is the bands they are built from: the first block is
restricted to the four filled bands, where those hybrids can only combine into bonding
orbitals; the second excludes the filled bands and disentangles four from the empty
states above, where the same hybrids pick out the antibonding partners.

.. question:: Why is ``alpha_guess`` 0.077 here, when ozone used 0.6?

    A screening parameter measures how much the rest of the system relaxes when you
    change one orbital's occupancy, and a solid relaxes far more than a molecule. To a
    first approximation :math:`\alpha \approx 1/\varepsilon_\infty`, and silicon's
    dielectric constant is around 12. Starting the loop near the answer saves iterations;
    it does not change where the loop converges to.

********************************
 Checking the Wannier functions
********************************

A Koopmans calculation on a solid is only as good as its variational orbitals, and the
Wannierization can be sensitive to the choice of projectors and windows. Let's run
it on its own first, to check it (this is why the input file has ``task: wannierize``):

.. code-block:: console

    $ koopmans run si.yaml

.. tip::

    ``pw.x`` steps parallelize well over k-points. Adding

    .. code-block:: yaml

        parallelization:
          pw:
            ntasks: 4
            npool: 4

    to the input file runs them on four MPI ranks divided into four k-point pools.

The progress table shows a self-consistent ``pw.x`` calculation, then a
non-self-consistent one that adds the empty bands, then a further ``pw.x`` calculation
that computes the DFT bands along the specified ``path``. One Wannierization per block
follows, each of which involves...

1. a ``wannier90.x`` preprocessing run,
2. a ``pw2wannier90.x`` run that extracts the overlaps and projections,
3. and the ``wannier90.x`` run that minimizes the spread.

The results land in ``si/``, one directory per step. The files worth opening are the
``aiida.wout`` files under the ``wannierize_occ_1`` and ``wannierize_emp_1`` steps, which
are Wannier90's own reports on each block. Each contains a table headed

.. code-block:: text

    *------------------------------- WANNIERISE ---------------------------------*
    +--------------------------------------------------------------------+<-- CONV
    | Iter  Delta Spread     RMS Gradient      Spread (Ang^2)      Time  |<-- CONV
    +--------------------------------------------------------------------+<-- CONV

whose rows are the minimization steps, and below it a ``Final State`` summary of where
the Wannier functions ended up.

.. question:: What do the converged Wannier functions look like?

    For the filled block:

    .. code-block:: text

        Final State
          WF centre and spread    1  ( -0.678815,  2.036446,  2.036446 )     1.02516597
          WF centre and spread    2  ( -0.678815,  0.678815,  0.678815 )     1.02516597
          WF centre and spread    3  ( -2.036446,  2.036446,  0.678815 )     1.02516597
          WF centre and spread    4  ( -2.036446,  0.678815,  2.036446 )     1.02516597

    Four Wannier functions, identical in spread and sitting at the midpoints of the four
    Si-Si bonds around an atom. This is the covalent bond of an undergraduate textbook,
    recovered from the Bloch states. The empty block gives four more at the same
    positions, roughly twice as spread out — the antibonding partners.

    That the four are degenerate is a check in itself: they are related by the crystal's
    symmetry, so a run that gives four different spreads has converged to something that
    is not the symmetric minimum.

To more rigorously assess the quality of the Wannier functions, let's inspect how they
interpolate the band structure. Good Wannier functions will give a faithful interpolation.
The ``koopmans plot bandstructure`` command can be used to plot the band structures:

.. code-block:: console

    $ koopmans plot bandstructure \
        si/02-bands --style x \
        si/03-wannierize_emp_1/01-wannier90/03-wannier90 \
        si/04-wannierize_occ_1/01-wannier90/03-wannier90

which writes ``bandstructure.png``. ``--style x`` draws the ``pw.x`` bands as crosses, and
leaves the two Wannier interpolations — one per block — as plain lines. The lines should
neatly traverse the crosses.

.. question:: Plot the band structure. What do you get?

    .. image:: bandstructure.png
        :align: center
        :width: 80%

    The Wannier interpolations (solid lines) follow the explicit evaluation (crosses)
    exactly at the k-points they were built from, but the interpolation is fairly
    inaccurate far away from those points! A :math:`2\times2\times2` grid is very coarse,
    so this is not a surprise.

.. question:: Increase ``grid`` to ``[4, 4, 4]`` and rerun. Do the Wannier functions get
    better or worse? Check the spreads and the band interpolation.

    The reported spread *grows* — from 1.03 to 1.62 Å² for the filled block, and to
    2.13 Å² on an :math:`8\times8\times8` grid. This is not the Wannier functions
    getting worse. A Wannier function lives in the Born-von Karman supercell that the
    k-point grid defines, and a :math:`2\times2\times2` grid gives it only eight cells to
    live in — too few to hold its tails. The Wannier functions remain degenerate and
    centered on the bonds.

    The interpolation gets better. A Wannier interpolation is exact on the k-points it was
    built from, so the :math:`2\times2\times2` is only exact at :math:`\Gamma`, :math:`X`
    and :math:`L`. The :math:`4\times4\times4` grid includes the mid-points of the
    :math:`\Gamma`-:math:`X` and :math:`\Gamma`-:math:`L` lines, so the interpolation is
    exact there, too.

.. question:: Read through ``koopmans plot bandstructure --help``, then polish the figure:
    give each series a name in the legend, and frame the energy window on the gap
    instead of the whole plotted range.

    There is no single right way to do this. One reasonable version:

    .. code-block:: console

        $ koopmans plot bandstructure \
            si/02-bands --style x --label "explicit evaluation" \
            si/03-wannierize_emp_1/01-wannier90/03-wannier90 --label "Wannier interpolation (emp)" \
            si/04-wannierize_occ_1/01-wannier90/03-wannier90 --label "Wannier interpolation (occ)" \
            --ylim -13 15

    ``--label`` names a folder on the legend the same way ``--style`` styles it: one per
    folder, written right after the folder it names. ``--ylim`` is not tied to any one
    folder — it sets the y-axis range for the whole figure, so it can sit anywhere on the
    command line.

********************
 The KI calculation
********************

Change one line of the input file:

.. code-block:: yaml

    workflow:
      task: singlepoint

and run it again. The new results replace the contents of ``si/``; nothing is lost, as the
KI workflow repeats the Wannierization (and its band structures) as its first step.

.. warning::

    We will use the :math:`2\times2\times2` grid for the rest of this tutorial, so the
    calculation runs quickly, even though we know the Wannier functions are not converged.

.. tip::

    ``koopmans`` records every calculation in a database, and if it sees the
    same input again it fetches the cached result, so the Wannierization you
    just ran will not be repeated.

-----------------------------------
 Initialization, and the supercell
-----------------------------------

Ozone's initialization was three ``kcp.x`` calculations that converged the base
functional's density. Silicon's is the Wannierization you just ran, followed by a stage
where we turn the :math:`k`-dependent primitive-cell Wannier functions into equivalent
quantities for a :math:`\Gamma`-only supercell.

------------------------------------
 Computing the screening parameters
------------------------------------

The supercell contains 64 variational orbitals: eight Wannier functions per primitive
cell, in eight cells, 32 filled and 32 empty. Computing a screening parameter for each
would be unnecessary — most of those orbitals are symmetrically equivalent to one another.

``koopmans`` can recognize this on its own by grouping together orbitals according to a
few metrics. The keywords behind this are ``group_orbitals_by`` and
``group_orbitals_tol``. In the provided input file we group the orbitals if their
self-Hartree energy (the default) is within 0.1 eV of one another.

From there the loop is the same as ever: a trial KI calculation at the guessed
:math:`\alpha_i`, then a constrained :math:`N-1` calculation for the filled
representative and an :math:`N+1` one for the empty representative, then the screening
parameters that follow.

*************
 The outputs
*************

The screening parameters end up in the final calculation's input files, one value per
orbital, in ``si/`` under the final KI step: ``file_alpharef.txt`` for the filled
orbitals and ``file_alpharef_empty.txt`` for the empty ones.

.. question:: What screening parameters do you get, and what do they tell you?

    Around 0.13 for the filled orbitals and 0.04 for the empty ones, from a starting
    guess of 0.077 for both.

    Two things are worth noticing. Both are far below the 0.66 to 0.79 that ozone gave: a
    covalent solid screens an added or removed charge far more effectively than an
    isolated molecule. The filled and empty orbitals come out a factor of three apart,
    so one :math:`\alpha` for the whole system would not do.

The orbital energies are in the final KI calculation's ``outputs/aiida.cpo``, and the
base-functional ones in the initialization output for comparison.

.. question:: How much does the KI correction change the spectrum?

    The initialization output — plain LDA — reports

    .. code-block:: text

        HOMO Eigenvalue (eV)

         3.8051

        LUMO Eigenvalue (eV)

         4.2218

        Electronic Gap (eV) =     0.4167

    and the final KI output

    .. code-block:: text

        HOMO Eigenvalue (eV)

         3.0646

        LUMO Eigenvalue (eV)

         4.3437

        Electronic Gap (eV) =     1.2791

    The correction pushes the filled states down and the empty states up, opening the
    gap by about 0.9 eV. This is the characteristic behavior of a Koopmans functional
    and the reason it repairs the band gaps that semilocal DFT underestimates.

.. warning::

    That 1.28 eV is *not* silicon's band gap, and neither is the LDA 0.42 eV. These
    are eigenvalues of the supercell at :math:`\Gamma`, which is the primitive cell's
    bands evaluated on the :math:`2\times2\times2` grid — the points :math:`\Gamma`,
    :math:`X` and :math:`L`, and nothing in between. Silicon's conduction minimum lies
    between :math:`\Gamma` and :math:`X` and is not among them.

**************************************
 From eigenvalues to a band structure
**************************************

Turning those supercell eigenvalues into a band structure means undoing the fold to a
supercell. This requires us to assign each supercell eigenvalue back to the primitive-cell
:math:`\mathbf{k}` it came from, and interpolating between them onto a path through the
Brillouin zone. This is the *unfold and interpolate* procedure of Ref.
:cite:`DeGennaro2022`.

The *unfolding* step involves assigning eigenstates to their corresponding
:math:`\mathbf{k}` point. This procedure is exact: the supercell's :math:`\Gamma`-point
eigenstates *are* the primitive cell's states on the :math:`2\times2\times2` grid, so each
one can be assigned its :math:`\mathbf{k}` without approximation.

The *interpolation* procedure involves mixing higher-resolution DFT data into the Koopmans
band structure. Written in the basis of Wannier functions, the KI Hamiltonian is the LDA
Hamiltonian plus a correction:

.. math::

    H^{\mathrm{KI}}_{\mathbf{R}} = H^{\mathrm{LDA}}_{\mathbf{R}} + \Delta H_{\mathbf{R}}

The Koopmans correction is short-ranged and interpolates well even from a coarse grid. The
LDA part does not — the first figure in this tutorial showed how poorly a
:math:`2\times2\times2` Wannier interpolation traces the LDA bands. One line in the
``kpoints`` block of the input file addresses this:

.. literalinclude:: si.yaml
    :language: yaml
    :start-at: kpoints:
    :end-at: smooth_interpolation_factor

``smooth_interpolation_factor: 4`` says to replace :math:`H^{\mathrm{LDA}}_{\mathbf{R}}`
with the same quantity computed on a grid four times denser in each direction, keeping
:math:`\Delta H_{\mathbf{R}}` from the supercell calculation. The denser grid requires a
non-self-consistent ``pw.x`` calculation and a Wannierization per block on an
:math:`8\times8\times8` grid, with the same projections and windows as before. This is not
expensive, as these are DFT calculations in the primitive cell, far cheaper than the
charged supercell calculations it complements.

These steps appear in the progress table either side of the final KI calculation.
``Smooth wannierization`` runs the dense-grid Wannierization, and ``Band interpolation``
extracts the KI Hamiltonian of each block from the final KI calculation, interpolates it
along ``path``, and joins the two blocks into one band structure. The result lands in
``si/outputs/band_structure`` as ``kpoints.npy`` and ``bands.npy``.

The same command as before draws it, here on top of the LDA bands that the dense-grid
Wannierization computed along the same path:

.. code-block:: console

    $ koopmans plot bandstructure \
        si/04-wannierize_smooth/01-bands --label LDA --style -- \
        si/06-interpolate_band_structure/05-build_band_structure --label "KI@LDA" --gap \
        --ylim -15 10 \
        -o ki_bandstructure.png

Each argument names one calculation directory inside the run, so each contributes one set
of bands with its own label and style; ``--gap`` marks the band gap of the set it follows.
Both sets are on the same energy scale, so no shift is needed to compare them: the zero
is the LDA valence-band maximum.

.. question:: What does the KI band structure look like, and what is the band gap?

    .. image:: ki_bandstructure.png
        :align: center
        :width: 80%

    The valence-band maximum sits at :math:`\Gamma` and the conduction-band minimum along
    :math:`\Gamma`-:math:`X`, about 80% of the way to :math:`X`: silicon's indirect gap,
    recovered from eight eigenvalues per supercell :math:`k`-point.

    The arrow on the figure puts a number on it: 1.18 eV for KI (against 0.29 eV for the
    LDA bands). Experiment gives 1.17 eV, and Ref. :cite:`Nguyen2018` reports 1.22 eV for
    KI with converged settings. Our :math:`2\times2\times2` grid is a long way from
    converged, so some of this agreement is luck.


.. tip::

    If you want the band structure data in an easy-to-digest format, you can extract the
    data to a json file by adding ``--data ki_bandstructure.json`` to the plot command.

.. question:: The final KI calculation reported a gap of 1.28 eV. Where is that number
    in the band structure?

    At :math:`X`. The conduction band there sits 1.28 eV above the valence-band maximum at
    :math:`\Gamma`, matching the supercell's HOMO-LUMO gap, as it must: the supercell's
    :math:`\Gamma`-point states are the primitive cell's states at :math:`\Gamma`,
    :math:`X` and :math:`L`, and among those the lowest empty one is at :math:`X`. The
    true minimum lies between the grid points, and only the interpolation can find it.

*********
 Up next
*********

Two settings govern the accuracy of this calculation, and both were left coarse here. The
plane-wave cutoff is moderately cheap to converge, but the :math:`k`-point grid is not: it
defines the size of the supercell used when computing the screening parameters, so
increasing the grid massively increases the cost of every constrained calculation.

The :doc:`next tutorial <../silicon_linear_response/index>` computes the screening
parameters by linear response instead, in the primitive cell, which avoids this expensive
supercell construction.
