#################################################
 The band structure of silicon, from a supercell
#################################################

This tutorial applies the KI functional to a crystal. Two things change once the system
becomes periodic:

- instead of using Kohn-Sham states as the variational orbitals, we use Wannier functions
- when computing screening parameters from total-energy differences, the calculations require a supercell to avoid the spurious interaction of charged periodic images

Everything else is the workflow you already know from :doc:`the previous tutorials on molecules <../../orbital_energies/index>`: initialize the variational orbitals,
compute one screening parameter per orbital from constrained calculations, then evaluate
the corrected functional.

.. note::

    Total-energy differences are not the only way to get the screening parameters. The
    :doc:`next tutorial <../silicon_linear_response/index>` computes them for the same
    system by linear response, which removes the need for a supercell throughout and is generally
    cheaper.

***********************************
 Variational orbitals in a crystal
***********************************

A Koopmans correction acts orbital by orbital, and for bulk systems the orbitals it acts on must be
localized. Bloch states are not: a Bloch state is spread evenly over every unit cell of
the crystal, so removing "one electron from a Bloch state" removes an infinitesimal
amount of charge from each cell and the Koopmans condition becomes trivially satisfied by Janak's theorem.
Applying the Koopmans correction to localized orbitals fixes this :cite:`Nguyen2018`.

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

The Wannier functions that minimize this metric are the maximally localized Wannier functions (MLWFs)
:cite:`Marzari2012`, and they are what ``koopmans`` uses as variational orbitals for a
periodic system. They are constructed by `Wannier90 <http://www.wannier.org/>`_ together
with ``pw.x`` and ``pw2wannier90.x`.

.. note::

    This tutorial does not teach Wannierization itself — the `Wannier90 tutorials
    <http://www.wannier.org/support/>`_ do that already. One point of departure is
    worth flagging, though: Most Wannier90 tutorials Wannierize occupied and empty states
    together. A Koopmans calculation requires separate representations of the occupied
    and empty manifolds. We therefore Wannierize the two manifolds separately.
    **blocks**.

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

    Likewise, turning off ``mp_correction`` keeps this tutorial short and its numbers comparable
    with published ones, but a charged periodic supercell really does need the
    correction. For production work leave it on and give the material's ``eps_inf``.

Most of the ``workflow`` block you have met before. Two entries are new:

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
calculations will run in — a :math:`2\times2\times2` grid means the :math:`N \pm 1`-electron calculations run in an 8-cell supercell.
Refining the sampling therefore costs a great deal more here than it would in a plain
DFT calculation.

Finally, the ``wannier90`` sub-block says which Wannier functions we want:

.. literalinclude:: si.yaml
    :language: yaml
    :start-at: projections:
    :end-before: dis_win_max

Each inner list is one block, and each block gets its own Wannierization. Both blocks
ask for four :math:`sp^3` hybrids on the bond-centre site: the first block takes the
four filled bonding combinations, the second the four empty antibonding ones.

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
Wannierization is the one step in this workflow that regularly needs adjusting. So run
it on its own first — that is what ``task: wannierize`` in the file is for.

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
non-self-consistent one that adds the empty bands, then one Wannierization per block —
each of which is itself a ``wannier90.x`` preprocessing run, a ``pw2wannier90.x`` run
that extracts the overlaps and projections, and the ``wannier90.x`` run that minimizes
the spread. A further ``pw.x`` calculation runs off the self-consistent density to get
the bands along ``path``; the last part of this section is what it is for.

The results land in ``si/``, one directory per step, exactly as :doc:`the ozone tutorial
<../../orbital_energies/ozone/automatically>` describes. The files worth opening are the
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
          WF centre and spread    1  ( -0.678815,  2.036446,  2.036446 )     1.02516698
          WF centre and spread    2  ( -0.678815,  0.678815,  0.678815 )     1.02516698
          WF centre and spread    3  ( -2.036446,  2.036446,  0.678815 )     1.02516698
          WF centre and spread    4  ( -2.036446,  0.678815,  2.036446 )     1.02516698

    Four Wannier functions, identical in spread and sitting at the midpoints of the four
    Si-Si bonds around an atom. This is the covalent bond of an undergraduate textbook,
    recovered from the Bloch states. The empty block gives four more at the same
    positions, roughly twice as spread out — the antibonding partners.

    That the four are degenerate is a check in itself: they are related by the crystal's
    symmetry, so a run that gives four different spreads has converged to something that
    is not the symmetric minimum.

.. question:: Increase ``grid`` to ``[4, 4, 4]`` and rerun. Do the Wannier functions get better or worse?

    The reported spread *grows* — from 1.025 to 1.617 Å² for the filled block, and to
    2.066 Å² on an :math:`8\times8\times8` grid. This is not the Wannier functions
    getting worse. A Wannier function lives in the Born-von Karman supercell that the
    k-point grid defines, and a :math:`2\times2\times2` grid gives it only eight cells to
    live in — too few to hold its tails. The coarse grid does not localize the orbital
    better; it truncates it, and reports a spread that is too small.

    This is the sense in which the grid in this file is unconverged, and it is worth
    knowing before you read the numbers at the end of this tutorial.

``aiida.wout`` reports what Wannier90 did. Whether the result still describes the
electronic structure ``pw.x`` computed is a separate question, and the band structure
answers it. Because ``si.yaml`` gives a ``path``, each block's Wannier functions are
interpolated along it, and the extra ``pw.x`` calculation supplies the same bands
directly. Draw them on one set of axes with

.. code-block:: console

    $ koopmans plot bandstructure \
        si/02-bands --style x \
        si/03-wannierize_emp_1/01-wannier90/03-wannier90 \
        si/04-wannierize_occ_1/01-wannier90/03-wannier90

which writes ``bandstructure.png`` — or add ``--show`` to open a window instead. A
``--style`` binds to the folder just before it, so this draws the ``pw.x`` bands as
crosses (``x``) while keeping each series' own color, and leaves the two Wannier
interpolations — one per block — as plain lines; crosses make it easy to see exactly
where a line runs through them and where it departs.

Two things are worth reading off it. *Which* bands the interpolation is obliged to
reproduce is what ``dis_froz_max`` sets: the empty block must span the states below it
and is free above, so tracking ``pw.x`` inside the frozen window and departing from it
higher up is the disentanglement doing exactly what it was told. *How closely* the
interpolation follows those bands between the k-points of the grid is set by the grid: a
Wannier interpolation is exact on the k-points it was built from, and
:math:`2\times2\times2` gives it eight of them to carry the whole path.

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

and run it again.

.. warning::

    The Wannierization runs again as the first stage of it. You will not pay for it
    twice: ``koopmans`` records every calculation in a database, and if it sees the
    same input again it fetches the cached result.

-----------------------------------
 Initialization, and the supercell
-----------------------------------

Ozone's initialization was three ``kcp.x`` calculations that converged the base
functional's density. Silicon's is the Wannierization you just ran, followed by a stage
called *fold to supercell*, whose ``wann2kcp.x`` and ``merge_evc.x`` calculations turn
the k-dependent primitive-cell Wannier functions into equivalent quantities for a
:math:`\Gamma`-only supercell.

.. question:: Why is a supercell needed at all?

    Because of what comes next. A screening parameter is computed from the energy cost
    of removing an electron from one variational orbital, and in the primitive cell there
    is no such thing: the orbital has a periodic image in every cell of the crystal, so
    emptying it empties every cell at once. That changes the charge density of the whole
    crystal; it does not probe one orbital.

    The way out is to make the cell big enough that one orbital can be emptied while the
    rest of the crystal stays put. The supercell commensurate with the k-point grid —
    eight primitive cells here — does exactly that, and sampling it at :math:`\Gamma`
    alone reproduces the primitive cell sampled on the full grid. ``kcp.x``, which runs
    every stage from here on, works only at :math:`\Gamma`, so this is the form it needs.

    This is also the whole reason the :doc:`linear-response route
    <../silicon_linear_response/index>` exists. It reaches the same screening parameters
    from the response of the primitive cell to an infinitesimal change in occupancy, and
    never builds a supercell.

------------------------------------
 Computing the screening parameters
------------------------------------

The supercell holds 64 variational orbitals: eight Wannier functions per primitive cell,
in eight cells, 32 filled and 32 empty. A constrained calculation for each would be
absurd, and unnecessary — most of those orbitals are images of one another.

``koopmans`` recognizes this on its own. The default for a Wannier-initialized
:math:`\Delta`\ SCF run groups orbitals by their self-Hartree energy, which is identical
for orbitals related by a lattice translation or by the crystal's point group; only one
representative of each group gets a constrained calculation, and every member of the
group takes the resulting :math:`\alpha`. Silicon's 32 filled orbitals are one bond
orbital seen 32 equivalent ways — four bonds in each of eight cells — and the 32 empty
ones likewise, so of the 64 just **two** are screened. The keywords behind this are
``group_orbitals_by`` and ``group_orbitals_tol``.

From there the loop is ozone's: a trial KI calculation at the guessed
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

    Around 0.133 for the filled orbitals and 0.040 for the empty ones, from a starting
    guess of 0.077 for both.

    Two things are worth noticing. Both are far below the 0.66 to 0.79 that ozone gave: a
    covalent solid screens an added or removed charge far more effectively than an
    isolated molecule, and the screening parameter measures exactly that. And filled and
    empty orbitals come out a factor of three apart, even though they sit on the same
    bonds — which is why one :math:`\alpha` for the whole system would not do.

The orbital energies are in the final KI calculation's ``outputs/aiida.cpo``, and the
base-functional ones in the initialization output for comparison.

.. question:: How much does the KI correction change the spectrum?

    The initialization output — plain LDA — reports

    .. code-block:: text

        HOMO Eigenvalue (eV)

         3.8051

        LUMO Eigenvalue (eV)

         4.2217

        Electronic Gap (eV) =     0.4167

    and the final KI output

    .. code-block:: text

        HOMO Eigenvalue (eV)

         3.0644

        LUMO Eigenvalue (eV)

         4.3437

        Electronic Gap (eV) =     1.2792

    The correction pushes the filled states down and the empty states up, opening the
    gap by about 0.9 eV. This is the characteristic behavior of a Koopmans functional
    and the reason it repairs the band gaps that semilocal DFT underestimates.

.. warning::

    That 1.2792 eV is *not* silicon's band gap, and neither is the LDA 0.4167 eV. These
    are eigenvalues of the supercell at :math:`\Gamma`, which is the primitive cell's
    bands evaluated on the :math:`2\times2\times2` grid — the points :math:`\Gamma`,
    :math:`X` and :math:`L`, and nothing in between. Silicon's conduction minimum lies
    between :math:`\Gamma` and :math:`X` and is not among them.

**************************************
 From eigenvalues to a band structure
**************************************

Turning those supercell eigenvalues into a band structure means undoing the fold:
assigning each supercell eigenvalue back to the primitive-cell :math:`\mathbf{k}` it came
from, and interpolating between them onto a path through the Brillouin zone. This is the
*unfold and interpolate* procedure of Ref. :cite:`DeGennaro2022`.

.. note::

    Unfolding is not yet available in this version of ``koopmans``. Until it is, a
    Koopmans band structure comes from the :doc:`linear-response route
    <../silicon_linear_response/index>`, which works in the primitive cell throughout and
    so needs no unfolding — this is a practical argument for that route on top of the
    cost argument above.

*******************
 Choosing a route
*******************

Both routes compute the same screening parameters. They differ in what they cost and in
what they can reach.

The :math:`\Delta`\ SCF route in this tutorial computes each screening parameter from an
actual constrained calculation, which makes no assumption of linearity. The price is the
supercell: refine the k-point grid and the supercell every screening calculation runs in
grows with it. It is the route to reach for on molecules and on small cells.

Linear response gets the same parameters from the primitive cell's response to an
infinitesimal change in occupancy. The cost does not grow with the k-point grid in the
same way, the primitive cell is never left, and the band structure follows directly.
:doc:`The next tutorial <../silicon_linear_response/index>` does exactly that for this
same silicon input.
