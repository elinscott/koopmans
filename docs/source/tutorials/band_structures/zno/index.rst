###############################################
 ZnO, and choosing Wannier blocks for yourself
###############################################

This tutorial calculates the KI band structure of wurtzite ZnO — a wide-gap
semiconductor that ordinary density-functional theory describes very badly, putting its
gap at a fraction of the measured one.

The screening parameters come from linear response in the primitive cell, evaluated on
an explicit mesh of k-points. The :doc:`silicon tutorials
<../silicon_finite_differences/index>` set out that choice: a supercell and total-energy
differences on one side, :doc:`linear response <../silicon_linear_response/index>` and
the primitive cell on the other. This page assumes their vocabulary — Wannier functions,
projections, screening parameters — and concentrates on what ZnO adds to it.

What it adds is the Wannierization. Silicon's four filled bands are one group of bands,
well separated from everything else, and one set of projections describes them. ZnO's
twenty-six are not: they fall into four groups at very different energies, and the two
empty bands we also need are buried in a continuum of higher ones. So the manifold has
to be broken into blocks, each Wannierized on its own. This tutorial gives you the
blocks; the :ref:`last section <zno-projections>` shows how to work them out yourself.

****************
 The input file
****************

Download :download:`zno.yaml <zno.yaml>` and place it in an empty directory. Here it is
in full:

.. literalinclude:: zno.yaml
    :language: yaml

.. warning::

    The cutoff and the k-point mesh in this file are deliberately rough, so that the
    workflow finishes in minutes. They are not converged! Use this file to learn the
    input format, not as something to copy-paste for production work.

Two lines in the ``workflow`` block choose the method:

.. literalinclude:: zno.yaml
    :language: yaml
    :start-at: screening_method
    :end-at: init_orbitals

The first asks for the screening parameters to be computed by linear response, with
``kcw.x``, rather than by the constrained total-energy calculations of the molecular
tutorials. The second says the variational orbitals — the orbitals the correction acts
on — are maximally localized Wannier functions. A periodic solid has no localized
orbitals to hand: the Kohn-Sham states are extended Bloch waves, and a correction
applied to them would not be a correction to anything localized in space. Wannier
functions supply them.

.. literalinclude:: zno.yaml
    :language: yaml
    :start-at: calculate_alpha
    :end-at: calculate_alpha

skips the calculation of the screening parameters altogether and uses the twenty-eight
values in ``alpha_guess`` instead, which were computed by exactly this workflow. That is
the expensive part of a Koopmans calculation, and leaving it out makes this tutorial
short. Set it back to ``true`` and the workflow will compute them for you.

.. question:: Why twenty-eight screening parameters?

    One per Wannier function, and the projections define twenty-eight of them: two Zn
    3s, six Zn 3p, two O 2s and sixteen from the hybridized Zn 3d and O 2p block, which
    is twenty-six filled ones, plus the two empty Zn 4s. Screening is an
    orbital-by-orbital affair — every orbital sits in a different environment and the
    rest of the system relaxes differently when you change its occupancy.

.. literalinclude:: zno.yaml
    :language: yaml
    :start-at: gb_correction
    :end-at: eps_inf

deal with a periodic-boundary artifact. Computing a screening parameter means perturbing
the occupancy of one orbital, which puts a charge in every periodic image of the cell,
and the Coulomb interaction between those images diverges as :math:`q \to 0`. The
Gygi-Baldereschi scheme removes the divergence, and it needs the dielectric constant of
the material, given here as a literal 5.3. Writing ``eps_inf: auto`` instead has the
workflow compute it, with ``ph.x``, at the cost of an extra step — see the
:doc:`dielectric constants tutorial <../../dielectric_constants/index>`.

The ``kpoints`` block carries two independent things: ``grid`` is the mesh the
ground-state calculation and the Wannier functions are built on, and ``path`` is the
line through the Brillouin zone the final band structure is interpolated along, written
in the usual letters for the high-symmetry points with ``G`` for :math:`\Gamma`.

Finally the projections, one entry per block of bands:

.. literalinclude:: zno.yaml
    :language: yaml
    :start-at: projections
    :end-at: Zn 4s

Each entry is a list of atom-and-angular-momentum pairs, and each defines its own
Wannierization: the first four cover the filled bands, the fifth the two empty Zn 4s
bands. The two ``dis_`` keywords that follow are energy windows for that last block; the
:ref:`last section <zno-projections>` explains where all of these numbers come from.

.. question:: Why not Wannierize all twenty-six filled bands at once?

    Because the Wannierization minimizes the spatial spread of the orbitals and nothing
    else. It is free to mix any bands you give it, and left to itself it will mix
    orbitals lying tens of electronvolts apart — the Zn 3s with the O 2p, say — if that
    buys it a little localization. The resulting Wannier functions are not recognizable
    atomic-like orbitals, and the Koopmans band structure built from them suffers.
    Splitting the manifold into blocks of bands that are well separated in energy
    forbids the mixing.

*************************
 Running the calculation
*************************

.. warning::

    Make sure you have installed ``koopmans``: see :doc:`here <../../../installation>`
    for more details.

    This workflow needs four executables registered — ``pw.x``, ``pw2wannier90.x``,
    ``wannier90.x`` and ``kcw.x``. ``kcw.x`` runs every stage of the linear-response
    part.

Run the calculation with

.. code-block:: console

    $ koopmans run zno.yaml

The terminal shows a live progress table that grows as the workflow proceeds. At the end
it reads

.. code-block:: text

     Step                                                                      Status
     Singlepoint Dfpt Workflow                                               finished
       SCF Nscf                                                              finished
         Nscf                                                                finished
       Wannierize                                                            finished
         Wannierize Occ 1                                                    finished
           Wannier 90                                                        finished
             Wannier 90 Pp                                                   finished
             Pw 2 Wannier 90                                                 finished
         Wannierize Occ 2                                                    finished
           ...
         Wannierize Occ 3                                                    finished
           ...
         Wannierize Occ 4                                                    finished
           ...
         Wannierize Emp                                                      finished
           ...
       Dfpt                                                                  finished
         Wann 2 KC                                                           finished
         Ham                                                                 finished

    Workflow completed successfully!

Three stages, in order:

---------------------------------
 The ground state (``SCF Nscf``)
---------------------------------

An LDA calculation, self-consistent on the 4×4×4 mesh and then repeated
non-self-consistently for the fifty-two bands the Wannierization needs. LDA rather than
PBE because of the pseudopotential library the input file names: the base functional a
Koopmans calculation corrects is the one its pseudopotentials were generated with.

---------------------------------
 Wannierization (``Wannierize``)
---------------------------------

One ``Wannierize`` step per block, five in all, each of which projects onto that block's
projections and then minimizes the spread. They are independent of one another and run
concurrently. The four filled blocks take exactly as many bands as they have
projections, so there is nothing for them to choose; the empty block has two projections
and twenty-six bands to find them in, and the two energy windows are what tells it where
to look.

The five sets of Wannier functions are then stitched into one manifold — a
block-diagonal unitary matrix and one list of Wannier centres — which is what the next
stage reads.

----------------------------
 Linear response (``Dfpt``)
----------------------------

``Wann 2 KC`` converts the ``Wannier90`` output into the format ``kcw.x`` reads. Had we
asked for the screening parameters to be computed, a ``Screen`` step would follow, one
linear-response calculation per orbital; with ``calculate_alpha: false`` the workflow
goes straight to ``Ham``, which builds the Koopmans Hamiltonian, and — because the input
file gave a band path — interpolates it along that path.

*************
 The outputs
*************

The results land in a directory named after the input file, here ``zno/``, laid out to
mirror the outline above:

.. code-block:: text

    zno
    ├── 01-scf_nscf
    │   ├── 01-scf
    │   └── 02-nscf
    ├── 02-wannierize
    │   ├── 01-wannierize_occ_1
    │   │   └── 01-wannier90
    │   │       ├── 01-wannier90_pp
    │   │       ├── 02-pw2wannier90
    │   │       └── 03-wannier90
    │   ├── ...
    │   └── 05-wannierize_emp
    │       └── ...
    ├── 03-dfpt
    │   ├── 01-prepare_kcw_wannier_files
    │   ├── 02-wann2kc
    │   └── 03-ham
    └── README

One directory per step, numbered in the order the steps ran, each holding the input
files the engine generated in ``inputs/`` and everything the calculation wrote in
``outputs/``.

.. note::

    The engine also keeps its own complete record of every calculation in its database —
    that is how an interrupted workflow resumes where it left off, and how a repeated
    calculation is served from cache instead of running twice. The ``zno/`` directory is
    a plain-file export of that record for you to read.

Two files are worth opening straight away. ``zno/03-dfpt/03-ham/inputs/file_alpharef.txt``
lists the screening parameters the final Hamiltonian was built with, one per Wannier
function, in the order the projections define them — the twenty-eight numbers from the
input file, in this run. And ``zno/03-dfpt/03-ham/outputs/aiida.kho`` is the ``kcw.x``
output, which ends with the interpolated eigenvalues at each point of the band path.

**************************
 Interpreting the results
**************************

.. question:: What is the KI band gap, and what does LDA make of it?

    ZnO's gap is direct, at :math:`\Gamma`. Find the last block of interpolated
    eigenvalues in ``zno/03-dfpt/03-ham/outputs/aiida.kho``:

    .. code-block:: text

        KC interpolated eigenvalues at k=      0.0000      0.0000      0.0000

        -122.5630  -122.4517   -75.4592   -75.4487   -75.3613   -75.3510   -75.3339   -75.3293
         -11.8377   -11.0733    -0.4195    -0.4113    -0.3590     0.0329     0.0525     0.2164
           0.2214     0.4945     1.0262     1.1818     2.1655     6.3615     6.4113     7.0942
           7.1300     7.2304    10.7486    14.9226

    The twenty-sixth of these is the valence band edge and the twenty-seventh the
    conduction band edge, so the KI gap is 10.75 − 7.23 = 3.5 eV. For the LDA gap, look
    in the ground-state output ``zno/01-scf_nscf/01-scf/outputs/aiida.out``:

    .. code-block:: text

         highest occupied, lowest unoccupied level (ev):     9.2875    9.9769

    which is 0.7 eV.

LDA understates ZnO's gap by a factor of five; KI puts it within a few tenths of an
electronvolt of the measured 3.4 eV. A converged calculation gives 3.6 eV
:cite:`Colonna2022`; the difference is the coarse cutoff and mesh this tutorial uses to
stay quick.

.. warning::

    The two empty Wannier functions are the delicate part of this calculation. They are
    disentangled from twenty-six bands rather than taken from a closed group, and the
    localization problem they solve has more than one solution: runs that start from
    slightly different guesses can settle on different Wannier functions and move the
    conduction bands by a few tenths of an electronvolt. The filled manifold has no such
    freedom and is reproducible. Treat the conduction bands of a disentangled manifold —
    and the gap that depends on them — as the quantity to check convergence of most
    carefully.

.. note::

    The natural thing to do with the interpolated eigenvalues is to plot them against
    the LDA bands along the same path. ``koopmans`` does not draw band structures for
    you yet — neither this comparison nor the plain LDA one — so for now the eigenvalues
    have to be taken from the outputs and plotted by hand. The LDA bands along the path
    need a ``dft_bands`` run, which the next section uses anyway.

.. _zno-projections:

**********************************
 Choosing the projections yourself
**********************************

The projections above were handed to you. Here is how to arrive at them.

Start from the band structure of the underlying LDA calculation. Take the input file,
change ``task: singlepoint`` to ``task: dft_bands``, and run it again: that runs the
ground state and then the bands along the path, and nothing else. What it shows is that
the filled bands of ZnO come in four groups, each separated from the next by a wide gap.
That is where four of the five blocks come from — one per group of bands. The fifth is
the pair of bands immediately above the gap, which are not separated from the empty
bands above them in the same clean way — which is what the energy windows below are for.

Which atomic orbitals each group is made of is the other half of the answer, and here it
is chemistry: two Zn 3s and six Zn 3p semicore bands, two O 2s, then sixteen bands of Zn
3d hybridized with O 2p, and Zn 4s for the two empty ones. Counting the bands in each
group of the plot against the electrons each shell holds will confirm the assignment.

.. note::

    The general tool for this is a projected density of states, which says outright how
    much of each band comes from which atomic orbital. ``koopmans`` does not yet produce
    one.

.. question:: How many Wannier functions does a block get?

    As many as its projections provide, counting the angular momentum: ``l=0`` on Zn is
    one function per Zn atom and there are two of them, so the Zn 3s block holds two;
    ``l=1`` is three per atom, so Zn 3p holds six; the fourth block holds ten from Zn
    ``l=2`` and six from O ``l=1``. You never state the number — the projections fix it,
    and ``kcw.x`` is told twenty-six filled and two empty on that basis.

The empty block is the one that needs thought, because its two bands are not separated
from the rest of the empty states: they overlap them. ``Wannier90`` handles this by
disentangling, and two energy windows steer it.

``dis_win_max``
    the top of the disentanglement window — the range of bands the two Wannier functions
    may be built from. It must comfortably contain both Zn 4s bands, which means it will
    inevitably admit some weight from higher bands too.

``dis_froz_max``
    the top of the frozen window — the range of bands that must be reproduced exactly.
    Make it as large as you can while excluding anything that is not Zn 4s.

Read both off the LDA band structure, then check them against one thing before you use
them.

.. warning::

    These two windows are absolute energies, in electronvolts, on the same scale as the
    Kohn-Sham eigenvalues — not energies relative to the valence band edge. Band
    structure plots almost always put the valence band edge at zero, so a window read off
    a plot must be shifted by the edge's own energy before it goes in the input file.
    Here that edge is at 9.29 eV, which you can find with

    .. code-block:: console

        $ grep 'highest occupied' zno/01-scf_nscf/01-scf/outputs/aiida.out

    so the 14.5 and 17.0 in the input file sit 5.2 and 7.7 eV above the valence band
    edge.

.. note::

    Disentanglement keywords apply to the last block of projections only. The other four
    blocks take exactly as many bands as they have projections, so there is nothing to
    disentangle.

To try a choice of windows without running the whole workflow, set ``task: wannierize``
and run that: it stops once the Wannier functions exist. Two of the ways a window can be
wrong are caught before ``Wannier90`` is even called — a frozen window that would freeze
more bands than the block has Wannier functions is rejected outright, naming the block
and the energy to go below, and a block with more bands than Wannier functions and no
window at all draws a warning that its Wannierization is unconstrained. What is left to
your judgement is how localized the result is, which each block's ``.wout`` file reports
as the final spread of every Wannier function.

.. question:: The two empty Wannier functions come out with large spreads. What went wrong?

    Most likely the frozen window. If ``dis_froz_max`` is too low the two Wannier
    functions are not required to reproduce the Zn 4s bands anywhere, and the
    disentanglement is free to trade their accuracy for localization; if it is too high
    it requires them to reproduce bands that are not Zn 4s at all, which two functions
    cannot do. Move it, rerun, and watch the spreads.

Finally, the code can find the blocks itself. Setting
``block_wannierization_threshold`` to an energy in electronvolts — available on the
``wannierize`` task — has it run the bands, split the manifold wherever a gap wider than
that threshold appears, and Wannierize each group separately, rather than relying on
your projections to have grouped the bands correctly. It is the tool for a system whose
block structure you do not know in advance, or one that changes as you sweep a
parameter. It needs one more code registered, ``wannierjl``, which performs the split.
