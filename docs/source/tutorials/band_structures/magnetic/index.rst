######################################################
 CrI\ :sub:`3`, and a band structure per spin channel
######################################################

In its low-temperature phase, bulk CrI\ :sub:`3` is a ferromagnetic semiconductor: each
of the two Cr atoms in the primitive cell carries three unpaired *d* electrons, all
aligned. This tutorial computes its KI band structure.

Most of what happens here happens in any Koopmans calculation on a crystal — Wannier
functions stand in for the molecular orbitals of a molecular calculation, and the
screening parameters come from linear response. What this page is about is the
ingredient a magnet adds. The two spin channels no longer hold the same electrons, so
they are Wannierized separately, screened separately, and come out as two band
structures rather than one.

****************
 The input file
****************

Download :download:`cri3.yaml <cri3.yaml>` and place it in an empty directory. Here it
is in full:

.. literalinclude:: cri3.yaml
    :language: yaml

.. warning::

    The cutoff and the k-point grid in this file are deliberately rough, so that the
    workflow finishes in a reasonable time. They are not converged, and neither is the
    band structure that comes out of them. Use this file to learn the input format, not
    as a starting point for production work.

---------------
 Spin channels
---------------

.. literalinclude:: cri3.yaml
    :language: yaml
    :start-at: spin: collinear
    :end-at: spin: collinear

is the keyword that makes this a magnetic calculation. ``spin`` takes four values:

``none``
    One density, one set of orbitals, every orbital doubly occupied. This is the
    default, and it is what every non-magnetic tutorial uses.

``collinear``
    Two densities, :math:`n^\uparrow(\mathbf{r})` and :math:`n^\downarrow(\mathbf{r})`,
    free to differ. Every spin points along one axis, up or down, and every orbital
    belongs to one channel or the other.

``non_collinear``
    Spinor wavefunctions, so the magnetization can point in different directions in
    different parts of the cell.

``spin_orbit``
    Spinors again, with spin-orbit coupling switched on.

A ferromagnet is the case ``collinear`` was made for. Every moment is parallel to every
other, so a single axis is enough — but the two channels are genuinely different, here
holding 38 electrons and 32.

For a Koopmans calculation the consequence runs deeper than a second density. Each
variational orbital now carries a spin channel as well as an index, each gets its own
screening parameter, and the correction is applied channel by channel.

.. note::

    Ferromagnets only, for the moment. An antiferromagnet needs its two magnetic
    sublattices started antiparallel, and a structure whose sublattices are inequivalent
    cannot yet be written in an input file — `issue #74
    <https://github.com/elinscott/koopmans/issues/74>`_ tracks that gap.

.. warning::

    Not every combination of ``spin`` and ``screening_method`` exists. ``kcp.x``, which
    is what ``screening_method: dscf`` runs, has no spinor mode, so that route supports
    ``none`` and ``collinear`` only. ``kcw.x``, used here, supports all four; the
    :doc:`spin-orbit tutorial <../spin_orbit/index>` exercises the other two.

.. literalinclude:: cri3.yaml
    :language: yaml
    :start-at: tot_magnetization
    :end-at: tot_magnetization

fixes how the electrons divide between the channels. The pseudopotentials put 70 valence
electrons in the cell, so :math:`N^\uparrow - N^\downarrow = 6` makes it 38 up and 32
down.

.. question:: Where does 6 come from?

    Each Cr atom in CrI\ :sub:`3` is a Cr\ :sup:`3+` ion — a *d*\ :sup:`3` configuration,
    whose three *d* electrons are unpaired. Two Cr atoms make six unpaired electrons, and
    in the ferromagnetic state all six point the same way.

    ``tot_magnetization`` constrains the calculation rather than merely starting it off,
    so it has to be the right answer rather than a guess. It is also a *net* moment, and
    a net moment is not the moment on any one atom: at this level of theory the moment
    integrated around each Cr is 2.66 :math:`\mu_B`, and around each of the six I atoms
    it is −0.03 :math:`\mu_B` — the iodines are polarized slightly the other way.

.. note::

    Constraining the net moment does not stop ``Quantum ESPRESSO`` taking a long route
    to it. The keyword ``starting_magnetization`` (documented `here
    <https://www.quantum-espresso.org/Doc/INPUT_PW.html#idm301>`_) puts the moment on
    the right atoms from the first iteration, which can save several cycles. As an
    optional exercise, add it under ``calculator_parameters.pw.system``, choosing values
    from the description above — the moment sits on the Cr atoms, not the I ones.

-------------------------
 Two sets of projections
-------------------------

Wannier functions are built from projections onto trial orbitals, and a block of
projections names a group of bands to be Wannierized together. A magnetic calculation
needs one list per channel, because the channels do not contain the same bands:

.. literalinclude:: cri3.yaml
    :language: yaml
    :start-at: wannier90:

The first four blocks of each list span the occupied manifold and the rest span the
low-lying empty states. Both lists open the same way — Cr *s*, Cr *p*, I *s* — and then
diverge.

.. question:: Why does the fourth block differ between the two channels?

    Because that is where the unpaired electrons are. In the up channel the fourth block
    is I *p* together with three of the five Cr *d* orbitals, which are occupied; the
    other two are empty and form a block of their own. In the down channel the fourth
    block is I *p* alone, and all five *d* orbitals are empty, split across two blocks.

    Counting the Wannier functions checks the arithmetic. The up list has :math:`2 + 6 +
    6 + (18 + 6) = 38` occupied functions and the down list :math:`2 + 6 + 6 + 18 = 32`,
    which is exactly the 38/32 split that ``tot_magnetization: 6`` asks for. Get this
    wrong and the calculation stops before it starts.

.. question:: Why is there no disentanglement window here, when the other band-structure tutorials have one?

    Because each block here is an isolated group of bands, separated from its neighbours
    by a gap right across the Brillouin zone. Disentanglement — ``dis_win_max``,
    ``dis_froz_max`` — is what you reach for when the bands you want are tangled with
    bands you do not, as the empty manifold of a semiconductor usually is. An isolated
    group needs none of it: the Wannier functions span the block exactly, and the
    eigenvalues of the Wannier Hamiltonian reproduce the ones ``pw.x`` computed.

.. question:: How would I choose these blocks for a material of my own?

    From the projected density of states, splitting the manifold wherever the character
    of the bands changes. That is the procedure the :doc:`ZnO tutorial <../zno/index>`
    goes through, and it also shows how to have the code find the blocks for you instead.

.. literalinclude:: cri3.yaml
    :language: yaml
    :start-at: group_orbitals_by
    :end-at: group_orbitals_tol

lets Wannier functions whose spreads agree to within 0.1 Å\ :sup:`2` share a screening
parameter. Within a block the *d* functions are equivalent by symmetry, so computing
:math:`\alpha` once and copying it to the rest costs nothing in accuracy and saves a
linear-response calculation apiece. Grouping happens inside a channel: an up orbital
never shares a parameter with a down one.

----------------------
 The rest of the file
----------------------

The remaining settings are the ones any crystal needs. ``atoms`` gives the rhombohedral
cell and the eight atoms in it, and ``kpoints`` gives both the mesh the Wannier
functions are built on and the path

.. literalinclude:: cri3.yaml
    :language: yaml
    :start-at: path:
    :end-at: path:

the final band structure is drawn along. Γ, L and F are labelled points of the
rhombohedral Brillouin zone, and ``koopmans`` resolves them against the lattice the cell
describes.

.. literalinclude:: cri3.yaml
    :language: yaml
    :start-at: screening_method
    :end-at: screening_method

computes the screening parameters from linear response, in the primitive cell. The
alternative is to compute them from total-energy differences in a supercell, which is
what the :doc:`silicon tutorials <../silicon_finite_differences/index>` compare the two
routes on. Linear response is the practical choice here: a supercell of CrI\ :sub:`3` is
a large object, and ``kcw.x`` interpolates the band structure at the end of the same
run.

.. note::

    One setting this file leaves out is ``eps_inf``, the macroscopic dielectric constant
    that the long-range part of the screening correction uses. For production work on a
    solid you should supply it — either as a number, or as ``'auto'``, which computes it
    with ``ph.x`` first. See the :doc:`dielectric constants tutorial
    <../../dielectric_constants/index>`.

*************************
 Running the calculation
*************************

.. warning::

    Make sure you have installed ``koopmans``: see :doc:`here <../../../installation>`
    for more details.

    This workflow needs ``pw.x``, ``pw2wannier90.x`` and ``kcw.x`` from ``Quantum
    ESPRESSO``, and ``wannier90.x`` alongside them.

Run the calculation with

.. code-block:: console

    $ koopmans run cri3.yaml

The terminal shows a live progress table as the workflow proceeds. It begins with a
shared pair of ``pw.x`` calculations — one SCF, one NSCF — which is the only stage the
two channels have in common, since a single spin-polarized calculation produces both
densities at once. Everything after it comes in twos: a Wannierization of each channel,
block by block, then a ``kcw.x`` chain per channel that converts the Wannier functions
into the form ``kcw.x`` reads, computes that channel's screening parameters, and finally
builds and diagonalizes the Koopmans Hamiltonian along the k-path.

*************
 The outputs
*************

The results land in a directory named after the input file — here ``cri3/`` — one
directory per step, numbered in the order the steps ran, laid out as the :doc:`ozone
tutorial <../../orbital_energies/ozone/automatically>` describes. What is new is that
there are two of nearly everything: a Wannierization and a ``kcw.x`` chain for the up
channel, and another pair for the down one.

Each channel's chain ends in a ``ham`` step, which holds the two results worth reading.
Its ``inputs/file_alpharef.txt`` lists the screening parameters that were used, one line
per variational orbital of that channel. Its ``outputs/aiida.kho`` is the ``kcw.x``
output, and near the bottom of it are two lines
reporting the highest occupied and lowest unoccupied level: one labelled ``KS``, which
is the underlying LDA result, and one labelled ``KI[2nd]``, which is the Koopmans one.
Above them, the interpolated eigenvalues along the k-path are printed point by point.

**************************
 Interpreting the results
**************************

Two band structures have to be read together. An electron leaving the crystal comes out
of whichever channel holds the highest occupied state, and one arriving goes into
whichever holds the lowest empty state, and those need not be the same channel.

.. question:: What does the underlying LDA calculation give?

    Gaps of about 1.14 eV in the up channel and 1.9 eV in the down one — but the
    fundamental gap is neither of those, because the valence band maximum lies in the up
    channel and the conduction band minimum in the down. It is about 1.13 eV, and it is
    an inter-spin gap.

KI opens both channels. How far it opens them is what the screening parameters decide,
and they are computed rather than assumed — which is the point of the calculation, so
the answer is best read off your own run rather than quoted here. For a sense of scale,
a reference calculation on this system with every screening parameter fixed at 0.122
instead of computed put the up-channel gap at 2.0 eV and the down-channel gap at 2.7 eV,
and moved the conduction band minimum into the up channel.

.. warning::

    Qualitatively this is the right picture, but the calculation is a long way from
    converged: the cutoff and the k-point grid both need raising before any of these
    numbers means anything quantitatively.
