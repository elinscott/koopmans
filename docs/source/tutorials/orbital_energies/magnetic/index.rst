##########################
 Spin-polarized molecules
##########################

A molecule with an open shell: the oxygen molecule, whose ground state is a spin
triplet. Treating the two spin channels separately changes what a Koopmans calculation
is working with — every orbital now carries a spin channel as well as an index, and it
gets its own screening parameter. This page runs the :doc:`ozone calculation
<../ozone/automatically>` again on O₂ and draws out that difference; anything it does
not mention works as it did there.

****************
 The input file
****************

Download :download:`o2.yaml <o2.yaml>` and place it in an empty directory. Here it is in
full:

.. literalinclude:: o2.yaml
    :language: yaml

.. warning::

    As in the ozone tutorial, the cell and the cutoff here are rough enough to finish in
    minutes, and are not converged.

Three settings make this an open-shell calculation.

.. literalinclude:: o2.yaml
    :language: yaml
    :start-at: spin
    :end-at: spin

gives the two channels their own orbitals, their own eigenvalues and their own screening
parameters.

.. literalinclude:: o2.yaml
    :language: yaml
    :start-at: tot_magnetization
    :end-at: tot_magnetization

fixes how the electrons divide between them. O₂ has 12 valence electrons, so a
magnetization of 2 puts 7 in the majority channel and 5 in the minority.

.. literalinclude:: o2.yaml
    :language: yaml
    :start-at: nbnd
    :end-at: nbnd

counts bands *per channel*, not in total: 7 filled and 1 empty in the majority channel,
5 filled and 3 empty in the minority. That is 16 orbitals to screen, against ozone's 10.

Two further settings are not about spin, but change what you will see. ``alpha_numsteps:
2`` runs the screening loop a second time, starting from the parameters the first pass
computed — the optional exercise at the end of the ozone tutorial. ``group_orbitals_by:
self_hartree`` lets orbitals whose self-Hartree energies agree to within
``group_orbitals_tol`` share a screening parameter: one member of each group is screened
and the rest inherit the result. Grouping never crosses a spin channel, and never puts a
filled orbital with an empty one.

*************************
 What changes in the run
*************************

Run it as before:

.. code-block:: console

    $ koopmans run o2.yaml

The progress table differs from ozone's in two places.

The initialization is a single step, ``DFT Init``, rather than the three-step detour
through a spin-unpolarized calculation. That detour exists to hand the spin-resolved
calculation an already-symmetric density; here the two channels are *meant* to differ,
so there is nothing to symmetrize and the workflow starts spin-resolved from scratch.

The per-orbital screening steps now name a channel as well as an index — ``Compute Alpha
Up Orb 1`` through ``Compute Alpha Down Orb 8`` — and there is one per group rather than
one per orbital. All of it happens twice, under ``Iteration 1`` and ``Iteration 2``.

*************
 The results
*************

The screening parameters the final calculation used are in its ``inputs``, in
``file_alpharef.txt``: the majority channel's list, then the minority channel's. For
closed-shell ozone the workflow computes one list and writes it into both blocks. Here
the two are computed independently.

The final KI output reports ``Eigenvalues`` for ``spin = 1`` and ``spin = 2``
separately, and a single ``HOMO Eigenvalue`` and ``LUMO Eigenvalue`` across both. The
majority channel fills both :math:`\pi^*` orbitals and the minority leaves them empty,
so the highest occupied and lowest unoccupied orbitals are the same orbital in opposite
channels, each screened by a parameter of its own.

KI puts the ionization potential at 12.35 eV and the electron affinity at 0.40 eV. As in
the ozone tutorial, an orbital energy is a vertical quantity, and photoemission puts O₂'s
vertical first ionization between 12.30 eV :cite:`Kimura1981` and 12.33 eV
:cite:`Banna1976` — a few hundredths of an electronvolt below KI.

The `tabulated values
<https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=20#Ion-Energetics>`_ of 12.07 eV
and 0.45 eV are the adiabatic ones. The electron affinity has no measured vertical
counterpart: photodetachment reaches the neutral molecule from O₂⁻, whose bond is longer
than O₂'s, so there is nothing to hold the 0.40 eV against.
