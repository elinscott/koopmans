#########################################################
 The ionization potential and electron affinity of ozone
#########################################################

This tutorial calculates the ionization potential and electron affinity of the ozone
molecule with the KI functional. Both are quantities a photoemission experiment measures
directly — the energy to remove an electron from the highest occupied molecular orbital
(HOMO), and the energy gained by adding one to the lowest unoccupied molecular orbital
(LUMO) — and both are quantities that ordinary density-functional theory gets badly
wrong. A Koopmans calculation reads them off the orbital energies of the corrected
functional, with screening parameters computed from first principles rather than
guessed.

It comes in two parts, which calculate the same thing twice over. The first runs the
underlying Quantum ESPRESSO calculations by hand, one command at a time, and computes a
single screening parameter with a pen-and-paper formula. The second hands the whole
procedure to ``koopmans``, which computes a screening parameter for every orbital
without further instruction. Read them in order and the second part is a bookkeeping
exercise you have already done yourself.

.. toctree::
    :maxdepth: 1

    by_hand
    automated
