########################
 The IP and EA of ozone
########################

This tutorial calculates the ionization potential (IP) and electron affinity (EA) of the
ozone molecule with the KI functional. Both are quantities a photoemission experiment
measures directly — the energy to remove an electron from the highest occupied molecular
orbital (HOMO), and the energy gained by adding one to the lowest unoccupied molecular
orbital (LUMO) — and both are quantities that ordinary density-functional theory gets
badly wrong. A Koopmans calculation reads them off the orbital energies of the corrected
functional, with screening parameters computed from first principles.

The tutorial comes in two parts. The first runs the calculation manually, one ``Quantum
ESPRESSO`` command at a time, and computes a single screening parameter with a
pen-and-paper formula. The second hands the whole procedure to ``koopmans``, which does
all of these calculations automatically.

.. toctree::
    :maxdepth: 1

    manually
    automatically
