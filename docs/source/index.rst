##########
 koopmans
##########

**Spectral properties of molecules and materials, from Koopmans functionals.**

Koopmans functionals give you orbital energies that can be compared against a
photoemission spectrum — ionization potentials, electron affinities and band structures
— at a cost close to that of density-functional theory. Obtaining them takes a chain of
calculations: a ground state, a set of localized variational orbitals, a screening
parameter for each of them, and a final band structure. ``koopmans`` takes an input file
describing your system and runs that chain for you, using `Quantum ESPRESSO
<https://www.quantum-espresso.org/>`_.

.. code-block:: console

    $ koopmans run silicon.json

New here? :doc:`Install the code <installation>`, then run :doc:`your first calculation
<tutorials/water>`.

************************
 What you can calculate
************************

.. list-table::
    :header-rows: 1
    :widths: 30 35 35
    :class: capability-matrix

    - -
      - Molecules
      - Periodic solids
    - - Orbital energies and band structures
      - :doc:`Ozone <tutorials/ozone>` — ionization potential and electron affinity
      - :doc:`Silicon from a supercell <tutorials/silicon_supercell>`, :doc:`silicon
        from linear response <tutorials/silicon_dfpt>`, :doc:`ZnO <tutorials/zno>`
    - - Spin-orbit coupled band structures
      - —
      - :doc:`Silicon with spin-orbit coupling <tutorials/spin_orbit>`
    - - Dielectric constants
      - —
      - :doc:`The dielectric constant of a semiconductor <tutorials/dielectric>`
    - - Optical spectra
      - *planned*
      - *planned*
    - - Real-time spectroscopy
      - *planned*
      - *planned*

Optical spectra, from the Bethe-Salpeter equation on top of a Koopmans ground state, and
real-time spectroscopy are both under development.

**************************
 How the calculation runs
**************************

The screening parameters are the expensive part of a Koopmans calculation, and there is
more than one way to obtain them. Which route you take is independent of whether your
system is a molecule or a solid, and of how the variational orbitals are chosen.

.. list-table::
    :header-rows: 1
    :widths: 45 55
    :class: capability-matrix

    - -
      - Where it is shown
    - - Screening from total-energy differences (ΔSCF)
      - :doc:`Ozone <tutorials/ozone>`, :doc:`silicon <tutorials/silicon_supercell>`
    - - Screening from linear response (DFPT)
      - :doc:`Silicon <tutorials/silicon_dfpt>`, :doc:`ZnO <tutorials/zno>`
    - - Screening predicted by a machine-learned model
      - :doc:`Water snapshots <tutorials/machine_learning>`
    - - Variational orbitals split into blocks automatically
      - :doc:`ZnO <tutorials/zno>`
    - - Convergence testing
      - *planned*

.. toctree::
    :maxdepth: 2
    :caption: Getting Started
    :name: start
    :hidden:

    installation
    tutorials/index

.. toctree::
    :maxdepth: 2
    :caption: Reference
    :name: reference
    :hidden:

    theory
    input_file
    cli

**********
 Citation
**********

If you use this code, please cite E. B. Linscott, N. Colonna, R. De Gennaro, N. L.
Nguyen, G. Borghi, A. Ferretti, I. Dabo and N. Marzari, *koopmans: An Open-Source
Package for Accurately and Efficiently Predicting Spectral Properties with Koopmans
Functionals*, J. Chem. Theory Comput. **19**, 7097 (2023), `doi:10.1021/acs.jctc.3c00652
<https://doi.org/10.1021/acs.jctc.3c00652>`_.

********************
 Indices and Tables
********************

- :ref:`genindex`
- :ref:`modindex`
- :ref:`search`
