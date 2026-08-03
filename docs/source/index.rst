##########
 koopmans
##########

.. container:: landing-logo

    .. image:: _static/logo.png
        :width: 400
        :alt: koopmans

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

.. container:: capability-grid

    .. container:: capability-cell

        .. image:: _static/capabilities/molecular_orbital_energies.png
            :alt: The density of states of an ozone molecule

        Orbital energies for :doc:`paramagnetic <tutorials/ozone>` and :doc:`magnetic
        <tutorials/magnetic_molecules>` molecules

    .. container:: capability-cell

        .. image:: _static/capabilities/silicon_bands.png
            :alt: The Koopmans band structure of bulk silicon

        Band structures of solids, screened in a :doc:`supercell
        <tutorials/silicon_supercell>` or by :doc:`linear response
        <tutorials/silicon_dfpt>`

    .. container:: capability-cell

        .. image:: _static/capabilities/zno_bands.png
            :alt: The LDA and Koopmans band structures of ZnO, with the band gap marked

        Valence manifolds that need :doc:`several Wannier blocks <tutorials/zno>`, found
        for you

    .. container:: capability-cell

        .. image:: _static/capabilities/magnetic_bands.png
            :alt: Spin-up and spin-down band structures of a ferromagnet

        A band structure per spin channel, for :doc:`magnetic solids
        <tutorials/magnetic_solids>`

    .. container:: capability-cell

        .. image:: _static/capabilities/ml_screening.png
            :alt: Predicted against calculated orbital energies, and their error
                distribution

        Screening parameters :doc:`predicted by a trained model
        <tutorials/machine_learning>`, not calculated

    .. container:: capability-cell

        .. image:: _static/capabilities/spin_orbit.svg
            :alt: Placeholder tile for spin-orbit coupled band structures

        Band structures with :doc:`spin-orbit coupling <tutorials/spin_orbit>`

    .. container:: capability-cell

        .. image:: _static/capabilities/dielectric.svg
            :alt: Placeholder tile for dielectric constants

        :doc:`Dielectric constants <tutorials/dielectric>`, which a Koopmans calculation
        on a solid needs

    .. container:: capability-cell planned

        .. image:: _static/capabilities/optical_spectra.svg
            :alt: Placeholder tile for optical spectra, planned

        Optical spectra, from the Bethe-Salpeter equation — *planned*

    .. container:: capability-cell planned

        .. image:: _static/capabilities/realtime.svg
            :alt: Placeholder tile for real-time spectroscopy, planned

        Real-time spectroscopy — *planned*

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
