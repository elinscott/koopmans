##########
 koopmans
##########

.. container:: landing-logo

    .. image:: _static/logo.png
        :width: 400
        :alt: koopmans

.. container:: landing-subtitle

    **Spectral properties of molecules and materials, from Koopmans functionals.**

Koopmans functionals give you orbital energies that can be compared against a
photoemission spectrum — ionization potentials, electron affinities and band structures —
much more cheaply than GW or a hybrid, and just as accurately. These calculations require
a chain of calculations (SCF, Wannierization, linear response, *etc.*), but ``koopmans``
takes care of this: all it needs is an input file describing your system and it will run
everything for you.

.. code-block:: console

    $ koopmans run silicon.json

New here? :doc:`Install the code <installation>`, then run :doc:`your first calculation
<tutorials/water>`.

************************
 What you can calculate
************************

.. container:: capability-grid

    .. container:: capability-cell

        .. image:: _static/capabilities/molecular_orbital_energies.svg
            :alt: Placeholder tile for the orbital energies of molecules

        Orbital energies for :doc:`paramagnetic <tutorials/ozone>` and :doc:`magnetic
        <tutorials/magnetic_molecules>` molecules

    .. container:: capability-cell

        .. image:: _static/capabilities/zno_bands.png
            :alt: The LDA and Koopmans band structures of ZnO, with the band gap marked

        Band structures of solids, with screening parameters from :doc:`finite differences
        <tutorials/silicon_supercell>` or :doc:`linear response <tutorials/silicon_dfpt>` —
        and with :doc:`automated Wannierization <tutorials/zno>`, for :doc:`magnetic systems
        <tutorials/magnetic_solids>`, or with :doc:`spin-orbit coupling
        <tutorials/spin_orbit>`

    .. container:: capability-cell

        .. image:: _static/capabilities/ml_screening.png
            :alt: Predicted against calculated orbital energies, and their error
                distribution

        Screening parameters via :doc:`machine learning <tutorials/machine_learning>`

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
