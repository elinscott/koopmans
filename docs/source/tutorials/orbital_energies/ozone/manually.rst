###########################
 Doing everything manually
###########################

A Koopmans calculation is a set of Quantum ESPRESSO calculations with quite a bit of
bookkeeping to link them all together. In this tutorial you will go through that process
manually, to get a sense of what is going on under the hood of ``koopmans``.

***************
 What you need
***************

Make sure the Quantum ESPRESSO executable ``kcp.x`` is on your ``PATH`` — see the
:doc:`installation page <../../../installation>` for more details.

Download :download:`ozone_manually.zip` and unpack it into an empty directory. It holds
the three ``kcp.x`` input files you will run

- ``ozone_dft.in``, the neutral :math:`N`-electron DFT calculation
- ``ozone_dft_n-1.in``, the constrained :math:`N-1`-electron DFT calculation
- ``ozone_ki.in``, the trial KI calculation

along with ``get_alpha.sh``, a script that applies the screening formula so you can
check your own arithmetic, and the oxygen pseudopotential the calculations read from
``pseudopotentials/``.

.. note::

    All of the calculations share one ``prefix`` (``kc``) and one ``outdir`` (``TMP/``),
    and hand wavefunctions to each other through restart files numbered by the ``ndr``
    (read) and ``ndw`` (write) units. Run them in the order below; each one starts from
    what the previous one wrote.

***********************************
 The neutral :math:`N`-electron run
***********************************

Run the DFT calculation for the neutral molecule:

.. code-block:: console

    $ mpirun kcp.x -in ozone_dft.in | tee ozone_dft.out

The molecule has 18 valence electrons (see ``nelec = 18`` in the input file), split
evenly between the two spin channels, and ``nbnd = 10`` asks for one empty orbital on
top of the nine filled ones.

.. question:: What does ``do_orbdep = .false.`` mean, and why start with such a calculation?

    It switches off the orbital-dependent part of the functional, leaving plain
    semi-local DFT — here PBE.

    That is where a Koopmans calculation starts: the density and the variational
    orbitals come from the base functional, and converging them in the cheaper DFT
    problem before switching the correction on saves work. For KI it costs nothing at
    all, since KI leaves the density and the occupied variational orbitals of its base
    functional unchanged — they are already the optimal ones.

Molecules use the Kohn-Sham orbitals directly as variational orbitals, so copy them into
the files the KI calculation will read as its starting variational orbitals:

.. code-block:: console

    $ cp TMP/kc_90.save/K00001/evc1.dat       TMP/kc_90.save/K00001/evc01.dat
    $ cp TMP/kc_90.save/K00001/evc2.dat       TMP/kc_90.save/K00001/evc02.dat
    $ cp TMP/kc_90.save/K00001/evc_empty1.dat TMP/kc_90.save/K00001/evc0_empty1.dat
    $ cp TMP/kc_90.save/K00001/evc_empty2.dat TMP/kc_90.save/K00001/evc0_empty2.dat

.. important::

    Do not skip this step. ``kcp.x`` reads its variational orbitals from the ``evc0``
    files and its Kohn-Sham orbitals from the ``evc`` ones; without the copy, the KI
    calculation starts from whatever the ``evc0`` files happen to contain, and it will
    run to completion and give you a wrong screening parameter rather than complain.

*************************************
 The constrained :math:`N-1` run
*************************************

Now the same molecule, with one electron removed from a particular orbital:

.. code-block:: console

    $ mpirun kcp.x -in ozone_dft_n-1.in | tee ozone_dft_n-1.out

This input differs from the first in a handful of keywords. It restarts from what the
neutral run wrote (``restart_mode = 'restart'``, ``ndr = 90``) instead of starting from
scratch, and its ``&SYSTEM`` block gains three lines:

.. code-block:: fortran

    fixed_state  = .true.
    fixed_band   = 9
    f_cutoff     = 1e-05

``f_cutoff`` is the occupation imposed on the variational orbital numbered
``fixed_band`` — here 10\ :sup:`-5`, which is zero for all practical purposes.

.. question:: What is this calculation doing?

    Ozone's ninth variational orbital is its HOMO, and ``f_cutoff`` sets that orbital's
    occupation to zero. This is an :math:`N-1`-electron calculation in which the hole is
    constrained to sit in the HOMO of the :math:`N`-electron solution, with the rest of
    the density free to relax around it.

The difference between the two total energies is a ΔSCF estimate of the ionization
potential. Both are printed in the output as ``total energy = ...``, in Hartree.

.. question:: What ionization potential do the two runs give?

    .. math::

        E^\text{DFT}[N] - E^\text{DFT}[N-1] &= (-47.5296) - (-47.0705)\ \text{Ha} \\
        &= -0.4591\ \text{Ha} \\
        &\approx -12.49\ \text{eV},

    so the ΔSCF ionization potential is 12.49 eV. Experiment puts it at `about 12.5 eV
    <https://webbook.nist.gov/cgi/cbook.cgi?ID=C10028156&Mask=20#Ion-Energetics>`_.

    Total-energy differences are a good estimate of the ionization potential even in
    plain DFT. Orbital energies are not: the PBE HOMO of this same calculation sits at
    −7.92 eV, more than 4 eV adrift. Fixing that gap is the entire point of a Koopmans
    functional.

***************************
 The trial KI calculation
***************************

The KI correction to an orbital energy is proportional to a screening parameter
:math:`\alpha`, and we do not yet know the right value. One calculation at a guessed
value is enough to pin it down.

Open ``ozone_ki.in`` and replace the ``<alpha>`` placeholder with ``0.7``:

.. code-block:: fortran

    &NKSIC
       nkscalfact         = 0.7
       which_orbdep       = 'nki'
       do_innerloop       = .false.
       esic_conv_thr      = 1.8000000000000002e-08
       do_innerloop_empty = .false.
    /

then run it:

.. code-block:: console

    $ mpirun kcp.x -in ozone_ki.in | tee ozone_ki.out

.. question:: This run has ``do_orbdep = .true.`` and ``which_orbdep = 'nki'``. What do those switch on?

    Together they turn on the orbital-density-dependent correction: ``do_orbdep``
    enables the orbital-dependent term in the functional, and ``which_orbdep`` picks the
    KI form of it.

.. question:: How does the KI HOMO at :math:`\alpha_0 = 0.7` compare with the PBE one?

    It has moved down, from −7.92 eV to −12.00 eV. The Koopmans correction counteracts
    the self-interaction error that leaves the PBE HOMO too shallow, so the predicted
    ionization potential goes up.

***************************
 The screening parameter
***************************

The optimal :math:`\alpha` is the one that makes the HOMO energy agree with the
total-energy difference you already computed — the Koopmans condition
:cite:`Borghi2014,Nguyen2018`,

.. math::

    \varepsilon^\text{KI}_\text{HOMO}(\alpha_\text{opt}) = E^\text{DFT}[N] -
    E^\text{DFT}[N-1].

You have the HOMO energy at two values of :math:`\alpha`: at :math:`\alpha_0 = 0.7` from
the trial run, and at :math:`\alpha = 0` from the DFT run, since the correction vanishes
there.

.. question:: Derive :math:`\alpha_\text{opt}` from those two points and the Koopmans condition.

    KI changes neither the ground-state density nor the occupied variational orbitals,
    so the only place :math:`\alpha` enters the corrected eigenvalue is as an explicit
    prefactor:

    .. math::

        \varepsilon^\text{KI}_\text{HOMO}(\alpha) = \varepsilon^\text{DFT}_\text{HOMO} +
        \alpha \lambda_\text{HOMO},

    with :math:`\lambda_\text{HOMO}` independent of :math:`\alpha`. The eigenvalue is
    therefore linear in :math:`\alpha`, and two points fix the line:

    .. math::

        \lambda_\text{HOMO} = \frac{\varepsilon^\text{KI}_\text{HOMO}(\alpha_0) -
        \varepsilon^\text{DFT}_\text{HOMO}}{\alpha_0}.

    Imposing the Koopmans condition and solving for :math:`\alpha_\text{opt}`,

    .. math::

        \alpha_\text{opt} = \alpha_0 \frac{\big(E^\text{DFT}[N] - E^\text{DFT}[N-1]\big)
        - \varepsilon^\text{DFT}_\text{HOMO}} {\varepsilon^\text{KI}_\text{HOMO}(\alpha_0)
        - \varepsilon^\text{DFT}_\text{HOMO}}.

.. question:: Put the numbers in. What is :math:`\alpha_\text{opt}` for ozone's HOMO?

    .. math::

        \alpha_\text{opt} &= 0.7 \times \frac{(-12.49) - (-7.92)}{(-12.00) - (-7.92)} \\
        &= 0.7 \times \frac{-4.57}{-4.08} \\
        &\approx 0.78.

Check your arithmetic against the script, which reads the two total energies and the two
HOMO eigenvalues out of the three output files, and the trial :math:`\alpha_0` out of
``ozone_ki.in``:

.. code-block:: console

    $ sh get_alpha.sh

***************************
 The final KI calculation
***************************

Copy ``ozone_ki.in`` to ``ozone_ki_opt.in`` and make two changes. Send the output to a
fresh restart unit, so that this run does not overwrite the trial's (``ndr`` stays at
90: like the trial, this calculation starts from the DFT orbitals, not from the trial
KI's):

.. code-block:: fortran

    ndw = 92

and put your screening parameter in place of the trial value:

.. code-block:: fortran

    nkscalfact = 0.78

Then run it:

.. code-block:: console

    $ mpirun kcp.x -in ozone_ki_opt.in | tee ozone_ki_opt.out

.. question:: Does the HOMO of this final run satisfy the Koopmans condition?

    It does, to within the precision you carried through the arithmetic:
    :math:`-\varepsilon^\text{KI}_\text{HOMO}` now agrees with the 12.49 eV ΔSCF
    ionization potential, where the PBE eigenvalue gave 7.92 eV. The eigenvalue has been
    made to mean what the total-energy difference says it should mean — and that is all
    a Koopmans functional does.

**********************************
 What the package does for you
**********************************

You have computed one screening parameter, for one orbital, and applied it to every
orbital in the molecule. A real calculation gives each variational orbital its own,
which means one constrained calculation per orbital rather than one in total.

.. question:: How many calculations would ozone's ten orbitals need?

    One DFT initialization, one trial KI, one constrained calculation for each of the
    ten orbitals, and one final KI: twelve in all. Iterating the screening parameters to
    self-consistency multiplies the middle part — twenty-two calculations for two
    iterations, and so on.

    In practice self-consistency is rarely needed for KI, and orbitals related by
    symmetry can share a screening parameter, which brings the count back down.

All of those calculations, each restarting from the right predecessor, with occupations
constrained orbital by orbital and the screening formula applied to every result: that
is the bookkeeping the ``koopmans`` package takes care of. The :doc:`next part
<automatically>` runs this same calculation, in one command.
