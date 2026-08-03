##############
 Installation
##############

Installing ``koopmans`` takes three steps: Quantum ESPRESSO, the Python package, and
then one command that sets up the engine that runs your calculations.

******************
 Quantum ESPRESSO
******************

``koopmans`` does not do any of the electronic-structure work itself. It writes the
inputs, runs `Quantum ESPRESSO <https://www.quantum-espresso.org/>`_ and reads the
outputs back. You need a Quantum ESPRESSO installation whose executables are on your
``PATH`` before you set up the engine.

Which executables you need depends on what you want to calculate:

- ``pw.x`` is needed by everything.
- ``kcp.x`` computes screening parameters from total-energy differences, and evaluates
  the corrected functional. On a periodic system that route also uses ``wann2kcp.x`` and
  ``merge_evc.x``.
- ``kcw.x`` computes the screening parameters from linear response instead, and
  interpolates the final band structure.
- ``wannier90.x`` and ``pw2wannier90.x`` construct the localized orbitals that periodic
  systems use as variational orbitals.
- ``projwfc.x`` is optional, and gives you a projected density of states alongside a
  Wannierization.
- ``ph.x`` computes dielectric constants.

Choosing the blocks of a Wannierization automatically additionally needs `Wannier.jl
<https://github.com/qiaojunfeng/Wannier.jl>`_ and a Julia interpreter. Everything else
works without them.

********************
 The Python package
********************

``koopmans`` requires Python 3.12 or 3.13. Install the most recent code from GitHub with
uv:

.. code-block:: console

    $ uv pip install git+https://github.com/elinscott/koopmans.git

or with pip:

.. code-block:: console

    $ python3 -m pip install git+https://github.com/elinscott/koopmans.git

************
 The engine
************

Calculations are dispatched, run and recorded by a background engine, which keeps track
of what has already been computed so that an interrupted workflow can be resumed and a
repeated calculation is not run twice. Set it up once, with

.. code-block:: console

    $ koopmans install

This creates the database that records your calculations, installs and starts the job
scheduler, finds the Quantum ESPRESSO executables on your ``PATH`` and registers them,
and starts the background engine. It reports which executables it found; if one you need
is missing, put it on your ``PATH`` and run the command again, or point at it
explicitly:

.. code-block:: console

    $ koopmans install --code pw=/opt/qe/bin/pw.x

By default each calculation is given as many MPI processes as your machine has physical
cores. Use ``--procs-per-calc`` to change that, and ``--max-procs`` to cap how many
processes may run at once across all concurrent calculations.

To check on the engine at any point:

.. code-block:: console

    $ koopmans backend status

The engine runs in the background between calculations. If you reboot, or if it stops
for any other reason, start it again with

.. code-block:: console

    $ koopmans backend daemon start

and stop it with ``koopmans backend daemon stop``. ``koopmans backend uninstall``
removes the whole setup, database included.

****************************
 Installing for development
****************************

To install in development mode with uv:

.. code-block:: console

    $ git clone git+https://github.com/elinscott/koopmans.git
    $ cd koopmans
    $ uv pip install -e .

or with pip:

.. code-block:: console

    $ python3 -m pip install -e .
