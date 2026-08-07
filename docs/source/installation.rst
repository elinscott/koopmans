##############
 Installation
##############

Installing ``koopmans`` takes three steps: the electronic-structure codes, the Python
package, and then one command that sets up the engine that runs your calculations.

********************************
 The electronic-structure codes
********************************

``koopmans`` does not do any of the electronic-structure work itself. It writes the
inputs, runs `Quantum ESPRESSO <https://www.quantum-espresso.org/>`_ and `Wannier90
<https://wannier.org/>`_, and reads the outputs back. Their executables must be on your
``PATH`` before you set up the engine.

Which executables you need depends on the ``task`` you ask for, and — for a
``singlepoint`` — on how it screens. Every executable a task can reach must be there
before it starts, whatever the rest of the input file says — with the one exception
noted below:

- ``dft_bands`` needs ``pw.x``, which everything needs.
- ``dft_eps`` adds ``ph.x``, which computes dielectric constants.
- ``wannierize`` adds ``wannier90.x`` and ``pw2wannier90.x``, which construct the
  localized orbitals, and ``projwfc.x``, which gives a projected density of states
  alongside them.
- ``singlepoint`` with ``screening_method = 'dscf'`` adds ``kcp.x``, which computes
  screening parameters from total-energy differences and evaluates the corrected
  functional, together with the ``wannier90.x``, ``pw2wannier90.x``, ``wann2kcp.x`` and
  ``merge_evc.x`` that build its variational orbitals on a periodic system.
- ``singlepoint`` with ``screening_method = 'dfpt'`` adds ``kcw.x``, which computes the
  screening parameters from linear response instead and interpolates the final band
  structure, together with ``wannier90.x``, ``pw2wannier90.x`` and ``ph.x``.
- ``trajectory`` runs one ``dscf`` singlepoint per snapshot, so it needs what that
  needs.

One Wannierization setting reaches past that list. Setting
``workflow.block_wannierization_threshold`` splits the Wannierization into blocks with
`Wannier.jl <https://github.com/qiaojunfeng/Wannier.jl>`_, which needs a Julia
interpreter — the one thing the engine setup cannot find on your ``PATH``. Build its
project with ``aiida_wannierjl.helpers.setup_julia_environment`` and register it with
``aiida_wannierjl.helpers.get_wannierjl_code``. Leave the threshold unset and you need
neither.

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
scheduler, finds those executables on your ``PATH`` and registers them, and starts the
background engine. It reports which executables it found; if one you need is missing,
put it on your ``PATH`` and run the command again, or point at it explicitly:

.. code-block:: console

    $ koopmans install --code pw=/opt/qe/bin/pw.x

By default each calculation is given as many MPI processes as your machine has physical
cores. Use ``--procs-per-calc`` to change that, and ``--max-procs`` to cap how many
processes may run at once across all concurrent calculations.

Not every executable is compiled with MPI, and running a serial build under ``mpirun``
starts several copies of it in one directory, where they overwrite each other's files.
``koopmans install`` therefore looks for a call to ``MPI_Init`` in each executable it
registers, and in the shared libraries that executable links, and reports what it
decided:

.. code-block:: text

    MPI:
      pw         parallel  (links libqe_modules.so.7, which calls mpi_init_)
      wannier90  serial    (no MPI_Init call in the binary or the libraries it links)

Merely linking an MPI runtime does not count: a build produced by ``mpif90`` records
``libmpi`` whether or not any MPI call survives compilation. An executable in which no
MPI call can be found is registered serial, which is slower but always correct. If the
answer is wrong for one of your executables, overrule it:

.. code-block:: console

    $ koopmans install --parallel wannier90
    $ koopmans install --serial pw

Rerunning ``koopmans install`` also re-inspects codes registered earlier and replaces any
that runs the wrong way. A replacement code node is a new node, so calculations cached
against the old one are no longer reused and will run again; the install reports which
codes it replaced, and ``--no-migrate`` skips the step entirely.

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
