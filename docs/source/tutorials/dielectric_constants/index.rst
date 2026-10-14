######################
 Dielectric constants
######################

A Koopmans calculation on a solid needs the material's macroscopic dielectric constant,
:math:`\varepsilon_\infty` in order to accurately compute the screening parameters. In these
calculations we measure how the crystal responds to a localized charge:
:doc:`total-energy differences <../band_structures/silicon_finite_differences/index>`
do it directly by adding or removing an electron in a periodic supercell, while
:doc:`linear response <../band_structures/silicon_linear_response/index>` measures the same thing to
second order in the primitive cell. Either way, the perturbing charge interacts with
its own periodic images, and the corrections that remove this spurious interaction need
:math:`\varepsilon_\infty` to scale the leading-order term. If you get this parameter wrong, the
image correction will be wrong, along with everything downstream of it!

This tutorial computes :math:`\varepsilon_\infty` for bulk silicon with density-functional
perturbation theory (DFPT) :cite:`Baroni2001`.

****************
 The input file
****************

Download :download:`si_eps.yaml <si_eps.yaml>` and place it in an empty directory. Here it
is in full:

.. literalinclude:: si_eps.yaml
    :language: yaml

.. literalinclude:: si_eps.yaml
    :language: yaml
    :start-at: task
    :end-at: task

runs one ``pw.x`` self-consistent calculation, then ``ph.x`` with ``epsil = .true.`` and
``trans = .false.`` at :math:`q = \Gamma`: the electric-field response, without the
phonon calculation ``ph.x`` would otherwise also do.

.. literalinclude:: si_eps.yaml
    :language: yaml
    :start-at: kpoints:
    :end-at: grid:

sets the Brillouin-zone sampling for both steps.

Run it with

.. code-block:: console

    $ koopmans run si_eps.yaml

*************
 The outputs
*************

The run produces the same two-step tree as any other DFT calculation, ``si_eps/01-scf``
and ``si_eps/02-ph``, plus a top-level ``si_eps/outputs`` collecting the results that
matter beyond either individual step:

.. code-block:: text

    si_eps/
    ├── 01-scf/
    ├── 02-ph/
    └── outputs/
        ├── dielectric_tensor.json
        ├── eps_inf.json
        └── ph_output_parameters.json

``dielectric_tensor.json`` is ph.x's :math:`3 \times 3` dielectric tensor; for cubic
silicon it is already isotropic. ``eps_inf.json`` is the single number a Koopmans
calculation wants: the isotropic average of that tensor.

.. question:: What dielectric constant do you get?

    25.10, read from ``si_eps/outputs/eps_inf.json``.

*************************
 Convergence with k-grid
*************************

.. question:: How does the dielectric constant converge with respect to the :math:`k`-point grid?

   25.10 is a long way from converged. :math:`\varepsilon_\infty` is a Brillouin-zone
   integral over the whole occupied-to-empty response, and it needs a much denser mesh than
   a ground-state total energy to converge.

   .. list-table::
       :header-rows: 1

       * - :math:`k`-point grid
         - :math:`\varepsilon_\infty`
       * - :math:`4\times4\times4`
         - 25.10
       * - :math:`8\times8\times8`
         - 14.59
       * - :math:`12\times12\times12`
         - 13.45
       * - :math:`16\times16\times16`
         - 13.30

   .. image:: convergence.svg
       :class: responsive-fig
       :align: center

*******************************************
 Using the value in a Koopmans calculation
*******************************************

To use this value, set it as ``eps_inf`` in the ``workflow`` block of a
``singlepoint`` input:

.. code-block:: yaml

    workflow:
      task: singlepoint
      eps_inf: 13.30

Both screening methods read it from there.

Alternatively, you can do everything in a single calculation: set ``eps_inf: auto`` and a
``screening_method: dfpt`` workflow runs the same scf-then-``ph.x`` calculation this
tutorial just ran alongside the rest of the chain, and ``kcw.x`` waits for its result in
place of a number you supply. (``screening_method: dscf`` does not accept ``auto`` yet.)

Left to itself, ``auto`` samples the dielectric step on the singlepoint's own scf mesh,
which as we saw above is unlikely to be dense enough for :math:`\varepsilon_\infty`. Specify
``kpoints.eps_inf_factor`` to increase the density of ``kpoints.grid`` for the dielectric step:

.. code-block:: yaml

    workflow:
      task: singlepoint
      screening_method: dfpt
      eps_inf: auto
    kpoints:
      grid: [4, 4, 4]
      eps_inf_factor: 4

runs the rest of the calculation on :math:`4\times4\times4` and the dielectric constant
on :math:`16\times16\times16`.

.. note::

    LDA systematically overestimates :math:`\varepsilon_\infty` relative to experiment
    — silicon's measured value is about 11.7, below every grid in the table above.
    This is a known shortcoming of the underlying exchange-correlation functional, not a
    convergence problem, and it is not something a denser :math:`k`-point grid fixes.
