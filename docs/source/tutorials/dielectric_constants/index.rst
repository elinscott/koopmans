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

.. question:: How does the dielectric constant converge with respect to the :math:`k`-point grid?

   .. image:: convergence.svg
       :class: responsive-fig
       :align: center
       :alt: The dielectric constant of silicon against a Gamma-centred k-point grid,
           from 25.10 at 4x4x4 to 13.30 at 16x16x16

.. tip::

    Those meshes all contain :math:`\Gamma`, where the occupied-to-empty response of a
    semiconductor is at its most structured. Add

    .. code-block:: yaml

        kpoints:
          offset: [0.5, 0.5, 0.5]

    to shift the mesh half a step off :math:`\Gamma` in every direction and rerun the
    series.

.. question:: How much does the shifted mesh help?

   .. image:: convergence_shifted.svg
       :class: responsive-fig
       :align: center
       :alt: The dielectric constant of silicon against the k-point grid for a
           Gamma-centred and a half-shifted mesh; the shifted series reaches 13.28
           by 12x12x12 where the Gamma-centred one is still at 13.45

   Up to a constant factor, and leaving aside the local-field corrections ``ph.x``
   includes, the electronic dielectric constant is a sum over vertical transitions,

   .. math::

       \varepsilon_\infty - 1 \propto \sum_{\mathbf{k}} \sum_{v, c}
       \frac{|\langle c\mathbf{k} | \hat{\mathbf{r}} | v\mathbf{k} \rangle|^2}
            {\varepsilon_{c\mathbf{k}} - \varepsilon_{v\mathbf{k}}}.

   A static field carries no momentum, so each :math:`k`-point couples only its own
   valence and conduction states; the *indirect* gap of silicon (:math:`\Gamma` to near
   :math:`X`) never enters. What matters is the *direct* gap
   :math:`\varepsilon_{c\mathbf{k}} - \varepsilon_{v\mathbf{k}}`, and in silicon that is
   smallest at :math:`\Gamma`, where the summand peaks sharply. A :math:`\Gamma`-centred
   mesh has a sample sitting exactly on that peak, and the sum gives it the same weight as
   every other point, so the peak is counted far too heavily until the mesh is dense enough
   for its neighbours to dilute it. A half-shifted mesh has no sample at :math:`\Gamma`:
   its nearest points sit half a step away on every side, where the summand has already
   come down, so the sum sees the peak only through its surroundings and settles with far
   fewer points.

*******************************************
 Using the value in a Koopmans calculation
*******************************************

To use this value, set it as ``eps_inf`` in the ``workflow`` block of a
``singlepoint`` input:

.. code-block:: yaml

    workflow:
      task: singlepoint
      eps_inf: 13.28

Both screening methods read it from there.

Alternatively, you can do everything in a single calculation: set ``eps_inf: auto`` and
the workflow runs the same scf-then-``ph.x`` calculation this tutorial just ran alongside
the rest of the chain, and the screening step waits for its result in place of a number
you supply.

Left to itself, ``auto`` samples the dielectric step on the singlepoint's own scf mesh,
which as we saw above is unlikely to be dense enough for :math:`\varepsilon_\infty`. Give
the dielectric step its own mesh under ``kpoints.overrides``, as you would any other step:

.. code-block:: yaml

    workflow:
      task: singlepoint
      screening_method: dfpt
      eps_inf: auto
    kpoints:
      grid: [4, 4, 4]
      overrides:
        dielectric:
          grid: [8, 8, 8]
          offset: [0.5, 0.5, 0.5]

runs the rest of the calculation on :math:`4\times4\times4` and the dielectric constant
on the half-shifted :math:`8\times8\times8` mesh that the figure above showed to be within
0.02 of converged.

.. note::

    LDA systematically overestimates :math:`\varepsilon_\infty` relative to experiment
    — silicon's measured value is about 11.7, below every value above.
    This is a known shortcoming of the underlying exchange-correlation functional, not a
    convergence problem, and it is not something a denser :math:`k`-point grid fixes.
