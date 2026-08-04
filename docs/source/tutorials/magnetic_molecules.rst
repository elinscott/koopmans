##########################
 Spin-polarized molecules
##########################

A molecule with an open shell: the oxygen molecule, whose ground state is a spin
triplet. Treating the two spin channels separately changes what a Koopmans calculation
is working with — every orbital now carries a spin channel as well as an index, and it
gets its own screening parameter — so this tutorial is the closed-shell ozone
calculation again, with that difference drawn out. The screening parameters come from
total-energy differences.

.. note::

    This tutorial has not been written yet. It is tracked by `issue #75
    <https://github.com/elinscott/koopmans/issues/75>`_.

A challenge in the meantime: modify the :doc:`ozone tutorial's <ozone>` input file for
molecular oxygen, and see whether the IP and EA compare as well to experiment. O₂ is a
linear molecule with a bond length of 1.21 Å, and — unlike ozone — it is paramagnetic,
so its two spin channels differ.

.. dropdown:: ❔ Question — Do the IP and EA of O₂ compare as well to experiment?
    :color: primary
    :class-title: question-header

    Set ``spin: collinear`` in the ``workflow`` block, add ``tot_magnetization: 2`` and
    lower ``nbnd`` to ``8`` in ``calculator_parameters``, and update the atoms. A
    complete input file — which also runs two screening iterations and lets
    near-degenerate orbitals share a screening parameter — is :download:`o2.yaml
    <o2.yaml>`. Running it gives an IP of 12.35 eV and an EA of
    0.40 eV, against experimental values of 12.07 and 0.45 eV (`NIST
    <https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=20#Ion-Energetics>`_).
