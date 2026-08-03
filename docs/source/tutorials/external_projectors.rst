#######################################
 Wannierizing from external projectors
#######################################

Wannier functions are usually built by projecting onto the pseudoatomic orbitals that
the pseudopotentials carry. You can supply your own instead, one file of radial
projectors per element, which is useful when the pseudopotentials' own orbitals are a
poor starting guess for the manifold you want.

The page will show what those files look like, how their angular momenta fix the number
of Wannier functions, and how choosing where the projectors come from is a separate
decision from how the Wannierization is split into blocks.

.. note::

    This tutorial has not been written yet. It is tracked by `issue #77
    <https://github.com/elinscott/koopmans/issues/77>`_.
