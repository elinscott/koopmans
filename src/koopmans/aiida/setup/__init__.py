"""Automatic AiiDA setup for koopmans.

This package handles automatic configuration of AiiDA profiles, computers,
codes, and the bundled HyperQueue scheduler so users can run koopmans
calculations without needing AiiDA knowledge. ``koopmans install`` is the
single user entry point; the implementation is split across submodules:

* :mod:`.profile` -- AiiDA profile setup (presto-style).
* :mod:`.computer` -- localhost Computer registration.
* :mod:`.codes` -- PATH scanning + Code registration for QE executables.
* :mod:`.pseudos` -- on-demand pseudopotential family installers.
* :mod:`.daemon` -- AiiDA daemon lifecycle helpers.
* :mod:`.hq` -- HyperQueue binary install + server/worker lifecycle.
* :mod:`.cores` -- physical core detection.
* :mod:`.orchestrate` -- ``koopmans install`` orchestration.

Import from the submodules directly; this package deliberately re-exports nothing.
"""
