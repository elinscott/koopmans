"""Display names for the steps the progress table lists.

Every name is looked up, never derived. A label the table does not name is
shown exactly as it is written, so a missing entry reads as an internal
name rather than as a guess at one.

The lookup is keyed on the pair ``(link label, process label)``, falling
back to the link label alone: one string can name a container in one place
and the calculation inside it in another.

Two conventions the entries follow:

* a row that stands for a physical step is a noun phrase (``Koopmans
  DFPT``), because the table lists things rather than instructions;
* a row that is pure mechanism gets a short mechanical name
  (``Preprocessing``, ``Overlaps``, ``Minimization``), and a step whose
  product the next step reads is named after that product (``Wannier
  gauge``, ``Merged Wannier manifold``).

Executables are quoted verbatim in a column of their own and never
inflected, so no display name repeats one.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["LabelDisplay", "describe_label", "executable_for", "prettify_label"]


class LabelDisplay(NamedTuple):
    """How one process is shown: its name, its executable, and its role."""

    text: str
    code: str | None = None
    transparent: bool = False
    numbered: bool = False


# The executable each code label runs, spelled the way its authors spell
# it. ``get_node_label`` prefixes a CalcJob's label with the label of the
# code it ran (``pw-scf``); the prefix names the code, the rest names the
# step. A code label with no entry here is shown as configured.
_EXECUTABLES = {
    "pw": "pw.x",
    "ph": "ph.x",
    "projwfc": "projwfc.x",
    "wannier90": "wannier90.x",
    "pw2wannier90": "pw2wannier90.x",
    "decompose": "pw2wannier90.x",
    "kcp": "kcp.x",
    "kcw": "kcw.x",
    "wann2kcp": "wann2kcp.x",
    "merge_evc": "merge_evc.x",
    "wannierjl": "wannier.jl",
}

# Containers that add no idea the row above does not already state. They
# get no row; their children rise to the parent's depth. A container that
# groups two genuinely different steps (``scf_nscf`` → "Ground state")
# keeps its row, because that grouping is information.
_TRANSPARENT: frozenset[str | tuple[str, str]] = frozenset(
    {
        "PwBandsWorkChain",
        "Wannier90WorkChain",
        "refine_screening_parameters",
        ("wannier90", "Wannier90WorkChain"),
    }
)

# Steps numbered by their position among their siblings rather than by
# anything in their label: the screening recursion names every iteration
# the same, and only the order distinguishes them.
_NUMBERED = frozenset({"ScreeningIteration", "screening_iteration"})

_DISPLAY: dict[str | tuple[str, str], str] = {
    # --- workflow roots ---
    "RunPwBands": "DFT band structure",
    "DielectricTask": "Dielectric constant",
    "WannierizeBlocks": "Wannierization",
    "Wannierize": "Wannierization",
    "KoopmansDSCFWorkflow": "Koopmans ΔSCF",
    "SinglepointDFPTWorkflow": "Koopmans DFPT",
    "TrajectoryWorkflow": "Trajectory",
    # --- pw.x ---
    "scf_nscf": "Ground state",
    "scf": "SCF",
    "nscf": "NSCF",
    "bands": "Band structure",
    # --- ph.x, projwfc.x ---
    "dielectric": "Dielectric constant",
    "ph": "Dielectric response",
    "projwfc": "Atomic projections",
    # --- wannier90 ---
    # The three calls of the wannier90 protocol are mechanism, not
    # physics: -pp writes the .nnkp that pw2wannier90.x needs, pw2wannier
    # writes the overlaps wannier90.x minimizes over.
    "wannier90_pp": "Preprocessing",
    "pw2wannier90": "Overlaps",
    "wannier90": "Minimization",
    "wannierize": "Wannierization",
    "wannierize_whole_block": "Whole-block Wannierization",
    "split_wannierization": "Parallel-transport split",
    "rewannierize_split_blocks": "Per-group Wannierization",
    "wannier_initialization": "Wannier initialization",
    # --- fold to supercell ---
    # wann2kcp.x writes each block's Wannier functions on the supercell
    # grid; merge_evc.x concatenates them into the single evc file the
    # supercell kcp.x run starts from.
    "fold_to_supercell": "Supercell folding",
    # --- kcp.x ---
    "dft_init": "DFT initialization",
    "dft_dummy": "DFT staging",
    "dft_init_nspin1": "DFT initialization (nspin=1)",
    "dft_init_nspin2_dummy": "DFT initialization (nspin=2, staging)",
    "dft_init_nspin2": "DFT initialization (nspin=2)",
    "ki_trial": "Trial KI",
    "kipz_trial": "Trial KIPZ",
    "dft_n_minus_1": "DFT (N-1)",
    "dft_n_plus_1": "DFT (N+1)",
    "dft_n_plus_1_dummy": "DFT (N+1, staging)",
    "kipz_n_minus_1": "KIPZ (N-1)",
    "kipz_n_plus_1": "KIPZ (N+1)",
    "pz_print": "PZ staging",
    "kipz_print": "KIPZ staging",
    "ki_final": "Final KI",
    "kipz_final": "Final KIPZ",
    "RunFinalKI": "Final KI",
    "run_final_ki_predicted": "Final KI (predicted alphas)",
    "ComputeScreeningParameters": "Screening parameters",
    "PredictScreeningParameters": "Predicted screening parameters",
    "compute_orbital_screening_parameters": "Orbital screening",
    "ScreeningIteration": "Iteration",
    "screening_iteration": "Iteration",
    # --- kcw.x ---
    "dfpt": "DFPT screening",
    "wann2kc": "Wannier gauge",
    "grouped_screen": "Orbital screening",
    "screen": "Screening parameters",
    "ham": "Koopmans Hamiltonian",
    # --- machine learning ---
    "descriptors": "Descriptors",
    "predicted_descriptors": "Descriptors",
    # --- names seen only in the failure summary ---
    # That summary is keyed on ``process_label``, not on the call link
    # label: a class name for a CalcJob or WorkChain, the function's own
    # name for a PyFunction. The two model steps here are PyFunctions,
    # which the table drops; the refinement is a graph the table sees
    # through, and a cascading failure names it all the same.
    "train_screening_model": "Screening model training",
    "evaluate_screening_model": "Screening model evaluation",
    "RefineScreeningParameters": "Screening refinement",
    "PwCalculation": "pw.x",
    "PwBaseWorkChain": "pw.x",
    "PwBandsWorkChain": "DFT band structure",
    "PhCalculation": "ph.x",
    "PhBaseWorkChain": "ph.x",
    "ProjwfcCalculation": "projwfc.x",
    "ProjwfcBaseWorkChain": "projwfc.x",
    "Wannier90Calculation": "wannier90.x",
    "Wannier90BaseWorkChain": "wannier90.x",
    "Wannier90WorkChain": "Wannierization",
    "Wannier90OptimizeWorkChain": "Wannierization",
    "Pw2wannier90Calculation": "pw2wannier90.x",
    "Pw2wannier90BaseWorkChain": "pw2wannier90.x",
    "Pw2wannierDecomposeCalculation": "pw2wannier90.x",
    "KcpCalculation": "kcp.x",
    "KcwCalculation": "kcw.x",
    "Wann2kcCalculation": "kcw.x",
    "KcwScreenCalculation": "kcw.x",
    "KcwHamCalculation": "kcw.x",
    "Wann2kcpCalculation": "wann2kcp.x",
    "MergeEvcCalculation": "merge_evc.x",
}

# The names of :data:`_EXECUTABLES`, so a display name can be recognised
# as naming a binary rather than a step.
_EXECUTABLE_NAMES = frozenset(_EXECUTABLES.values())

_SPIN = {"up": "spin up", "down": "spin down"}
_MANIFOLD = {"occ": "occupied block", "emp": "empty block", "block": "block"}

# Stems whose remainder identifies a block: ``occ``, ``occ_1``,
# ``emp_up_2``, ``block_1``. Longest stem first, so the split variant wins.
_BLOCK_STEMS = (
    ("wannierize_split_", "Split Wannierization"),
    ("wannierize_", "Wannierization"),
    ("fold_", "Supercell Wannier functions"),
    ("decompose_", "Decomposition"),
    ("descriptors_", "Descriptors"),
)

# Stems whose remainder identifies one orbital. The parent row already
# says the step is a screening, so these keep only the identity.
_ORBITAL_STEMS = ("compute_alpha_", "screen_")


def _block_qualifier(rest: str) -> str | None:
    """Render an ``occ`` / ``occ_1`` / ``emp_up_2`` / ``block_1`` remainder, or ``None``.

    The index is optional: a manifold Wannierized as one block is labelled
    without one, and then the qualifier names the manifold alone.
    """
    match = re.fullmatch(r"(occ|emp|block)(?:_(up|down))?(?:_(\d+))?", rest)
    if not match:
        return None
    manifold, spin, index = match.groups()
    text = _MANIFOLD[manifold] if index is None else f"{_MANIFOLD[manifold]} {index}"
    return f"{text}, {_SPIN[spin]}" if spin else text


def _orbital_qualifier(rest: str) -> str | None:
    """Render an ``orb_3`` / ``up_orb_10`` remainder, or ``None``."""
    match = re.fullmatch(r"(?:(up|down)_)?orb_(\d+)", rest)
    if not match:
        return None
    spin, index = match.groups()
    return f"Orbital {index} ({_SPIN[spin]})" if spin else f"Orbital {index}"


# Labels a run-time index writes whole. Indices read as the user counts
# them, from 1, even where the label counts from 0.
_ASSEMBLED_PATTERNS: tuple[tuple[re.Pattern[str], Callable[[re.Match[str]], str]], ...] = (
    (
        re.compile(r"(wannierize|dfpt)_(up|down)"),
        lambda m: f"{_DISPLAY[m.group(1)]} ({_SPIN[m.group(2)]})",
    ),
    (
        re.compile(r"wannier90_split_block_(\d+)"),
        lambda m: f"Minimization (group {int(m.group(1)) + 1})",
    ),
    (
        re.compile(r"merge_evc0?_(occupied|empty)(\d+)"),
        lambda m: f"Merged Wannier manifold ({m.group(1)}, spin {m.group(2)})",
    ),
    (re.compile(r"dscf_snapshot_(\d+)"), lambda m: f"Snapshot {m.group(1)}"),
    (
        re.compile(r"alpha_and_eigenvalue_deltas_snapshot_(\d+)"),
        lambda m: f"Alpha and eigenvalue deltas (snapshot {m.group(1)})",
    ),
)


def _assembled(raw: str) -> str | None:
    """Render a label built at run time from a step plus an identity.

    Returns ``None`` for anything that is not one of these forms, which
    leaves it to be shown verbatim.
    """
    for stem, base in _BLOCK_STEMS:
        if raw.startswith(stem):
            qualifier = _block_qualifier(raw[len(stem) :])
            if qualifier:
                return f"{base} ({qualifier})"
    for stem in _ORBITAL_STEMS:
        if raw.startswith(stem):
            qualifier = _orbital_qualifier(raw[len(stem) :])
            if qualifier:
                return qualifier
    for pattern, render in _ASSEMBLED_PATTERNS:
        match = pattern.fullmatch(raw)
        if match:
            return render(match)
    return None


def describe_label(raw: str, process_label: str = "") -> LabelDisplay:
    """Return how one process is displayed, given its label.

    ``raw`` is a call link label as :func:`~koopmans.aiida.utils.get_node_label`
    builds it — prefixed with the code label for a CalcJob (``pw-scf``) —
    or a ``process_label`` for the root row and the failure summary.
    ``process_label`` disambiguates a link label that names different
    things in different places.

    Examples:
    >>> describe_label("pw-scf").text
    'SCF'
    >>> describe_label("pw-scf").code
    'pw.x'
    >>> describe_label("wannier90-wannier90_pp")
    LabelDisplay(text='Preprocessing', code='wannier90.x', transparent=False, numbered=False)
    >>> describe_label("wannier90", "Wannier90WorkChain").transparent
    True
    >>> describe_label("beam_me_up").text
    'beam_me_up'
    """
    if not raw:
        return LabelDisplay(raw)
    code: str | None = None
    if "-" in raw and raw.split("-", 1)[0].islower():
        prefix, raw = raw.split("-", 1)
        code = _EXECUTABLES.get(prefix, prefix)
    # ``aiida-workgraph`` wraps the top-level process_label as
    # ``WorkGraph<KoopmansDSCFWorkflow>``; the root row is the only place
    # it appears, and there the envelope says nothing the context does not.
    match = re.fullmatch(r"WorkGraph<(.+)>", raw)
    if match:
        raw = match.group(1)
    text = _DISPLAY.get((raw, process_label)) or _DISPLAY.get(raw) or _assembled(raw) or raw
    transparent = (raw, process_label) in _TRANSPARENT or raw in _TRANSPARENT
    return LabelDisplay(text, code, transparent, raw in _NUMBERED)


def executable_for(process_label: str) -> str | None:
    """Return the executable a process label names, or ``None``.

    Only a display name filed under one of the entries of
    :data:`_EXECUTABLES` qualifies, so a process named after a step
    rather than after a binary, and one the table does not name at all,
    both answer ``None``.

    Examples:
    >>> executable_for("PwBaseWorkChain")
    'pw.x'
    >>> executable_for("ScreeningIteration") is None
    True
    """
    name = _DISPLAY.get(process_label)
    return name if name in _EXECUTABLE_NAMES else None


def prettify_label(raw: str, process_label: str = "") -> str:
    """Return the display text for one process's label.

    Examples:
    >>> prettify_label("ki_trial")
    'Trial KI'
    >>> prettify_label("kcp-dft_n_plus_1_dummy")
    'DFT (N+1, staging)'
    >>> prettify_label("WorkGraph<KoopmansDSCFWorkflow>")
    'Koopmans ΔSCF'
    >>> prettify_label("PwCalculation")
    'pw.x'
    """
    return describe_label(raw, process_label).text
