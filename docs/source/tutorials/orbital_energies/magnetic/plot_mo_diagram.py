"""Draw a molecular-orbital diagram for O2 with atomic-orbital correlation.

A textbook three-column diagram, not to scale: no energy axis, no
eigenvalue data. The center column is the standard single-ladder MO
diagram for O2 (Z >= 8, so sigma(2p) sits below the pi(2p) pair); the
left and right columns are each O atom's 2s and 2p atomic levels, with
thin correlation lines fanning out to the molecular levels they form.
Filling is for neutral O2's 12 valence electrons, with the degenerate
pi*(2p) pair each carrying one unpaired spin-up electron -- the triplet
ground state. Each atom's 2s2 2p4 configuration is filled by Hund's
rule: one paired 2p bar, two singly occupied. Each degenerate MO pair
is tinted as a single group, foreshadowing that koopmans screens one
orbital per group and copies the result to the rest; the atomic 2p
trio is degenerate too but is not a screening group, so it is left
untinted.

Every text position is computed from the bar geometry it annotates,
never hand-placed: MO labels sit a fixed gap left of their level's
leftmost bar tip, and AO labels (2s/2p, the only per-atom labels --
there is no separate "O" header) sit a fixed gap outward from their
group's outermost bar tip. The two atom columns share one x_atom
magnitude and one label gap, mirrored by a side sign -- left/right
symmetry follows from that, rather than from independently tuned
numbers, and main() checks it with an assertion after layout. Run
standalone; it writes ``mo_diagram.svg``.
"""

import math
import re

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

X = 0.0
X_ATOM = 2.9  # horizontal offset of each atom's AO column from the center

# One bar width everywhere -- MO singles, MO pair members, AO singles, AO
# trio members -- so no level reads as visually more "important" than any
# other; only the arrangement (single bar, side-by-side pair, side-by-side
# trio) carries degeneracy.
BAR_HALF_WIDTH = 0.26
BAR_GAP = 0.14  # clear space kept between adjacent bars in a degenerate group
PAIR_OFFSET = BAR_HALF_WIDTH + BAR_GAP / 2
_AO_TRIO_OFFSET = 2 * BAR_HALF_WIDTH + BAR_GAP
AO_TRIO_OFFSETS = [-_AO_TRIO_OFFSET, 0.0, _AO_TRIO_OFFSET]

# Bottom-to-top level order, each with its bar count (1 = nondegenerate,
# 2 = degenerate pair) and each bar's occupation: "paired" (up+down arrows),
# "single_up" (one spin-up arrow, as in the pi* triplet electrons), or
# "empty" (dashed bar, no arrow). Names stay shell-tagged (2s/2p) so they are
# unique keys for LEVEL_Y/correlates/mo_tip_x; the drawn label strips the
# shell tag (draw_mo_column) since the AO columns and correlation lines
# already show each MO's atomic parentage.
LEVELS = [
    (r"$\sigma(2s)$", ["paired"]),
    (r"$\sigma^*(2s)$", ["paired"]),
    (r"$\sigma(2p)$", ["paired"]),
    (r"$\pi(2p)$", ["paired", "paired"]),
    (r"$\pi^*(2p)$", ["single_up", "single_up"]),
    (r"$\sigma^*(2p)$", ["empty"]),
]
LEVEL_Y = {name: y for y, (name, _) in enumerate(LEVELS)}

# Each O atom's 2s and 2p levels, placed halfway between the bonding and
# antibonding molecular levels they correlate with -- the classic
# schematic energy correlation -- and the MO levels each fans out to.
AO_LEVELS = [
    {
        "label": "2s",
        "y": (LEVEL_Y[r"$\sigma(2s)$"] + LEVEL_Y[r"$\sigma^*(2s)$"]) / 2,
        "bars": ["paired"],
        "offsets": [0.0],
        "half_width": BAR_HALF_WIDTH,
        "correlates": [r"$\sigma(2s)$", r"$\sigma^*(2s)$"],
    },
    {
        "label": "2p",
        "y": (LEVEL_Y[r"$\sigma(2p)$"] + LEVEL_Y[r"$\sigma^*(2p)$"]) / 2,
        "bars": ["paired", "single_up", "single_up"],
        "offsets": AO_TRIO_OFFSETS,
        "half_width": BAR_HALF_WIDTH,
        "correlates": [r"$\sigma(2p)$", r"$\pi(2p)$", r"$\pi^*(2p)$", r"$\sigma^*(2p)$"],
    },
]

ARROW_LEN = 0.62
ARROW_GAP = 0.15  # horizontal offset between the up/down arrows on a shared bar

# Tint marking a degenerate MO pair as one screening group, sized to clear
# both bars and their arrows.
GROUP_BOX_COLOR = "#e3e3e3"
GROUP_BOX_PAD = 0.12
GROUP_BOX_HALF_WIDTH = PAIR_OFFSET + BAR_HALF_WIDTH + GROUP_BOX_PAD
GROUP_BOX_HALF_HEIGHT = ARROW_LEN / 2 + GROUP_BOX_PAD

# The one pair that also gets a text note; the tint alone marks the other.
ANNOTATED_LEVEL = r"$\pi^*(2p)$"
ANNOTATION_TEXT = "degenerate —\none screening\nparameter\nsuffices"
ANNOTATION_GAP = 0.15  # smaller than LABEL_GAP: clears a tinted box, not a bare bar tip

# Gap between a level's outermost bar tip and its label's anchor point. Used
# for both the MO column (always to the left) and the AO columns (mirrored
# left/right by sign), so left/right AO labels are symmetric by construction
# rather than by two separately tuned numbers.
LABEL_GAP = 0.4

CORRELATION_COLOR = "0.8"


def draw_arrow(ax, x: float, y: float, spin_up: bool) -> None:
    """Draw one spin arrow (shaft plus triangular head) centered on (x, y)."""
    half = ARROW_LEN / 2
    y0, y1 = (y - half, y + half) if spin_up else (y + half, y - half)
    ax.plot([x, x], [y0, y1], color="black", linewidth=2.0, solid_capstyle="butt")
    ax.plot([x], [y1], marker="^" if spin_up else "v", markersize=7, color="black")


def draw_bar(ax, x_center: float, y: float, occupation: str, half_width: float) -> None:
    """Draw one level bar and its arrow(s)."""
    if occupation == "empty":
        ax.plot(
            [x_center - half_width, x_center + half_width],
            [y, y],
            color="0.4",
            linewidth=1.4,
            linestyle=(0, (3, 2)),
            solid_capstyle="butt",
        )
        return

    ax.plot(
        [x_center - half_width, x_center + half_width],
        [y, y],
        color="black",
        linewidth=2.0,
        solid_capstyle="butt",
    )
    if occupation == "single_up":
        draw_arrow(ax, x_center, y, spin_up=True)
    elif occupation == "paired":
        arrow_gap = min(ARROW_GAP, half_width * 0.5)
        draw_arrow(ax, x_center - arrow_gap, y, spin_up=True)
        draw_arrow(ax, x_center + arrow_gap, y, spin_up=False)


def draw_group_box(ax, x_center: float, y: float) -> None:
    """Tint the footprint of a degenerate MO pair to mark it as one screening group."""
    box = FancyBboxPatch(
        (x_center - GROUP_BOX_HALF_WIDTH, y - GROUP_BOX_HALF_HEIGHT),
        2 * GROUP_BOX_HALF_WIDTH,
        2 * GROUP_BOX_HALF_HEIGHT,
        boxstyle="round,pad=0,rounding_size=0.08",
        linewidth=0,
        facecolor=GROUP_BOX_COLOR,
        zorder=0,
    )
    ax.add_patch(box)


def draw_level(
    ax,
    x_center: float,
    y: float,
    bars: list[str],
    offsets: list[float],
    half_width: float,
    tint: bool,
) -> None:
    """Draw one level's bar(s) at the given offsets from x_center."""
    if tint:
        draw_group_box(ax, x_center, y)
    for offset, occupation in zip(offsets, bars, strict=True):
        draw_bar(ax, x_center + offset, y, occupation, half_width)


def mo_tip_x(name: str, side: str) -> float:
    """X of the outer tip of the MO bar on the given side that faces that atom."""
    bars = dict(LEVELS)[name]
    sign = -1 if side == "left" else 1
    if len(bars) == 1:
        return X + sign * BAR_HALF_WIDTH
    return X + sign * (PAIR_OFFSET + BAR_HALF_WIDTH)


def ao_inner_tip_x(ao: dict, x_atom: float, side: str) -> float:
    """X of the AO bar tip that faces the center column (correlation-line endpoint)."""
    offsets = ao["offsets"]
    inner_offset = max(offsets) if side == "left" else min(offsets)
    sign = 1 if side == "left" else -1
    return x_atom + inner_offset + sign * ao["half_width"]


def ao_outer_tip_x(ao: dict, x_atom: float, side: str) -> float:
    """X of the AO bar tip that faces away from the center column (label anchor)."""
    offsets = ao["offsets"]
    outer_offset = min(offsets) if side == "left" else max(offsets)
    sign = -1 if side == "left" else 1
    return x_atom + outer_offset + sign * ao["half_width"]


def mo_label_x(name: str) -> float:
    """X of the MO level label: LABEL_GAP left of the level's leftmost bar tip."""
    return mo_tip_x(name, "left") - LABEL_GAP


def ao_label_x(ao: dict, x_atom: float, side: str) -> float:
    """X of an AO level label: LABEL_GAP outward from the group's outermost bar tip."""
    sign = -1 if side == "left" else 1
    return ao_outer_tip_x(ao, x_atom, side) + sign * LABEL_GAP


def draw_mo_column(ax) -> None:
    """Draw the center column: the molecular-orbital ladder and its labels."""
    for y, (name, bars) in enumerate(LEVELS):
        offsets = [0.0] if len(bars) == 1 else [-PAIR_OFFSET, PAIR_OFFSET]
        draw_level(ax, X, y, bars, offsets, BAR_HALF_WIDTH, tint=len(bars) > 1)
        display_name = re.sub(r"\(2[sp]\)", "", name)
        ax.text(
            mo_label_x(name), y, display_name, fontsize=11, ha="right", va="center", color="0.15"
        )
        if name == ANNOTATED_LEVEL:
            ax.text(
                X + GROUP_BOX_HALF_WIDTH + ANNOTATION_GAP,
                y,
                ANNOTATION_TEXT,
                fontsize=8,
                ha="left",
                va="center",
                color="0.35",
                style="italic",
            )


def draw_atom_column(ax, x_atom: float, side: str) -> None:
    """Draw one O atom's AO column (2s, 2p) and its label, mirrored by side."""
    for ao in AO_LEVELS:
        draw_level(ax, x_atom, ao["y"], ao["bars"], ao["offsets"], ao["half_width"], tint=False)
        ax.text(
            ao_label_x(ao, x_atom, side),
            ao["y"],
            ao["label"],
            fontsize=10,
            ha="right" if side == "left" else "left",
            va="center",
            color="0.15",
        )


def draw_correlations(ax, x_atom: float, side: str) -> None:
    """Draw thin bar-tip-to-bar-tip lines from one atom's AO levels to the MOs they form."""
    for ao in AO_LEVELS:
        x_ao = ao_inner_tip_x(ao, x_atom, side)
        for mo_name in ao["correlates"]:
            ax.plot(
                [x_ao, mo_tip_x(mo_name, side)],
                [ao["y"], LEVEL_Y[mo_name]],
                color=CORRELATION_COLOR,
                linewidth=0.7,
                zorder=-1,
                solid_capstyle="butt",
            )


def check_ao_label_symmetry() -> None:
    """Assert every AO label's x-position mirrors exactly between the two sides."""
    for ao in AO_LEVELS:
        left_x = ao_label_x(ao, -X_ATOM, "left")
        right_x = ao_label_x(ao, X_ATOM, "right")
        if not math.isclose(abs(left_x), abs(right_x), rel_tol=1e-12):
            raise ValueError(
                f"AO label '{ao['label']}' not mirrored: left={left_x}, right={right_x}"
            )


def main() -> None:
    """Build the three-column O2 MO diagram and write mo_diagram.svg."""
    check_ao_label_symmetry()

    fig, ax = plt.subplots(figsize=(8.6, 5.6))

    for x_atom, side in [(-X_ATOM, "left"), (X_ATOM, "right")]:
        draw_correlations(ax, x_atom, side)

    draw_mo_column(ax)

    for x_atom, side in [(-X_ATOM, "left"), (X_ATOM, "right")]:
        draw_atom_column(ax, x_atom, side)

    ax.set_xlim(-4.7, 4.7)
    ax.set_ylim(-0.7, len(LEVELS) - 1 + 0.7)  # symmetric margin now that the "O" headers are gone
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.savefig("mo_diagram.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
