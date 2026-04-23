"""Central stat-journal-style plotting palette and matplotlib rcParams.

Used by ``chess_tutor.utils.visualization`` helpers, ``scripts/run_*.py``,
and the demo notebook so that every figure in the project shares a single
coherent look: serif typography, thin axes, blue + gray families only,
and fixed role-colour maps for policies and ELO brackets.
"""

from __future__ import annotations

import matplotlib.pyplot as plt


# --- Colours --------------------------------------------------------------
# Two families only: blues (primary, learning policies, hero categories)
# and grays (baselines, reference lines, secondary).

BLUE_DARK  = "#08306B"
BLUE       = "#0072B2"   # primary — used for single-series plots
BLUE_MID   = "#4292C6"
BLUE_LIGHT = "#9ECAE1"
BLUE_PALE  = "#DEEBF7"
GRAY_DARK  = "#4A4A4A"
GRAY       = "#999999"
GRAY_LIGHT = "#BDBDBD"

PRIMARY = BLUE
ACCENT  = GRAY_DARK       # for reference / chance lines
CHANCE  = GRAY            # for chance / diagonal lines

# Fixed role map: 5 bandit policies — learning methods in blue shades,
# baselines in gray shades.
POLICY_COLOR: dict[str, str] = {
    "Thompson Sampling":  BLUE_DARK,
    "ε-Greedy (ε=0.1)":   BLUE,
    "LinUCB (α=1)":       BLUE_MID,
    "Random":             GRAY,
    "Rule-Based":         GRAY_DARK,
}

# Fixed role map: 5 ELO brackets — ordered sequential blue gradient.
BRACKET_COLOR: dict[int, str] = {
    1100: BLUE_PALE,
    1300: BLUE_LIGHT,
    1500: BLUE_MID,
    1700: BLUE,
    1900: BLUE_DARK,
}

# For ≤5 categories with no natural order: use one blue with different
# linestyles rather than 5 different colours.
LINESTYLE_CYCLE = ["-", "--", "-.", ":", (0, (1, 1))]


# --- rcParams helper ------------------------------------------------------

_JOURNAL_RCPARAMS = {
    "font.family":         "serif",
    "font.size":           10,
    "axes.labelsize":      10,
    "axes.titlesize":      11,
    "axes.linewidth":      0.7,
    "axes.edgecolor":      "#333333",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.labelsize":     9,
    "ytick.labelsize":     9,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "legend.fontsize":     9,
    "legend.frameon":      False,
    "grid.alpha":          0.25,
    "grid.linewidth":      0.4,
    "grid.linestyle":      "--",
    "lines.linewidth":     1.6,
    "axes.grid":           True,
    "axes.axisbelow":      True,
    "figure.dpi":          120,
    "savefig.dpi":         150,
    "savefig.bbox":        "tight",
}


def apply_journal_style() -> None:
    """Install the project-wide stat-journal style on the global rcParams.

    Safe to call multiple times; last call wins. Intended to be called once
    at the top of each script or notebook that produces figures.
    """
    plt.rcParams.update(_JOURNAL_RCPARAMS)


def policy_color(name: str) -> str:
    """Return the fixed colour for a bandit-policy name, PRIMARY as fallback."""
    return POLICY_COLOR.get(name, PRIMARY)


def bracket_color(elo: int) -> str:
    """Return the fixed colour for an ELO bracket, PRIMARY as fallback."""
    return BRACKET_COLOR.get(int(elo), PRIMARY)
