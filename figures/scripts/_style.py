"""Shared IEEE-figure style for the PureProtX resubmission figures.

Hard rules enforced here:
  - NO title is ever drawn on the canvas (captions live in LaTeX).
  - Vector PDF with embedded (Type-42) fonts + 300 dpi PNG preview.
  - Colorblind-safe Okabe-Ito palette; callers must also vary a second channel
    (hatch / marker / linestyle) for grayscale safety.
  - Serif fonts, >= 8 pt at print size, constrained_layout to avoid overlaps.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7", "grey": "#999999",
}

FIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,      # (titles are not used, but keep sane)
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "pdf.fonttype": 42,       # embed TrueType (editable/searchable)
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.constrained_layout.use": True,
})

COL_WIDTH = 3.5    # IEEE single column (in)
DBL_WIDTH = 7.16   # IEEE double column (in)


def save(fig, name):
    """Save <name>.pdf (vector) + <name>.png (300 dpi) into figures/. Assert no title."""
    for ax in fig.get_axes():
        assert not ax.get_title(), f"canvas title present on an axis in {name} (forbidden)"
    assert not fig._suptitle, f"suptitle present in {name} (forbidden)"
    pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    png = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    return pdf, png
