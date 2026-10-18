"""
Chart styling, so every figure in the repo looks like it came from one place.

Palette is a validated categorical set: adjacent pairs clear both the
colour-vision-deficiency separation floor and the normal-vision floor, so the
series stay distinguishable for a colourblind reader and in greyscale print.
Slots are assigned in fixed order and never cycled — if a chart needs a ninth
series, that is a signal to facet it, not to invent a colour.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots, in fixed assignment order.
SERIES = ["#2a78d6",  # 1 blue
          "#eb6834",  # 2 orange
          "#1baf7a",  # 3 aqua
          "#eda100",  # 4 yellow
          "#e87ba4",  # 5 magenta
          "#008300",  # 6 green
          "#4a3aa7",  # 7 violet
          "#e34948"]  # 8 red

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2df"


def use_style() -> None:
    """Apply the project chart style. Call once per notebook."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",

        # Recessive frame: keep the ink on the data, not the furniture.
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,

        "axes.labelcolor": INK_SOFT,
        "text.color": INK,
        "xtick.color": INK_SOFT,
        "ytick.color": INK_SOFT,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,

        "lines.linewidth": 2.0,      # 2px lines
        "lines.markersize": 5,
        "legend.frameon": False,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def label_end(ax, x, y, text, color, dx=0.0, dy=0.0):
    """
    Direct-label the end of a line.

    Identity should never rest on colour alone. Labelling the last point
    means a reader who cannot separate two hues can still tell the series
    apart, and it removes a legend lookup.
    """
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy),
                color=color, fontsize=9, fontweight="600",
                va="center", ha="left")
