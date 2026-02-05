"""
Shared plotting style for paper experiments.

Professional, publication-quality style for Economics and Computation papers.
Based on best practices from:
- https://github.com/jbmouret/matplotlib_for_papers
- https://github.com/garrettj403/SciencePlots
- https://allanchain.github.io/blog/post/mpl-paper-tips/
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# Style Configuration
# =============================================================================

# Figure sizes (inches) - based on typical LaTeX column widths
SINGLE_COL_WIDTH = 3.5      # Single column (~89mm)
DOUBLE_COL_WIDTH = 7.0      # Double column / full width (~180mm)
GOLDEN_RATIO = 1.618

# Color palette - muted, colorblind-friendly
# Inspired by Tableau 10 but desaturated for print
COLORS = {
    "blue": "#4878A6",       # Muted blue
    "orange": "#D4682C",     # Muted orange
    "green": "#59A257",      # Muted green
    "red": "#C44E52",        # Muted red
    "purple": "#8C6BAD",     # Muted purple
    "gray": "#6B6B6B",       # Neutral gray
    "brown": "#937860",      # Warm brown
}

# Metric-specific colors (consistent across all experiments)
METRIC_COLORS = {
    "pd": COLORS["blue"],
    "polis": COLORS["orange"],
    "pmean": COLORS["green"],
}

# Markers (distinct shapes)
METRIC_MARKERS = {
    "pd": "o",      # Circle
    "polis": "s",   # Square
    "pmean": "D",   # Diamond
}

# Labels
METRIC_LABELS = {
    "pd": "PD",
    "polis": "Pol.is GIC",
    "pmean": "$p$-mean",
}


def setup_style(use_latex=True):
    """Apply publication-quality matplotlib style settings.

    Args:
        use_latex: If True, use LaTeX for text rendering (requires LaTeX installation).
                   This gives exact font matching with ACM/EC papers.
    """

    # Use non-GUI backend
    matplotlib.use("Agg")

    # Common style parameters for both modes
    common_params = {
        # Axes styling
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.axisbelow": True,

        # Tick styling - inward ticks
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "xtick.major.width": 0.6,
        "xtick.minor.size": 1.5,
        "xtick.minor.width": 0.4,
        "ytick.major.size": 3,
        "ytick.major.width": 0.6,
        "ytick.minor.size": 1.5,
        "ytick.minor.width": 0.4,
        "xtick.color": "#333333",
        "ytick.color": "#333333",

        # Grid - subtle
        "axes.grid": False,
        "grid.color": "#E0E0E0",
        "grid.linewidth": 0.4,
        "grid.alpha": 0.7,

        # Lines
        "lines.linewidth": 1.5,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.8,
        "lines.markeredgecolor": "white",

        # Legend - no frame
        "legend.frameon": False,
        "legend.borderpad": 0.4,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.5,
        "legend.handletextpad": 0.4,

        # Figure
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "figure.edgecolor": "white",
        "figure.constrained_layout.use": True,

        # Saving
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",

        # Patch (for fill_between, etc.)
        "patch.linewidth": 0.5,
    }

    if use_latex:
        # Use LaTeX rendering with libertine package for exact ACM font matching
        style_params = {
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{libertine}\usepackage[libertine]{newtxmath}",
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    else:
        # Fallback: use best available system fonts
        style_params = {
            "font.family": "serif",
            "font.serif": ["Palatino", "Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "mathtext.fontset": "stix",
        }

    # Merge common params with mode-specific params
    style_params.update(common_params)
    plt.rcParams.update(style_params)


def get_figure_size(width="single", aspect=None):
    """
    Get figure size for publication.

    Args:
        width: "single" (~3.5in), "double" (~7in), or float (inches)
        aspect: height/width ratio (default: 1/golden_ratio)

    Returns:
        (width, height) tuple in inches
    """
    if width == "single":
        w = SINGLE_COL_WIDTH
    elif width == "double":
        w = DOUBLE_COL_WIDTH
    else:
        w = float(width)

    if aspect is None:
        aspect = 1 / GOLDEN_RATIO

    return (w, w * aspect)


def add_panel_label(ax, label, loc="upper left", offset=(-0.12, 0.02)):
    """
    Add panel label (a), (b), etc. for multi-panel figures.

    Args:
        ax: matplotlib axes
        label: string like "(a)" or "A"
        loc: location string
        offset: (x, y) offset from corner in axes coordinates
    """
    ax.text(
        offset[0], 1 + offset[1], label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def format_metric_value(value, precision=3):
    """Format a metric value with consistent precision."""
    if np.isnan(value):
        return "—"
    return f"{value:.{precision}f}"


def despine(ax, left=False, bottom=False):
    """Remove spines from axes (top/right always, optionally left/bottom)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if left:
        ax.spines["left"].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(False)


# Ridgeline-specific settings
RIDGELINE_CMAP = "RdBu_r"  # Red-Blue reversed (blue=left, red=right)
RIDGELINE_ALPHA = 0.75
RIDGELINE_OUTLINE_COLOR = "#333333"
RIDGELINE_OUTLINE_WIDTH = 0.4
RIDGELINE_BASELINE_COLOR = "#CCCCCC"
RIDGELINE_BASELINE_WIDTH = 0.3


# Initialize style on import
setup_style()
