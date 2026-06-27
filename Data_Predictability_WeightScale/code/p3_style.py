"""
Paper#3 shared figure style — single source of truth for palette / font / sizes.
Usage in every make_*.py:

    from p3_style import apply_style, PALETTE, eta_colors, FIGW, FIGH
    apply_style()
    ...
    fig.savefig("figures/p3_xxx.pdf")   # vector for line/scatter/bar
    # heatmaps -> png dpi>=300

Spec: 03_paper/figure_style_spec.md.  Honors 守 #19 (EN) / #76 (no in-fig title) /
#82 (no callout arrows) / 6-19 colormap rule (sequential=viridis/turbo, signed=RdBu_r symmetric).
"""

# ---- semantic palette (no bare hex in scripts) ----
PALETTE = {
    "INK":     "#1a1a1a",   # text / axes / default edge
    "GRID":    "#d9d9d9",   # gridlines
    "FIT":     "#1f3b73",   # regression / theory line (navy)
    "BAND":    "#c9d6e8",   # CI / prediction band (use alpha ~0.5)
    "ARCH_A":  "#2c5aa0",   # Pythia (blue)
    "ARCH_B":  "#e6800d",   # Llama  (orange)  -- replaces cyan/brick-red
    "ACCENT":  "#d62728",   # outlier / emphasis (e.g. code), use sparingly
    "NEUTRAL": "#7a7a7a",   # secondary guide lines
}

# physical sizes (inches) for TMLR single-column text width ~6.5in
FIGW = 6.5          # full width
FIGW_HALF = 3.2
FIGH = 4.0          # single-panel scatter/line
FIGH_WIDE = 3.2     # two-panel row
FIGH_BAR = 3.5

# sequential colormap for ORDERED categories (e.g. the 4 eta levels)
SEQ_CMAP = "viridis"
# magnitude maps
MAG_CMAP = "turbo"
# signed/diverging maps  (use symmetric vmin/vmax via diverging_norm below)
DIV_CMAP = "RdBu_r"


def eta_colors(n=4):
    """n perceptually-ordered colors for an ORDERED variable (eta), low->high."""
    import matplotlib.cm as cm
    import numpy as np
    return [cm.get_cmap(SEQ_CMAP)(x) for x in np.linspace(0.12, 0.88, n)]


def diverging_norm(values):
    """TwoSlopeNorm centered at 0 with symmetric range, for signed quantities (守 6-19)."""
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm
    m = float(np.nanmax(np.abs(values)))
    m = m if m > 0 else 1.0
    return TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)


def apply_style():
    """Global rcParams: one font (CM math + serif text), no bold, thin grid, vector-friendly."""
    import matplotlib as mpl
    ink, grid = PALETTE["INK"], PALETTE["GRID"]
    mpl.rcParams.update({
        # ---- font: Computer-Modern math to match LaTeX body; serif text ----
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,          # use ascii hyphen-minus, avoid missing glyph
        # ---- sizes (paper body 10pt) ----
        "font.size": 9.0,
        "axes.titlesize": 10,                  # panels should NOT set titles (守#76); kept for safety
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "figure.titlesize": 10,
        # ---- weight: never bold ----
        "axes.titleweight": "normal",
        "axes.labelweight": "normal",
        "font.weight": "normal",
        # ---- color / spines / grid ----
        "text.color": ink, "axes.edgecolor": ink, "axes.labelcolor": ink,
        "xtick.color": ink, "ytick.color": ink,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": grid, "grid.linewidth": 0.6, "grid.alpha": 0.4,
        # ---- lines / markers ----
        "lines.linewidth": 1.5, "lines.markersize": 5.5,
        "lines.markeredgecolor": ink, "lines.markeredgewidth": 0.5,
        # ---- legend: no heavy frame ----
        "legend.frameon": False,
        # ---- output: vector-friendly, fonts as text in PDF ----
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


# convenience: explicit two-arch color map
ARCH = {"pythia": PALETTE["ARCH_A"], "llama": PALETTE["ARCH_B"]}
