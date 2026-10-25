"""Shared figure style for the UA-FLP benchmark paper figures.

Print-friendly: white background, colour-blind-safe categorical palette (validated with the dataviz
skill validator, adjacent CVD dE >= 9.1), plus a distinct marker and hatch per method so every figure
also reads in grayscale. Method order and colours are fixed across all figures.
"""
import matplotlib as mpl

# single / double column widths (inches)
COL1, COL2 = 3.5, 6.3   # 6.3 in = text width of the main paper (A4, 25 mm margins); figures are placed at 100 %

# fixed method order (benchmark comparison)
METHODS = ["construct", "swap", "ga", "sa", "cro", "ours"]
LABEL = {"construct": "Construct", "swap": "Swap", "ga": "GA", "sa": "SA", "cro": "CRO",
         "ours": "Ours", "mip": "MIP", "cpsat": "CP-SAT", "minlp": "MINLP",
         "sa+polish": "SA + polish", "cro+polish": "CRO + polish",
         "ours_s1": "Stage 1 only", "ours_s2": "Stage 1 + polish", "ours_nos1": "No Stage 1",
         "ours_noelite": "Elite pool 1", "ours_unif": "Uniform partner"}
COLOR = {"ours": "#2a78d6", "sa": "#eb6834", "cro": "#1baf7a", "ga": "#eda100", "swap": "#e87ba4",
         "construct": "#7a7974", "minlp": "#0b0b0b"}
# darker edge of the same hue, so light fills still have a visible outline in print
EDGE = {"ours": "#1c5cab", "sa": "#b8481d", "cro": "#127a55", "ga": "#a66f00", "swap": "#b04f77",
        "construct": "#3d3c39", "minlp": "#0b0b0b"}
MARKER = {"minlp": "P", "construct": "X", "swap": "v", "ga": "D", "sa": "s", "cro": "^", "ours": "o"}
HATCH = {"construct": "", "swap": "\\\\\\\\", "ga": "xxxx", "sa": "////", "cro": "....", "ours": ""}
LINESTYLE = {"construct": ":", "swap": (0, (1, 1)), "ga": "-.", "sa": "--", "cro": (0, (5, 1, 1, 1)),
             "ours": "-"}

# ablation variants: one hue family (grays) + Ours in its own colour, distinct markers
ABL = ["ours", "ours_noelite", "ours_unif", "ours_s2", "ours_s1", "ours_nos1"]
ABL_COLOR = {"ours": "#2a78d6", "ours_noelite": "#3d3c39", "ours_unif": "#3d3c39", "ours_s2": "#6b6a66",
             "ours_s1": "#6b6a66", "ours_nos1": "#8c8b86"}
ABL_MARKER = {"ours": "o", "ours_noelite": "s", "ours_unif": "D", "ours_s2": "^", "ours_s1": "v",
              "ours_nos1": "P"}

INSTANCES = ["O7", "vC10a", "AB20", "SC30", "Du62"]
REF = {"O7": 131.6531381, "vC10a": 18522.79, "AB20": 5021.17, "SC30": 3343.47, "Du62": 3635307.30}
BUDGET = {"O7": 20, "vC10a": 20, "AB20": 60, "SC30": 60, "Du62": 150}

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def apply():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "legend.frameon": False, "lines.linewidth": 1.4, "hatch.linewidth": 0.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,          # embed TrueType fonts
        "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, path_noext):
    fig.savefig(path_noext + ".pdf")
    fig.savefig(path_noext + ".png", dpi=300)
