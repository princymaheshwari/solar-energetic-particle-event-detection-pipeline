"""
experiments/poster_flowchart.py

Generate a methodology flowchart for the research poster.
GSU blue/white theme. Sized for a vertical column slot (~1/4 poster width).

Usage:
    cd sep_detection_pipeline/
    python -m experiments.poster_flowchart
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTPUT_DIR = PROJECT_ROOT / "output" / "poster_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# GSU Theme
GSU_BLUE = "#0039A6"
GSU_BLUE_LIGHT = "#4A7FCC"
GSU_BLUE_PALE = "#D6E4F5"
WHITE = "#FFFFFF"
DARK_TEXT = "#1A1A1A"
ACCENT_ORANGE = "#CC5500"
GRAY_BORDER = "#BBBBBB"
LIGHT_BG = "#F5F7FA"

FONT = "Arial"

DPI = 300


def draw_box(ax, cx, cy, w, h, text, fill=GSU_BLUE, text_color=WHITE,
             fontsize=7.5, fontweight="bold", border_color=None, border_lw=1.2,
             alpha=1.0, style="round,pad=0.15", linestyle="-"):
    """Draw a rounded box centered at (cx, cy) with multiline text."""
    if border_color is None:
        border_color = fill

    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=style,
        facecolor=fill, edgecolor=border_color,
        linewidth=border_lw, alpha=alpha, linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(box)

    ax.text(cx, cy, text,
            ha="center", va="center", fontsize=fontsize,
            fontweight=fontweight, color=text_color,
            fontfamily=FONT, zorder=3,
            linespacing=1.35, wrap=False)

    return box


def draw_arrow(ax, x_start, y_start, x_end, y_end,
               color=GSU_BLUE, lw=1.5, head_w=0.08, head_l=0.06):
    """Draw a straight arrow between two points."""
    arrow = FancyArrowPatch(
        (x_start, y_start), (x_end, y_end),
        arrowstyle=f"->,head_width={head_w},head_length={head_l}",
        color=color, linewidth=lw, zorder=1,
    )
    ax.add_patch(arrow)


def draw_side_label(ax, cx, cy, text, fontsize=6, color="#555555"):
    """Small italic annotation to the side of a box."""
    ax.text(cx, cy, text,
            ha="center", va="center", fontsize=fontsize,
            color=color, fontfamily=FONT, style="italic", zorder=3)


def main():
    fig_w, fig_h = 5.5, 11.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    fig.patch.set_facecolor(WHITE)

    box_w = 4.6
    box_w_half = 2.15
    gap_between_halves = 0.2
    x_center = fig_w / 2
    x_left = x_center - box_w_half / 2 - gap_between_halves / 2
    x_right = x_center + box_w_half / 2 + gap_between_halves / 2

    arrow_gap = 0.25

    # ── Tier 0: Section Title ──
    ax.text(x_center, 11.1, "METHODOLOGY",
            ha="center", va="center", fontsize=15,
            fontweight="bold", color=GSU_BLUE, fontfamily=FONT)

    # ── Tier 1: Raw Data Sources ──
    y1 = 10.35
    h1 = 0.55
    draw_box(ax, x_center, y1, box_w, h1,
             "Raw Satellite Data\n"
             "GOES CSV   |   SOHO CDF   |   GOES-16 NetCDF",
             fill=GSU_BLUE_PALE, text_color=DARK_TEXT,
             fontsize=7.5, fontweight="normal", border_color=GSU_BLUE)

    ax.text(x_center, y1 - h1 / 2 - 0.12,
            "5-min cadence   |   1995 - 2025   |   3 archive sources",
            ha="center", va="center", fontsize=5.5, color="#777777",
            fontfamily=FONT, style="italic")

    draw_arrow(ax, x_center, y1 - h1 / 2 - 0.2, x_center, y1 - h1 / 2 - 0.2 - arrow_gap)

    # ── Tier 2: Instrument Adapters (split into two boxes) ──
    y2 = 8.75
    h2 = 1.35

    # GOES box
    draw_box(ax, x_left, y2, box_w_half, h2,
             "",
             fill=WHITE, text_color=DARK_TEXT,
             fontsize=6, fontweight="normal",
             border_color=GSU_BLUE, border_lw=1.8)

    ax.text(x_left, y2 + h2 / 2 - 0.15, "GOES Adapter",
            ha="center", va="center", fontsize=8,
            fontweight="bold", color=GSU_BLUE, fontfamily=FONT, zorder=3)

    goes_lines = [
        "EPS / EPEAD (1995-2020):",
        "  read pre-computed integral",
        "  flux (p3_flux_ic, ZPGT10E)",
        "",
        "SGPS (2020-2025):",
        "  derive >10 MeV from",
        "  13 differential channels",
    ]
    for i, line in enumerate(goes_lines):
        ax.text(x_left - box_w_half / 2 + 0.18, y2 + h2 / 2 - 0.38 - i * 0.14,
                line, ha="left", va="center", fontsize=5.8,
                color=DARK_TEXT, fontfamily=FONT, zorder=3)

    # SOHO box
    draw_box(ax, x_right, y2, box_w_half, h2,
             "",
             fill=WHITE, text_color=DARK_TEXT,
             fontsize=6, fontweight="normal",
             border_color=ACCENT_ORANGE, border_lw=1.8)

    ax.text(x_right, y2 + h2 / 2 - 0.15, "SOHO Adapter",
            ha="center", va="center", fontsize=8,
            fontweight="bold", color=ACCENT_ORANGE, fontfamily=FONT, zorder=3)

    soho_lines = [
        "EPHIN (1995-2025):",
        "  differential flux x eff. width",
        "",
        "  P8  x 15.0  (10-25 MeV)",
        "  P25 x 15.9  (25-41 MeV)",
        "  P41 x 12.1  (41-53 MeV)",
        "  = >10 MeV proxy",
    ]
    for i, line in enumerate(soho_lines):
        ax.text(x_right - box_w_half / 2 + 0.18, y2 + h2 / 2 - 0.38 - i * 0.14,
                line, ha="left", va="center", fontsize=5.8,
                color=DARK_TEXT, fontfamily=FONT, zorder=3)

    ax.text(x_center, y2 - h2 / 2 - 0.13,
            "Both produce:  >10 MeV flux in pfu   (flat-spectrum partial-bin correction applied)",
            ha="center", va="center", fontsize=5.5, color="#555555",
            fontfamily=FONT, style="italic")

    y2_bot = y2 - h2 / 2 - 0.2
    draw_arrow(ax, x_left, y2_bot, x_center - 0.15, y2_bot - arrow_gap)
    draw_arrow(ax, x_right, y2_bot, x_center + 0.15, y2_bot - arrow_gap)

    # ── Tier 3: Detection Engine ──
    y3 = 6.50
    h3 = 1.6

    draw_box(ax, x_center, y3, box_w, h3,
             "",
             fill=GSU_BLUE, text_color=WHITE,
             fontsize=7, border_color=GSU_BLUE)

    ax.text(x_center, y3 + h3 / 2 - 0.2, "Detection Engine",
            ha="center", va="center", fontsize=10,
            fontweight="bold", color=WHITE, fontfamily=FONT, zorder=3)

    ax.plot([x_center - box_w / 2 + 0.3, x_center + box_w / 2 - 0.3],
            [y3 + h3 / 2 - 0.35, y3 + h3 / 2 - 0.35],
            color="#6699CC", linewidth=0.8, zorder=3)

    rules_x = x_center - box_w / 2 + 0.4
    val_x = x_center - box_w / 2 + 1.15
    rules_top = y3 + h3 / 2 - 0.55

    rules = [
        ("Entry:",    "flux >= 10 pfu  AND  rising gradient"),
        ("",          "(3 of 4 consecutive steps must be positive)"),
        ("Exit:",     "flux < 5 pfu for 2 hours (hysteresis)"),
        ("",          "(quiet period prevents decay-phase fragmentation)"),
        ("Filter:",   "events < 30 min discarded as noise"),
    ]

    for i, (label, desc) in enumerate(rules):
        ry = rules_top - i * 0.21
        if label:
            ax.text(rules_x, ry, label,
                    ha="left", va="center", fontsize=7,
                    fontweight="bold", color=WHITE, fontfamily=FONT, zorder=3)
        ax.text(val_x if label else rules_x + 0.45, ry, desc,
                ha="left", va="center",
                fontsize=6 if not label else 6.5,
                fontweight="normal",
                color="#C8D8EE" if not label else "#E8F0FF",
                fontfamily=FONT, zorder=3)

    draw_arrow(ax, x_center, y3 - h3 / 2, x_center, y3 - h3 / 2 - arrow_gap,
               color=WHITE)
    draw_arrow(ax, x_center, y3 - h3 / 2 - arrow_gap + 0.02,
               x_center, y3 - h3 / 2 - arrow_gap - 0.02)

    # ── Tier 4: Event Extraction ──
    y4 = 5.05
    h4 = 0.5
    draw_box(ax, x_center, y4, box_w, h4,
             "Event Extraction & Merge\n"
             "Contiguous detection mask -> event intervals   |   merge gaps <= 30 min",
             fill=GSU_BLUE_PALE, text_color=DARK_TEXT,
             fontsize=6.5, fontweight="normal", border_color=GSU_BLUE)

    draw_arrow(ax, x_center, y4 - h4 / 2, x_center, y4 - h4 / 2 - arrow_gap)

    # ── Tier 5: Multi-Instrument Fusion ──
    y5 = 4.1
    h5 = 0.5
    draw_box(ax, x_center, y5, box_w, h5,
             "Multi-Instrument Fusion\n"
             "Union of GOES + SOHO events   |   merge overlapping or close (<= 30 min)",
             fill=GSU_BLUE_LIGHT, text_color=WHITE,
             fontsize=6.5, fontweight="normal", border_color=GSU_BLUE)

    y5_bot = y5 - h5 / 2
    draw_arrow(ax, x_center - 0.6, y5_bot, x_left, y5_bot - 0.45)
    draw_arrow(ax, x_center + 0.6, y5_bot, x_right, y5_bot - 0.45)

    # ── Tier 6: Two output boxes ──
    y6 = 2.90
    h6 = 0.75

    draw_box(ax, x_left, y6, box_w_half, h6,
             "Validation\nvs. GSEP Catalog\n(1995 - 2017)",
             fill=GSU_BLUE, text_color=WHITE,
             fontsize=7.5, fontweight="bold")

    draw_box(ax, x_right, y6, box_w_half, h6,
             "Original\nDetections\n(2018 - 2025)",
             fill=ACCENT_ORANGE, text_color=WHITE,
             fontsize=7.5, fontweight="bold")

    ax.text(x_left, y6 - h6 / 2 - 0.15,
            "159 GSEP events  |  event-level TP/FP/FN",
            ha="center", va="center", fontsize=5.5, color="#555555",
            fontfamily=FONT, style="italic")
    ax.text(x_right, y6 - h6 / 2 - 0.15,
            "No external catalog  |  same detection logic",
            ha="center", va="center", fontsize=5.5, color="#555555",
            fontfamily=FONT, style="italic")

    # ── Bottom: Formula reference box ──
    y7 = 1.65
    h7 = 0.75
    draw_box(ax, x_center, y7, box_w, h7,
             "",
             fill="#F8FAFE", text_color=DARK_TEXT,
             fontsize=6, fontweight="normal",
             border_color=GSU_BLUE, border_lw=0.8,
             style="round,pad=0.12", linestyle="--")

    ax.text(x_center, y7 + 0.22, "Flux Derivation Formula",
            ha="center", va="center", fontsize=7,
            fontweight="bold", color=GSU_BLUE, fontfamily=FONT, zorder=3)

    ax.text(x_center, y7 - 0.02,
            r"$J(>\!10\;\mathrm{MeV}) \;=\; \sum_{i}\; I(E_i) \;\times\; \Delta E_i^{\;\mathrm{eff}}$",
            ha="center", va="center", fontsize=10,
            color=DARK_TEXT, zorder=3)

    ax.text(x_center, y7 - 0.27,
            "Channels crossing 10 MeV use effective width via flat-spectrum approximation",
            ha="center", va="center", fontsize=5.5,
            color="#666666", fontfamily=FONT, style="italic", zorder=3)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    path = OUTPUT_DIR / "methodology_flowchart.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white",
                pad_inches=0.15)
    plt.close(fig)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
