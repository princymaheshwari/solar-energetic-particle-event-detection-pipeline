"""
Generate square GSU blue/white poster tables:

  1) Event-level validation metrics (1995–2017) for GOES, SOHO, and fused
     streams: precision, recall, F1, false alarm rate (micro-averaged from
     output/full_validation/summary_metrics.csv).

  2) 2018–2025 detection summary: yearly GOES / SOHO / fused event counts,
     plus fusion support breakdown (both instruments vs SOHO-only) and a
     small N/A block for catalog-based error labels (no GSEP past 2017).

Usage:
    cd sep_detection_pipeline/
    python -m experiments.poster_metrics_tables
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

OUTPUT_DIR = PROJECT_ROOT / "output" / "poster_figures"
SUMMARY_CSV = PROJECT_ROOT / "output" / "full_validation" / "summary_metrics.csv"
DETECTIONS_2018_CSV = PROJECT_ROOT / "detections_2018_to_2025.csv"
FUSED_2018_DIR = PROJECT_ROOT / "output" / "fused events"

GSU_BLUE = "#0039A6"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F5F5F5"
BORDER_GRAY = "#CCCCCC"
TEXT_DARK = "#1A1A1A"
FONT = "Arial"
DPI = 300


def micro_metrics_from_summary(df: pd.DataFrame, source: str) -> tuple[float, float, float, float]:
    """Precision, recall, F1, FAR from summed TP/FP/FN and detection count."""
    tp = float(df[f"{source}_TP"].sum())
    fp = float(df[f"{source}_FP"].sum())
    fn = float(df[f"{source}_FN"].sum())
    det_col = f"{source}_events"
    det = float(df[det_col].sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    rec = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)
    else:
        f1 = np.nan
    far = fp / det if det > 0 else np.nan
    return prec, rec, f1, far


def fmt_metric(x: float) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.4f}"


def draw_table(
    ax: plt.Axes,
    headers: list[str],
    rows: list[list[str]],
    *,
    col_width_fracs: list[float] | None = None,
    table_left: float = 0.5,
    table_bottom: float = 0.8,
    table_width: float = 9.0,
    table_height: float = 8.0,
    fontsize_header: int = 11,
    fontsize_cell: int = 10,
) -> None:
    """Draw a GSU-styled table in data coordinates (ax xlim 0–10, ylim 0–10)."""
    n_rows = len(rows) + 1
    n_cols = len(headers)
    row_h = table_height / n_rows
    if col_width_fracs is None:
        col_width_fracs = [1.0] * n_cols
    cw = [c / sum(col_width_fracs) * table_width for c in col_width_fracs]

    top = table_bottom + table_height

    # Header
    x0 = table_left
    for hi, w in zip(headers, cw):
        ax.add_patch(
            Rectangle(
                (x0, top - row_h),
                w,
                row_h,
                facecolor=GSU_BLUE,
                edgecolor=WHITE,
                linewidth=1.2,
                zorder=2,
            )
        )
        ax.text(
            x0 + w / 2,
            top - row_h / 2,
            hi,
            ha="center",
            va="center",
            fontsize=fontsize_header,
            fontweight="bold",
            color=WHITE,
            fontfamily=FONT,
            zorder=3,
        )
        x0 += w

    # Body
    for ri, row in enumerate(rows):
        yb = top - (ri + 2) * row_h
        bg = LIGHT_GRAY if ri % 2 == 0 else WHITE
        x0 = table_left
        for ci, (cell, w) in enumerate(zip(row, cw)):
            ax.add_patch(
                Rectangle(
                    (x0, yb),
                    w,
                    row_h,
                    facecolor=bg,
                    edgecolor=BORDER_GRAY,
                    linewidth=0.75,
                    zorder=1,
                )
            )
            ax.text(
                x0 + w / 2,
                yb + row_h / 2,
                cell,
                ha="center",
                va="center",
                fontsize=fontsize_cell,
                color=TEXT_DARK,
                fontfamily=FONT,
                zorder=2,
            )
            x0 += w


def fig_validation_metrics() -> None:
    df = pd.read_csv(SUMMARY_CSV)
    headers = ["Instrument", "Precision", "Recall", "F1", "False alarm rate"]
    body: list[list[str]] = []
    for label, src in [
        ("GOES", "goes"),
        ("SOHO", "soho"),
        ("Fused (GOES + SOHO)", "fused"),
    ]:
        p, r, f1, far = micro_metrics_from_summary(df, src)
        body.append(
            [
                label,
                fmt_metric(p),
                fmt_metric(r),
                fmt_metric(f1),
                fmt_metric(far),
            ]
        )

    fig, ax = plt.subplots(figsize=(10.0, 10.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(
        5.0,
        9.35,
        "GSEP validation (1995–2017), event-level metrics",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=GSU_BLUE,
        fontfamily=FONT,
    )
    ax.text(
        5.0,
        8.85,
        "Micro-averaged from annual TP/FP/FN; FAR = FP / (all detections)",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
        fontfamily=FONT,
        style="italic",
    )

    draw_table(
        ax,
        headers,
        body,
        col_width_fracs=[2.2, 1.2, 1.2, 1.2, 1.6],
        table_left=0.35,
        table_bottom=1.15,
        table_width=9.3,
        table_height=7.0,
        fontsize_header=10,
        fontsize_cell=10,
    )

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    path = OUTPUT_DIR / "validation_event_metrics_table.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def load_detections_2018_df() -> pd.DataFrame:
    p = DETECTIONS_2018_CSV
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)


def fusion_support_counts() -> tuple[int, int, int, int]:
    """Returns both_inst, soho_only, goes_only, total from fused_events_2018..2025."""
    both = soho_only = goes_only = 0
    for year in range(2018, 2026):
        fp = FUSED_2018_DIR / f"fused_events_{year}.csv"
        if not fp.exists():
            continue
        sub = pd.read_csv(fp)
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            inst = str(row.get("instruments", ""))
            ni = int(row.get("n_instruments", 1))
            total = 1
            if ni >= 2:
                both += 1
            elif "SOHO" in inst.upper() and "GOES" not in inst.upper():
                soho_only += 1
            else:
                goes_only += 1
    total = both + soho_only + goes_only
    return both, soho_only, goes_only, total


def fig_detections_2018_2025() -> None:
    dfc = load_detections_2018_df()
    both, soho_only, goes_only, fused_total = fusion_support_counts()

    headers = ["Year", "GOES detections", "SOHO detections", "Fused detections"]
    body: list[list[str]] = []
    for _, row in dfc.iterrows():
        body.append(
            [
                str(int(row["year"])),
                str(int(row["goes_events"])),
                str(int(row["soho_events"])),
                str(int(row["fused_events"])),
            ]
        )
    body.append(
        [
            "Total",
            str(int(dfc["goes_events"].sum())),
            str(int(dfc["soho_events"].sum())),
            str(int(dfc["fused_events"].sum())),
        ]
    )

    fig, ax = plt.subplots(figsize=(10.0, 10.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_aspect("equal")

    ax.text(
        5.0,
        9.35,
        "Pipeline detections (2018–2025)",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=GSU_BLUE,
        fontfamily=FONT,
    )
    ax.text(
        5.0,
        8.85,
        "No GSEP labels; event-level TP/FP/FN are not defined for this period.",
        ha="center",
        va="center",
        fontsize=8,
        color="#555555",
        fontfamily=FONT,
        style="italic",
    )

    draw_table(
        ax,
        headers,
        body,
        col_width_fracs=[1.0, 1.3, 1.3, 1.4],
        table_left=0.35,
        table_bottom=3.25,
        table_width=9.3,
        table_height=5.6,
        fontsize_header=10,
        fontsize_cell=10,
    )

    # Second mini-table: fusion support + placeholder "error" labels
    sub_headers = ["Category", "Count"]
    sub_rows = [
        ["GOES + SOHO (same fused interval)", str(both)],
        ["SOHO-only fused events", str(soho_only)],
        ["GOES-only fused events", str(goes_only)],
        ["All fused events (sum check)", str(fused_total)],
        ["", ""],
        ["Catalog TP / FP / FN (labeled)", "N/A"],
    ]
    draw_table(
        ax,
        sub_headers,
        sub_rows,
        col_width_fracs=[2.8, 1.0],
        table_left=0.35,
        table_bottom=0.45,
        table_width=9.3,
        table_height=2.55,
        fontsize_header=9,
        fontsize_cell=9,
    )

    ax.text(
        5.0,
        0.22,
        "Fusion support counts are from fused_events_2018..2025 CSVs (instrument tags).",
        ha="center",
        va="center",
        fontsize=7,
        color="#666666",
        fontfamily=FONT,
        style="italic",
    )

    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    path = OUTPUT_DIR / "detections_2018_2025_metrics_table.png"
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_validation_metrics()
    fig_detections_2018_2025()


if __name__ == "__main__":
    main()
