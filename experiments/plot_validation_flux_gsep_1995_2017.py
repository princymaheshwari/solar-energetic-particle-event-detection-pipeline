"""
Plot GOES >10 MeV flux vs time (1995–2017) as log10(1 + flux [pfu]),
with event-level GSEP validation shading (fused detections):

  - True positive  : GSEP Flag==1 interval matched by a fused detection — blue
  - False positive : fused detection with no catalog overlap — red
  - False negative : GSEP event with no matching fused detection — yellow
  - True negative  : quiet periods — no shading (explained in legend)

Requires:
  - GSEP_List.csv
  - output/full_validation/fused_events_{1995..2017}.csv
    from: python -m experiments.validation.run_full_validation

Usage:
    cd sep_detection_pipeline/
    python -m experiments.plot_validation_flux_gsep_1995_2017
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

from sep_core.adapters.goes import GOESAdapter
from sep_core.events import Event
from sep_core.evaluation.gsep_catalog import load_gsep_catalog
from sep_core.evaluation.matching import match_events_to_catalog

START_YEAR = 1995
END_YEAR = 2017
GSEP_PATH = PROJECT_ROOT / "GSEP_List.csv"
FULL_VAL_DIR = PROJECT_ROOT / "output" / "full_validation"
GOES_CACHE = PROJECT_ROOT / "data" / "cache" / "goes"
OUTPUT_PATH = (
    PROJECT_ROOT / "output" / "poster_figures"
    / "validation_flux_tp_fp_fn_1995_2017.png"
)

# Colors: darker/saturated fills + higher alpha so bands read behind the flux line
COLOR_TP = "#002A8C"
COLOR_FP = "#A00810"
COLOR_FN = "#B8860B"
LINE_COLOR = "#252525"
# Opacity for vertical spans (poster / screen visibility)
ALPHA_SPAN = 0.58
DPI = 300


def log_flux(pfu: np.ndarray) -> np.ndarray:
    """log10(1 + pfu); stable for pfu >= 0."""
    x = np.asarray(pfu, dtype=np.float64)
    x = np.clip(x, 0.0, np.inf)
    return np.log10(1.0 + x)


def load_catalog() -> pd.DataFrame:
    return load_gsep_catalog(
        str(GSEP_PATH),
        start_year=START_YEAR,
        end_year=END_YEAR,
        significant_only=True,
    )


def load_fused_events_from_validation() -> list[Event]:
    events: list[Event] = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = FULL_VAL_DIR / f"fused_events_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        for _, row in df.iterrows():
            st = pd.to_datetime(row["start_time"], utc=True).tz_localize(None)
            et = pd.to_datetime(row["end_time"], utc=True).tz_localize(None)
            dur = float(
                row["duration_minutes"]
                if "duration_minutes" in row
                and pd.notna(row["duration_minutes"])
                else (et - st).total_seconds() / 60.0
            )
            events.append(Event(st, et, dur))
    if not events:
        raise FileNotFoundError(
            f"No fused event CSVs found under {FULL_VAL_DIR}. "
            "Run: python -m experiments.validation.run_full_validation"
        )
    events.sort(key=lambda e: e.start_time)
    return events


def load_goes_flux_concat() -> tuple[pd.DatetimeIndex, np.ndarray]:
    adapter = GOESAdapter(cache_dir=str(GOES_CACHE))
    times: list[pd.DatetimeIndex] = []
    fluxes: list[np.ndarray] = []
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"  Loading GOES {year}...", flush=True)
        r = adapter.detect_year(year)
        if r.n_timestamps == 0:
            print(f"    (no data)", flush=True)
            continue
        times.append(r.time)
        fluxes.append(r.flux)
        print(f"    {r.n_timestamps:,} points", flush=True)
    if not times:
        raise RuntimeError("No GOES data for 1995–2017.")
    t = pd.DatetimeIndex(
        np.concatenate([np.asarray(ti, dtype="datetime64[ns]") for ti in times])
    )
    f = np.concatenate(fluxes)
    order = t.argsort()
    t = t[order]
    f = f[order]
    dup = ~t.duplicated(keep="first")
    return t[dup], f[dup]


def main() -> None:
    print("Loading GSEP catalog (Flag==1)...", flush=True)
    catalog = load_catalog()
    print(f"  {len(catalog)} catalog events", flush=True)

    print("Loading fused events from full_validation...", flush=True)
    fused = load_fused_events_from_validation()
    print(f"  {len(fused)} fused detections", flush=True)

    print("Event-level matching (fused vs GSEP)...", flush=True)
    ev = match_events_to_catalog(fused, catalog)
    tp_n, fp_n, fn_n = ev["TP"], ev["FP"], ev["FN"]
    print(f"  TP={tp_n}  FP={fp_n}  FN={fn_n}", flush=True)

    print("Loading GOES flux (long run on first use)...", flush=True)
    time, flux = load_goes_flux_concat()
    yplot = log_flux(flux)
    ok = np.isfinite(flux) & (flux >= 0) & np.isfinite(yplot)

    fig, ax = plt.subplots(figsize=(14, 4.8), constrained_layout=True)

    # Shading order: FN, FP, then TP on top
    for m in ev["matched_catalog"]:
        if m["detected"]:
            continue
        ax.axvspan(
            m["start_time"],
            m["end_time"],
            facecolor=COLOR_FN,
            alpha=ALPHA_SPAN,
            linewidth=0,
            zorder=1,
        )

    for m in ev["matched_detected"]:
        if m["has_catalog_match"]:
            continue
        ax.axvspan(
            m["start_time"],
            m["end_time"],
            facecolor=COLOR_FP,
            alpha=ALPHA_SPAN,
            linewidth=0,
            zorder=2,
        )

    for m in ev["matched_catalog"]:
        if not m["detected"]:
            continue
        ax.axvspan(
            m["start_time"],
            m["end_time"],
            facecolor=COLOR_TP,
            alpha=ALPHA_SPAN,
            linewidth=0,
            zorder=3,
        )

    ax.plot(
        time[ok],
        yplot[ok],
        color=LINE_COLOR,
        linewidth=0.35,
        zorder=4,
        rasterized=True,
        label="GOES >10 MeV",
    )

    ax.set_ylabel(r"$\log_{10}(1 + \mathrm{flux})$  (flux in pfu)", fontsize=11)
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_title(
        "GSEP validation (1995–2017): fused detections vs Flag==1 catalog",
        fontsize=12,
        fontweight="bold",
    )

    legend_handles = [
        Patch(facecolor=COLOR_TP, alpha=ALPHA_SPAN, edgecolor="#222222", linewidth=0.6,
              label=f"True positive (catalog hit): {tp_n}"),
        Patch(facecolor=COLOR_FP, alpha=ALPHA_SPAN, edgecolor="#222222", linewidth=0.6,
              label=f"False positive (no catalog match): {fp_n}"),
        Patch(facecolor=COLOR_FN, alpha=ALPHA_SPAN, edgecolor="#222222", linewidth=0.6,
              label=f"False negative (missed catalog event): {fn_n}"),
        Patch(facecolor="white", edgecolor="#888888", linewidth=1.0, linestyle="--",
              label="True negative: quiet Sun (not shaded)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.95)

    ax.grid(True, which="both", linestyle=":", alpha=0.45)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
