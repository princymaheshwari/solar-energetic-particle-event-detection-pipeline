"""
Plot GOES >10 MeV integral proton flux vs time (2018–2025) with fused
detection intervals shaded in GSU blue.

Event intervals are read from per-year CSVs produced by the detection run:
    output/fused events/fused_events_{year}.csv
(columns: start_time, end_time, ...)

Flux is from GOES only (EPEAD through Mar 2020, SGPS from Nov 2020). The
Apr–Oct 2020 GOES gap appears as a break in the curve; fused events in that
period are still shaded (SOHO-only detections).

Usage:
    cd sep_detection_pipeline/
    python -m experiments.plot_flux_detections_2018_2025
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

from sep_core.adapters.goes import GOESAdapter

START_YEAR = 2018
END_YEAR = 2025
GOES_CACHE = PROJECT_ROOT / "data" / "cache" / "goes"
FUSED_EVENTS_DIR = PROJECT_ROOT / "output" / "fused events"
OUTPUT_PATH = PROJECT_ROOT / "output" / "poster_figures" / "flux_detections_2018_2025.png"

GSU_BLUE = "#002A8C"
LINE_COLOR = "#252525"
SHADE_ALPHA = 0.55
DPI = 300


def load_fused_events() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for year in range(START_YEAR, END_YEAR + 1):
        path = FUSED_EVENTS_DIR / f"fused_events_{year}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty or "start_time" not in df.columns:
            continue
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True).dt.tz_localize(None)
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True).dt.tz_localize(None)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No fused event CSVs found under {FUSED_EVENTS_DIR}. "
            "Run experiments.detection.run_detection_2018_2025 or add CSVs."
        )
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("start_time").reset_index(drop=True)


def load_goes_flux_concat() -> tuple[pd.DatetimeIndex, np.ndarray]:
    adapter = GOESAdapter(cache_dir=str(GOES_CACHE))
    times: list[pd.DatetimeIndex] = []
    fluxes: list[np.ndarray] = []
    for year in range(START_YEAR, END_YEAR + 1):
        print(f"  Loading GOES {year}...", flush=True)
        r = adapter.detect_year(year)
        if r.n_timestamps == 0:
            print(f"    (no timestamps for {year})", flush=True)
            continue
        times.append(r.time)
        fluxes.append(r.flux)
        print(f"    {r.n_timestamps:,} points", flush=True)
    if not times:
        raise RuntimeError("No GOES data returned for 2018–2025 (check cache / network).")
    t = pd.DatetimeIndex(np.concatenate([np.asarray(ti, dtype="datetime64[ns]") for ti in times]))
    f = np.concatenate(fluxes)
    order = t.argsort()
    t = t[order]
    f = f[order]
    dup = ~t.duplicated(keep="first")
    return t[dup], f[dup]


def main() -> None:
    print("Loading fused event intervals...", flush=True)
    events = load_fused_events()
    print(f"Loading GOES flux ({START_YEAR}–{END_YEAR}); first run may take several minutes.", flush=True)
    time, flux = load_goes_flux_concat()

    fig, ax = plt.subplots(figsize=(14, 4.5), constrained_layout=True)

    # Shaded detections (behind the line)
    for _, row in events.iterrows():
        s, e = row["start_time"], row["end_time"]
        ax.axvspan(s, e, facecolor=GSU_BLUE, alpha=SHADE_ALPHA, linewidth=0, zorder=1)

    pos = np.isfinite(flux) & (flux > 0)
    ax.plot(
        time[pos],
        flux[pos],
        color=LINE_COLOR,
        linewidth=0.38,
        label="GOES >10 MeV (integral)",
        zorder=3,
        rasterized=True,
    )

    ax.set_yscale("log")
    ax.set_ylabel("Proton flux (pfu)", fontsize=11)
    ax.set_xlabel("Time (UTC)", fontsize=11)
    ax.set_title(
        "GOES >10 MeV proton flux with fused SEP detections (shaded)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    leg = ax.legend(loc="upper right", framealpha=0.9)
    leg.get_frame().set_edgecolor("#CCCCCC")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT_PATH}")
    print(f"  Events shaded: {len(events)}")
    print(f"  GOES points:   {len(time):,}")


if __name__ == "__main__":
    main()
