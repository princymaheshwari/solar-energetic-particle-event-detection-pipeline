"""
experiments/detection/run_detection_2018_2025.py

Generate original SEP event detections for 2018–2025.

This script uses the same detection logic validated against the GSEP
catalog in 1995–2017 (see experiments/validation/) to produce new
detections where no external catalog exists for comparison.

Instruments:
    - GOES-15 EPEAD (2018 – Mar 2020): pre-computed >10 MeV integral flux
    - GOES-16 SGPS (Nov 2020 – 2025): derived >10 MeV integral flux
    - SOHO COSTEP/EPHIN (2018 – 2025): >10 MeV proxy from differential channels
    - Fused (GOES + SOHO): union of both instruments' detections

Data gap:
    April–October 2020 has no GOES data (GOES-15 decommissioned, GOES-16
    SGPS avg5m not yet available). SOHO-only detection covers this gap.

Detection logic (identical to validation):
    - Entry threshold: 10 pfu
    - Exit threshold: 5 pfu (hysteresis)
    - Quiet period: 2 hours below exit threshold
    - Rising gradient: 3/4 positive steps
    - Minimum duration: 30 minutes

Outputs:
    - Per-year event CSVs saved to output/detection_2018_2025/
    - Per-year console output with detection counts
    - Summary table at the end

Usage:
    cd sep_detection_pipeline/
    python -m experiments.detection.run_detection_2018_2025
"""

import sys
import time as time_module
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from sep_core.adapters.goes import GOESAdapter
from sep_core.adapters.soho import SOHOAdapter
from sep_core.events import (
    extract_events, merge_close_events, events_to_dataframe,
    events_to_mask, Event,
)
from sep_core.fusion import fuse_events, fused_events_to_dataframe


# ============================================
# CONFIGURATION
# ============================================

START_YEAR = 2018
END_YEAR = 2025

GOES_CACHE = str(PROJECT_ROOT / "data" / "cache" / "goes")
SOHO_CACHE = str(PROJECT_ROOT / "data" / "cache" / "soho")

OUTPUT_DIR = PROJECT_ROOT / "output" / "detection_2018_2025"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# SINGLE-YEAR PIPELINE
# ============================================

def run_year(
    year: int,
    goes_adapter: GOESAdapter,
    soho_adapter: SOHOAdapter,
) -> dict:
    """
    Run detection (no evaluation) for one year.

    Returns a dict with detection counts, event lists, and timing info.
    """

    t0 = time_module.time()

    print(f"\n{'#' * 70}")
    print(f"#  YEAR {year}")
    print(f"{'#' * 70}")

    result = {
        "year": year,
        "goes_n_detected": 0,
        "soho_n_detected": 0,
        "fused_n_detected": 0,
        "elapsed_seconds": 0.0,
    }

    # ----------------------------------------------------------
    # GOES detection
    # ----------------------------------------------------------
    print(f"\n  [GOES] Running detection for {year}...")
    goes_result = goes_adapter.detect_year(year)
    goes_events = []

    if goes_result.n_timestamps > 0:
        goes_events = extract_events(goes_result.time, goes_result.mask)
        goes_events = merge_close_events(goes_events, gap_minutes=30)
        result["goes_n_detected"] = len(goes_events)
        print(f"  [GOES] {goes_result.n_timestamps:,} timestamps, "
              f"{len(goes_events)} events detected")
    else:
        print(f"  [GOES] No data for {year}")

    goes_df = events_to_dataframe(goes_events)
    goes_df.to_csv(OUTPUT_DIR / f"goes_events_{year}.csv", index=False)

    # ----------------------------------------------------------
    # SOHO detection
    # ----------------------------------------------------------
    print(f"\n  [SOHO] Running detection for {year}...")
    soho_result = soho_adapter.detect(year=year)
    soho_events = []

    if soho_result.n_timestamps > 0:
        soho_events = extract_events(soho_result.time, soho_result.mask)
        soho_events = merge_close_events(soho_events, gap_minutes=30)
        result["soho_n_detected"] = len(soho_events)
        print(f"  [SOHO] {soho_result.n_timestamps:,} timestamps, "
              f"{len(soho_events)} events detected")
    else:
        print(f"  [SOHO] No data for {year}")

    soho_df = events_to_dataframe(soho_events)
    soho_df.to_csv(OUTPUT_DIR / f"soho_events_{year}.csv", index=False)

    # ----------------------------------------------------------
    # Fusion (GOES + SOHO)
    # ----------------------------------------------------------
    has_goes = goes_result.n_timestamps > 0
    has_soho = soho_result.n_timestamps > 0

    if has_goes or has_soho:
        print(f"\n  [FUSION] Fusing events...")

        events_by_instrument = {}
        if goes_events:
            events_by_instrument[goes_result.instrument] = goes_events
        if soho_events:
            events_by_instrument[soho_result.instrument] = soho_events

        fused = fuse_events(events_by_instrument, gap_minutes=30)
        result["fused_n_detected"] = len(fused)
        print(f"  [FUSION] {len(fused)} fused events")

        fused_df = fused_events_to_dataframe(fused)
        fused_df.to_csv(
            OUTPUT_DIR / f"fused_events_{year}.csv", index=False
        )

    result["elapsed_seconds"] = time_module.time() - t0
    print(f"\n  Year {year} completed in "
          f"{result['elapsed_seconds']:.1f}s")

    return result


# ============================================
# SUMMARY TABLE
# ============================================

def build_summary_table(all_results: list) -> pd.DataFrame:
    """Build a summary DataFrame from per-year results."""
    rows = []
    for r in all_results:
        rows.append({
            "year": r["year"],
            "goes_events": r["goes_n_detected"],
            "soho_events": r["soho_n_detected"],
            "fused_events": r["fused_n_detected"],
            "elapsed_seconds": r["elapsed_seconds"],
        })
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def print_summary_table(df: pd.DataFrame) -> None:
    """Print detection summary to console."""

    print(f"\n\n{'=' * 80}")
    print(f"  DETECTION SUMMARY — {df['year'].min()} to {df['year'].max()}")
    print(f"{'=' * 80}")

    print(f"\n  {'Year':>4}  {'GOES':>5}  {'SOHO':>5}  {'Fused':>5}  "
          f"{'Time(s)':>7}")
    print(f"  {'-' * 40}")

    for _, row in df.iterrows():
        print(
            f"  {int(row['year']):>4}  "
            f"{int(row['goes_events']):>5}  "
            f"{int(row['soho_events']):>5}  "
            f"{int(row['fused_events']):>5}  "
            f"{row['elapsed_seconds']:>7.1f}"
        )

    print(f"  {'-' * 40}")
    print(
        f"  {'TOTAL':>4}  "
        f"{int(df['goes_events'].sum()):>5}  "
        f"{int(df['soho_events'].sum()):>5}  "
        f"{int(df['fused_events'].sum()):>5}  "
        f"{df['elapsed_seconds'].sum():>7.1f}"
    )

    print(f"\n  Note: No external catalog exists for 2018–2025.")
    print(f"  These are original pipeline detections using the same")
    print(f"  detection logic validated against GSEP in 1995–2017")
    print(f"  (Fused Precision=0.92, Recall=1.00, F1=0.96, FAR=0.09).")
    print(f"\n  Data gap: April–October 2020 (no GOES data).")
    print(f"  SOHO-only detection covers this gap.")
    print(f"{'=' * 80}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print(f"\n{'=' * 70}")
    print(f"  SEP Detection Pipeline — Original Detections {START_YEAR}–{END_YEAR}")
    print(f"  Instruments: GOES (EPEAD/SGPS) + SOHO (COSTEP/EPHIN)")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    goes_adapter = GOESAdapter(cache_dir=GOES_CACHE)
    soho_adapter = SOHOAdapter(cache_dir=SOHO_CACHE)

    all_results = []
    total_start = time_module.time()

    for year in range(START_YEAR, END_YEAR + 1):
        year_result = run_year(year, goes_adapter, soho_adapter)
        all_results.append(year_result)

    total_elapsed = time_module.time() - total_start

    summary_df = build_summary_table(all_results)
    print_summary_table(summary_df)

    summary_path = OUTPUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to: {summary_path}")

    print(f"\n  Total runtime: {total_elapsed / 60:.1f} minutes")
    print(f"\nDone.")
