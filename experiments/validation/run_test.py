"""
experiments/validation/run_test.py

End-to-end test of the SEP detection pipeline for a single year.

Ground truth: GSEP catalog (Papaioannou et al.)
  - Uses actual event end times (slice_end), NOT peak times
  - Validates against Flag==1 events (245 significant events)
  - Year range in catalog: ~1986–2017

This script is for validation (1995–2017) where GSEP ground truth exists.
For original detections (2018–2025), see experiments/detection/.

Usage:
    cd sep_detection_pipeline/
    python -m experiments.validation.run_test
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# ============================================
# CONFIGURATION
# ============================================

# Test year — pick any year in 1995-2017 for validated results
# Good active years: 2001 (14 events), 2003 (10 events), 2000 (15 events)
TEST_YEAR = 2001

# GSEP catalog path — place this file in your project root
GSEP_CATALOG_PATH = PROJECT_ROOT / "GSEP_List.csv"

# Legacy NOAA catalog (kept for comparison)
NOAA_CATALOG_PATH = PROJECT_ROOT / "noaa_sep_catalog_1995_2025.csv"

# Cache directories
GOES_CACHE = PROJECT_ROOT / "data" / "cache" / "goes"
SOHO_CACHE = PROJECT_ROOT / "data" / "cache" / "soho"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# HELPERS
# ============================================

def load_ground_truth(year: int):
    """
    Load the GSEP catalog filtered to a single year.
    Falls back to NOAA catalog if GSEP is unavailable.
    Returns (catalog_df, catalog_name) tuple.
    """
    if GSEP_CATALOG_PATH.exists():
        from sep_core.evaluation.gsep_catalog import (
            load_gsep_catalog, gsep_catalog_summary
        )
        catalog = load_gsep_catalog(
            str(GSEP_CATALOG_PATH),
            start_year=year,
            end_year=year,
            significant_only=True   # Flag==1 only
        )
        name = f"GSEP (Flag==1, {year})"
        print(f"\n    {gsep_catalog_summary(catalog)}")
        return catalog, name

    elif NOAA_CATALOG_PATH.exists():
        from sep_core.evaluation.metrics import load_noaa_catalog
        catalog = load_noaa_catalog(
            str(NOAA_CATALOG_PATH),
            start_year=year,
            end_year=year
        )
        name = f"NOAA legacy (WARNING: end_time=peak_time, {year})"
        print(f"\n    WARNING: Using NOAA catalog — end_time is peak time,")
        print(f"    not actual end. Precision will be artificially low.")
        print(f"    Place GSEP_List.csv in the project root for better results.")
        return catalog, name

    else:
        print(f"\n    No catalog found. Skipping evaluation.")
        return pd.DataFrame(), "no catalog"


# ============================================
# GOES TEST
# ============================================

def run_goes_test():
    """Run GOES detection and evaluation for TEST_YEAR."""

    from sep_core.adapters.goes import GOESAdapter
    from sep_core.events import extract_events, merge_close_events, events_to_dataframe
    from sep_core.evaluation.metrics import evaluate_detection, print_metrics
    from sep_core.evaluation.matching import match_events_to_catalog, print_event_metrics

    print("=" * 60)
    print(f"  GOES TEST — Year {TEST_YEAR}")
    print("=" * 60)

    # Step 1: Detect
    print(f"\n[1] Running GOES detection for {TEST_YEAR}...")
    adapter = GOESAdapter(cache_dir=str(GOES_CACHE))
    result = adapter.detect_year(TEST_YEAR)
    print(f"\n{result.summary()}")

    # Step 2: Extract events
    print(f"\n[2] Extracting event intervals...")
    events = extract_events(result.time, result.mask)
    events = merge_close_events(events, gap_minutes=30)
    events_df = events_to_dataframe(events)
    print(f"    Detected {len(events)} events after merging")

    if not events_df.empty:
        for _, row in events_df.iterrows():
            print(f"      {row['start_time']} — {row['end_time']}  "
                  f"({row['duration_minutes']:.0f} min)")

    # Step 3: Load ground truth and evaluate
    print(f"\n[3] Loading ground truth catalog...")
    catalog, catalog_name = load_ground_truth(TEST_YEAR)

    if not catalog.empty:
        print(f"\n[4] Pointwise evaluation...")
        pw = evaluate_detection(result.mask, result.time, catalog)
        print_metrics(pw, title=f"GOES {TEST_YEAR} — Pointwise",
                      catalog_note=catalog_name)

        print(f"\n[5] Event-level evaluation...")
        ev = match_events_to_catalog(events, catalog)
        print_event_metrics(ev, title=f"GOES {TEST_YEAR} — Event Level")

    # Step 4: Save
    events_df.to_csv(OUTPUT_DIR / f"goes_events_{TEST_YEAR}.csv", index=False)
    print(f"\n    Saved to output/goes_events_{TEST_YEAR}.csv")

    return result, events


# ============================================
# SOHO TEST
# ============================================

def run_soho_test():
    """Run SOHO detection and evaluation for TEST_YEAR."""

    from sep_core.adapters.soho import SOHOAdapter
    from sep_core.events import extract_events, merge_close_events, events_to_dataframe
    from sep_core.evaluation.metrics import evaluate_detection, print_metrics
    from sep_core.evaluation.matching import match_events_to_catalog, print_event_metrics

    print("\n\n" + "=" * 60)
    print(f"  SOHO TEST — Year {TEST_YEAR}")
    print("=" * 60)

    print(f"\n[1] Running SOHO detection for {TEST_YEAR}...")
    adapter = SOHOAdapter(cache_dir=str(SOHO_CACHE))
    result = adapter.detect(year=TEST_YEAR)
    print(f"\n{result.summary()}")

    print(f"\n[2] Extracting event intervals...")
    events = extract_events(result.time, result.mask)
    events = merge_close_events(events, gap_minutes=30)
    events_df = events_to_dataframe(events)
    print(f"    Detected {len(events)} events after merging")

    if not events_df.empty:
        for _, row in events_df.iterrows():
            print(f"      {row['start_time']} — {row['end_time']}  "
                  f"({row['duration_minutes']:.0f} min)")

    print(f"\n[3] Loading ground truth catalog...")
    catalog, catalog_name = load_ground_truth(TEST_YEAR)

    if not catalog.empty:
        print(f"\n[4] Pointwise evaluation...")
        pw = evaluate_detection(result.mask, result.time, catalog)
        print_metrics(pw, title=f"SOHO {TEST_YEAR} — Pointwise",
                      catalog_note=catalog_name)

        print(f"\n[5] Event-level evaluation...")
        ev = match_events_to_catalog(events, catalog)
        print_event_metrics(ev, title=f"SOHO {TEST_YEAR} — Event Level")

    events_df.to_csv(OUTPUT_DIR / f"soho_events_{TEST_YEAR}.csv", index=False)
    print(f"\n    Saved to output/soho_events_{TEST_YEAR}.csv")

    return result, events


# ============================================
# FUSION TEST
# ============================================

def run_fusion_test(goes_result, goes_events, soho_result, soho_events):
    """Fuse GOES + SOHO and evaluate."""

    from sep_core.fusion import fuse_events, fused_events_to_dataframe, compute_support_labels
    from sep_core.events import events_to_mask, Event
    from sep_core.evaluation.metrics import evaluate_detection, print_metrics
    from sep_core.evaluation.matching import match_events_to_catalog, print_event_metrics

    print("\n\n" + "=" * 60)
    print(f"  FUSION TEST — GOES + SOHO — Year {TEST_YEAR}")
    print("=" * 60)

    print(f"\n[1] Fusing events...")
    fused = fuse_events({
        goes_result.instrument: goes_events,
        soho_result.instrument: soho_events,
    }, gap_minutes=30)
    fused_df = fused_events_to_dataframe(fused)

    print(f"    Fused events: {len(fused)}")
    if not fused_df.empty:
        for _, row in fused_df.iterrows():
            print(f"      {row['start_time']} — {row['end_time']}  "
                  f"({row['duration_minutes']:.0f} min)  [{row['instruments']}]")

    # Build common time axis
    if goes_result.n_timestamps == 0 or soho_result.n_timestamps == 0:
        print("\n    One instrument has no data — skipping fusion eval.")
        return

    common_start = max(goes_result.time[0], soho_result.time[0])
    common_end = min(goes_result.time[-1], soho_result.time[-1])
    common_time = pd.date_range(common_start, common_end, freq="5min")
    print(f"\n    Common time range: {common_start} to {common_end}")

    fused_event_list = [
        Event(fe.start_time, fe.end_time, fe.duration_minutes)
        for fe in fused
    ]
    fused_mask = events_to_mask(fused_event_list, common_time)

    print(f"\n[2] Loading ground truth catalog...")
    catalog, catalog_name = load_ground_truth(TEST_YEAR)

    if not catalog.empty:
        print(f"\n[3] Evaluating fused detection...")
        pw = evaluate_detection(fused_mask, common_time, catalog)
        print_metrics(pw, title=f"FUSED {TEST_YEAR} — Pointwise",
                      catalog_note=catalog_name)

        ev = match_events_to_catalog(fused_event_list, catalog)
        print_event_metrics(ev, title=f"FUSED {TEST_YEAR} — Event Level")

    # Support labels
    support = compute_support_labels(
        fused, common_time,
        [goes_result.instrument, soho_result.instrument]
    )
    support_counts = support.value_counts()
    print(f"\n[4] Instrument support distribution:")
    for label, count in support_counts.items():
        print(f"    {label}: {count:,} timestamps")

    fused_df.to_csv(OUTPUT_DIR / f"fused_events_{TEST_YEAR}.csv", index=False)
    print(f"\n    Saved to output/fused_events_{TEST_YEAR}.csv")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print(f"\nSEP Detection Pipeline — End-to-End Test")
    print(f"Test Year:    {TEST_YEAR}")
    print(f"Ground truth: GSEP catalog (Flag==1 significant events)")
    print(f"Catalog path: {GSEP_CATALOG_PATH}")
    print()

    # Check that at least one catalog exists
    if not GSEP_CATALOG_PATH.exists() and not NOAA_CATALOG_PATH.exists():
        print(f"ERROR: No catalog found.")
        print(f"  Place GSEP_List.csv in: {PROJECT_ROOT}")
        sys.exit(1)

    if not GSEP_CATALOG_PATH.exists():
        print(f"NOTE: GSEP_List.csv not found at {GSEP_CATALOG_PATH}")
        print(f"      Falling back to legacy NOAA catalog.")
        print(f"      Precision metrics will be pessimistic.")
        print()

    goes_result, goes_events = run_goes_test()
    soho_result, soho_events = run_soho_test()

    if goes_result.has_detections or soho_result.has_detections:
        run_fusion_test(goes_result, goes_events, soho_result, soho_events)
    else:
        print(f"\nNo detections from either instrument in {TEST_YEAR}.")
        print(f"Try an active year: 2000, 2001, 2003, 2005.")

    print("\n\nDone.")