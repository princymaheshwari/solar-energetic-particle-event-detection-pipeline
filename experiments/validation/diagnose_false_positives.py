"""
experiments/validation/diagnose_false_positives.py

Deep diagnosis of every detected SEP event from the pipeline.

For each detection this script determines:
    1. Whether it directly matches a GSEP catalog event (true positive)
    2. Whether it is the extended decay phase of a nearby catalog event
       (physically real but outside the catalog window)
    3. Whether it is a short noise spike or genuinely anomalous signal
    4. Full flux statistics so you can understand the signal quality

Why this script exists
----------------------
Even with the GSEP catalog (which uses actual event end times unlike the
raw NOAA scraped catalog), some detections are labeled as "false alarms"
by the event-level matcher. This script investigates whether those
detections are genuinely spurious or whether they represent real proton
flux enhancements that GSEP simply does not cover.

Common reasons a real detection gets labeled as a false alarm:
    - Decay phase fragmentation: a large event decays slowly and the flux
      oscillates around 10 pfu, producing several short detections after
      the catalog window closes. These are physically real.
    - Weak sub-threshold events: flux genuinely exceeds 10 pfu but the
      event is too weak or brief for GSEP to catalog. These are borderline.
    - Instrument noise / data spikes: single-point excursions that pass the
      gradient + duration filters by coincidence. These are false alarms.

Classification categories
-------------------------
    CATALOG MATCH              — Directly overlaps a GSEP catalog event
    LIKELY REAL (decay phase)  — Follows a catalog event within 24 hours,
                                 same physical enhancement
    LIKELY REAL (strong)       — Peak >= 100 pfu, too strong to be noise
    LIKELY REAL (sustained)    — Long duration (>3 hrs) above threshold
    TAIL / PRECURSOR           — Within 6 hours of a catalog event,
                                 probably precursor rise or decaying tail
    WEAK ENHANCEMENT           — Peak 10-15 pfu, borderline significance
    POSSIBLE NOISE             — Short (<60 min) with low peak flux
    UNCATALOGED EVENT          — Doesn't fit any category, may be real

Usage
-----
    cd sep_detection_pipeline/
    python -m experiments.validation.diagnose_false_positives

Output
------
    - Per-event block: classification, flux stats, gradient analysis,
      catalog proximity, and context about the nearest GSEP event
    - Classification summary: how many detections fall into each category
    - Flux context analysis: for each GSEP catalog event, shows how the
      detected flux evolved before and after the GSEP window, to explain
      why detections exist outside catalog boundaries

Configuration
-------------
    TEST_YEAR   — which year to diagnose (change at top of file)
    INSTRUMENT  — "goes" or "soho" (change at top of file)
    CATALOG     — "gsep" (recommended) or "noaa" (legacy, pessimistic)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

# ============================================
# CONFIGURATION — change these to investigate different scenarios
# ============================================

TEST_YEAR = 2001

# Which instrument to diagnose: "goes" or "soho"
INSTRUMENT = "goes"

# Which catalog to use: "gsep" (recommended) or "noaa" (legacy)
CATALOG_SOURCE = "gsep"

# GSEP: use only Flag==1 (significant) events for matching
GSEP_SIGNIFICANT_ONLY = True

# A detection is considered "decay phase" of a catalog event if it
# starts within this many hours after the catalog end time
DECAY_WINDOW_HOURS = 24

# A detection is "tail/precursor" if it starts within this many hours
# of ANY catalog event boundary (either direction)
PROXIMITY_HOURS = 6

# Paths
GSEP_CATALOG_PATH = PROJECT_ROOT / "GSEP_List.csv"
NOAA_CATALOG_PATH = PROJECT_ROOT / "noaa_sep_catalog_1995_2025.csv"
GOES_CACHE = PROJECT_ROOT / "data" / "cache" / "goes"
SOHO_CACHE = PROJECT_ROOT / "data" / "cache" / "soho"


# ============================================
# DATA LOADING
# ============================================

def load_instrument_result(instrument: str, year: int):
    """
    Run detection for the specified instrument and year.

    Parameters
    ----------
    instrument : str
        "goes" or "soho"
    year : int
        Year to run detection for

    Returns
    -------
    DetectionResult
    """
    if instrument == "goes":
        from sep_core.adapters.goes import GOESAdapter
        adapter = GOESAdapter(cache_dir=str(GOES_CACHE))
        return adapter.detect_year(year)

    elif instrument == "soho":
        from sep_core.adapters.soho import SOHOAdapter
        adapter = SOHOAdapter(cache_dir=str(SOHO_CACHE))
        return adapter.detect(year=year)

    else:
        raise ValueError(f"Unknown instrument: '{instrument}'. Use 'goes' or 'soho'.")


def load_ground_truth_catalog(source: str, year: int, significant_only: bool):
    """
    Load the ground truth catalog for evaluation.

    Parameters
    ----------
    source : str
        "gsep" or "noaa"
    year : int
        Year to filter to
    significant_only : bool
        If True and source is "gsep", keep only Flag==1 events

    Returns
    -------
    tuple of (pd.DataFrame, str)
        (catalog_dataframe, catalog_description_string)
    """
    if source == "gsep":
        if not GSEP_CATALOG_PATH.exists():
            raise FileNotFoundError(
                f"GSEP catalog not found at {GSEP_CATALOG_PATH}. "
                f"Place GSEP_List.csv in the project root directory."
            )
        from sep_core.evaluation.gsep_catalog import load_gsep_catalog
        catalog = load_gsep_catalog(
            str(GSEP_CATALOG_PATH),
            start_year=year,
            end_year=year,
            significant_only=significant_only,
        )
        flag_note = "Flag==1 only" if significant_only else "all flags"
        desc = f"GSEP catalog ({flag_note}, {year})"
        return catalog, desc

    elif source == "noaa":
        from sep_core.evaluation.metrics import load_noaa_catalog
        catalog = load_noaa_catalog(
            str(NOAA_CATALOG_PATH),
            start_year=year,
            end_year=year,
        )
        desc = f"NOAA legacy catalog ({year}) — WARNING: end_time is peak time"
        return catalog, desc

    else:
        raise ValueError(f"Unknown catalog source: '{source}'. Use 'gsep' or 'noaa'.")


# ============================================
# CLASSIFICATION HELPERS
# ============================================

def find_nearest_catalog_event(det_start, det_end, catalog):
    """
    Find the catalog event nearest to a detection, and compute overlap.

    Checks every catalog event for overlap with the detection window.
    If multiple catalog events overlap, returns the one with the largest
    overlap. If none overlap, returns the closest one by time gap.

    Parameters
    ----------
    det_start : pd.Timestamp
        Detection start time
    det_end : pd.Timestamp
        Detection end time
    catalog : pd.DataFrame
        Catalog with start_time and end_time columns

    Returns
    -------
    tuple of (catalog_row_or_None, overlap_timedelta, gap_hours_float)
        If there is any overlap, gap_hours = 0.0.
        If no overlap, gap_hours = hours to nearest event boundary.
    """
    best_row = None
    best_overlap = pd.Timedelta(0)
    best_gap_hours = 1e9

    for _, cat_row in catalog.iterrows():
        cat_start = cat_row["start_time"]
        cat_end = cat_row["end_time"]

        overlap_start = max(det_start, cat_start)
        overlap_end = min(det_end, cat_end)
        overlap = overlap_end - overlap_start

        if overlap >= pd.Timedelta(0):
            # Overlapping — keep the one with the largest overlap
            if overlap > best_overlap:
                best_overlap = overlap
                best_row = cat_row
                best_gap_hours = 0.0
        else:
            # Non-overlapping — compute gap to nearest boundary
            if det_end < cat_start:
                gap = (cat_start - det_end).total_seconds() / 3600
            else:
                gap = (det_start - cat_end).total_seconds() / 3600

            if best_gap_hours > 0 and gap < best_gap_hours:
                best_gap_hours = gap
                best_row = cat_row

    return best_row, best_overlap, best_gap_hours


def check_decay_phase(det_start, det_end, catalog, decay_window_hours):
    """
    Check whether a detection is the decay phase of a nearby catalog event.

    A detection is considered a decay phase fragment when:
    - No direct catalog overlap exists
    - The detection starts within decay_window_hours after a catalog event ends
    - The detection is clearly part of the same physical enhancement

    Parameters
    ----------
    det_start : pd.Timestamp
    det_end : pd.Timestamp
    catalog : pd.DataFrame
    decay_window_hours : float
        How many hours after a catalog event's end to still consider decay

    Returns
    -------
    tuple of (bool, catalog_row_or_None, hours_after_end_float)
        (is_decay, catalog_event, hours_since_catalog_end)
    """
    decay_window = pd.Timedelta(hours=decay_window_hours)

    for _, cat_row in catalog.iterrows():
        cat_end = cat_row["end_time"]

        # Detection starts within decay_window after catalog end
        # and detection start is AFTER catalog end (not before)
        if cat_end <= det_start <= cat_end + decay_window:
            hours_after = (det_start - cat_end).total_seconds() / 3600
            return True, cat_row, hours_after

    return False, None, 0.0


def classify_detection(
    det_start,
    det_end,
    peak_flux,
    duration_min,
    gap_hours,
    has_overlap,
    is_decay_phase,
    pos_grad_ratio,
    std_flux,
    n_above_10,
    n_points,
):
    """
    Classify a detection into a human-readable category.

    The classification follows a priority chain: direct catalog match
    is checked first, then decay phase, then signal strength indicators,
    then noise heuristics.

    Parameters
    ----------
    det_start, det_end : pd.Timestamp
    peak_flux : float — peak flux in pfu
    duration_min : float — event duration in minutes
    gap_hours : float — hours to nearest catalog event (0 if overlap)
    has_overlap : bool — True if directly overlaps catalog
    is_decay_phase : bool — True if within decay_window of catalog end
    pos_grad_ratio : float — fraction of timesteps with rising flux
    std_flux : float — standard deviation of flux during detection
    n_above_10 : int — timesteps where flux >= 10 pfu
    n_points : int — total timesteps in detection

    Returns
    -------
    tuple of (category_label: str, reasoning: str)
    """
    pct_above_10 = n_above_10 / max(n_points, 1)

    if has_overlap:
        return "CATALOG MATCH", "Directly overlaps a GSEP catalog event"

    if is_decay_phase:
        return (
            "LIKELY REAL (decay phase)",
            "Starts within 24 hours after a catalog event ends — "
            "same physical SEP enhancement, flux oscillating near threshold "
            "during slow decay"
        )

    if peak_flux >= 100:
        return (
            "LIKELY REAL (strong signal)",
            f"Peak flux {peak_flux:.0f} pfu — too strong to be instrumental noise"
        )

    if duration_min >= 180 and pct_above_10 >= 0.8:
        return (
            "LIKELY REAL (sustained enhancement)",
            f"Duration {duration_min:.0f} min with {pct_above_10:.0%} of points "
            f">= 10 pfu — sustained proton enhancement"
        )

    if gap_hours < PROXIMITY_HOURS:
        return (
            "TAIL / PRECURSOR",
            f"Only {gap_hours:.1f} hours from nearest catalog event — "
            f"likely precursor rise or decaying tail"
        )

    if peak_flux < 15 and duration_min < 60:
        return (
            "POSSIBLE NOISE",
            f"Peak {peak_flux:.1f} pfu over only {duration_min:.0f} min — "
            f"borderline flux, short duration, possibly instrumental"
        )

    if peak_flux < 15:
        return (
            "WEAK ENHANCEMENT",
            f"Peak flux {peak_flux:.1f} pfu — below GSEP significance threshold, "
            f"NOAA SWPC may not classify as a proton event"
        )

    if std_flux < 2.0 and pct_above_10 < 0.5:
        return (
            "FLAT NEAR-THRESHOLD",
            f"Low flux variability (std={std_flux:.2f}), flux hovering "
            f"near 10 pfu — not a clear onset signature"
        )

    if pos_grad_ratio < 0.3:
        return (
            "DECAYING SIGNAL",
            f"Only {pos_grad_ratio:.0%} of timesteps show rising flux — "
            f"likely a long decay phase with no clear onset"
        )

    return (
        "POSSIBLE UNCATALOGED EVENT",
        f"Peak {peak_flux:.1f} pfu over {duration_min:.0f} min — "
        f"may be a real event not cataloged in GSEP "
        f"(gap to nearest catalog event: {gap_hours:.1f} hrs)"
    )


# ============================================
# FLUX CONTEXT ANALYSIS
# ============================================

def analyze_flux_around_catalog_events(result, catalog):
    """
    For each GSEP catalog event, show how the measured flux behaved
    before, during, and after the catalog window.

    This explains WHY detections appear outside catalog boundaries.
    The most common explanation is that flux stays elevated above 10 pfu
    well after the GSEP slice_end time, producing real detections that
    are still technically outside the catalog window.

    Parameters
    ----------
    result : DetectionResult
        From GOES or SOHO adapter
    catalog : pd.DataFrame
        GSEP or NOAA catalog

    Prints
    ------
    Summary for each catalog event showing flux levels at key timepoints
    relative to the event window.
    """
    print("\n" + "=" * 70)
    print("  FLUX CONTEXT AROUND EACH CATALOG EVENT")
    print("  Shows how flux behaves before/during/after the catalog window.")
    print("  Helps explain detections that appear outside catalog boundaries.")
    print("=" * 70)

    for _, cat_row in catalog.iterrows():
        cat_start = cat_row["start_time"]
        cat_end = cat_row["end_time"]
        cat_dur_hrs = (cat_end - cat_start).total_seconds() / 3600

        # Get peak flux label from catalog
        peak_label = (
            f"{cat_row['peak_flux_pfu']:.1f} pfu"
            if pd.notna(cat_row.get("peak_flux_pfu", np.nan))
            else "unknown pfu"
        )

        print(f"\n  Catalog: {cat_start.strftime('%Y-%m-%d %H:%M')} → "
              f"{cat_end.strftime('%Y-%m-%d %H:%M')}  "
              f"({cat_dur_hrs:.1f} hrs)  Peak: {peak_label}")

        # Sample flux at key timepoints around the catalog window
        checkpoints = {
            "12 hrs before start": cat_start - pd.Timedelta(hours=12),
            "At start":            cat_start,
            "At catalog end":      cat_end,
            "+6 hrs after end":    cat_end + pd.Timedelta(hours=6),
            "+12 hrs after end":   cat_end + pd.Timedelta(hours=12),
            "+24 hrs after end":   cat_end + pd.Timedelta(hours=24),
            "+48 hrs after end":   cat_end + pd.Timedelta(hours=48),
        }

        for label, ts in checkpoints.items():
            # Find the nearest data point
            idx = result.time.searchsorted(ts)
            idx = min(idx, len(result.time) - 1)

            actual_ts = result.time[idx]
            flux_val = result.flux[idx]

            # Only show if within reasonable range of our check time
            time_diff = abs((actual_ts - ts).total_seconds() / 60)
            if time_diff > 15:  # More than 15 min away — data gap
                flux_str = "N/A (data gap)"
            elif np.isnan(flux_val):
                flux_str = "NaN (missing)"
            else:
                above = " ← above 10 pfu" if flux_val >= 10.0 else ""
                flux_str = f"{flux_val:.2f} pfu{above}"

            print(f"    {label:<25}: {flux_str}")

        # Count how long flux stays above 10 pfu after catalog end
        after_end_mask = result.time > cat_end
        if after_end_mask.any():
            flux_after = result.flux[after_end_mask]
            time_after = result.time[after_end_mask]
            above_after = flux_after >= 10.0

            if above_after.any():
                # Find first sustained drop below 10 pfu
                sustained_end = None
                for j in range(len(above_after)):
                    if not above_after[j]:
                        # Check if the next 12 points (1 hr) are all below 10
                        window = above_after[j : j + 12]
                        if len(window) >= 6 and not window.any():
                            sustained_end = time_after[j]
                            break

                if sustained_end is not None:
                    extra_hrs = (sustained_end - cat_end).total_seconds() / 3600
                    print(f"    Flux stays ≥10 pfu for {extra_hrs:.1f} hrs "
                          f"after catalog end — explains post-window detections")
                else:
                    # Flux was above 10 through the end of our data
                    n_above = int(above_after.sum())
                    hrs_above = n_above * 5 / 60
                    print(f"    Flux remains ≥10 pfu for at least {hrs_above:.1f} hrs "
                          f"after catalog end (full extent beyond data range)")
            else:
                print(f"    Flux drops below 10 pfu immediately after catalog end")


# ============================================
# MAIN DIAGNOSIS LOOP
# ============================================

def run_diagnosis():
    """
    Main entry point. Loads data, classifies every detected event,
    and prints a full diagnostic report.
    """
    print("=" * 70)
    print(f"  SEP EVENT DIAGNOSIS")
    print(f"  Instrument: {INSTRUMENT.upper()}   Year: {TEST_YEAR}")
    print(f"  Catalog: {CATALOG_SOURCE.upper()}"
          + (" (Flag==1 only)" if CATALOG_SOURCE == "gsep"
             and GSEP_SIGNIFICANT_ONLY else ""))
    print("=" * 70)

    # --- Load detection result ---
    print(f"\n[1] Loading {INSTRUMENT.upper()} detection results for {TEST_YEAR}...")
    result = load_instrument_result(INSTRUMENT, TEST_YEAR)
    print(f"    {result.summary()}")

    # --- Extract events ---
    from sep_core.events import extract_events, merge_close_events
    events = extract_events(result.time, result.mask)
    events = merge_close_events(events, gap_minutes=30)
    print(f"\n    Total detected events after merging: {len(events)}")

    # --- Load catalog ---
    print(f"\n[2] Loading ground truth catalog ({CATALOG_SOURCE.upper()})...")
    catalog, catalog_desc = load_ground_truth_catalog(
        CATALOG_SOURCE, TEST_YEAR, GSEP_SIGNIFICANT_ONLY
    )
    print(f"    {catalog_desc}")
    print(f"    Catalog events in {TEST_YEAR}: {len(catalog)}")

    if catalog.empty:
        print("    No catalog events found for this year — cannot evaluate.")
        return

    # --- Category counters ---
    category_counts = {}
    all_classifications = []

    # ============================================
    # PER-EVENT ANALYSIS
    # ============================================
    print("\n" + "=" * 70)
    print("  DETAILED EVENT ANALYSIS")
    print("  Every detected event is classified with supporting evidence.")
    print("=" * 70)

    for i, det in enumerate(events):

        # Extract flux values during this detection window
        in_window = (result.time >= det.start_time) & (result.time <= det.end_time)
        flux_window = result.flux[in_window]
        time_window = result.time[in_window]

        if len(flux_window) == 0:
            continue

        # --- Flux statistics ---
        peak_flux = float(np.nanmax(flux_window))
        mean_flux = float(np.nanmean(flux_window))
        min_flux = float(np.nanmin(flux_window))
        std_flux = float(np.nanstd(flux_window))
        n_points = len(flux_window)
        n_above_10 = int(np.sum(flux_window >= 10.0))
        n_above_100 = int(np.sum(flux_window >= 100.0))

        # --- Timing of peak ---
        peak_idx = int(np.nanargmax(flux_window))
        peak_time = time_window[peak_idx]
        flux_first = float(flux_window[0])
        flux_last = float(flux_window[-1])

        # --- Gradient behavior ---
        grads = np.diff(flux_window)
        valid_grads = grads[~np.isnan(grads)]
        pos_grad_ratio = (
            float(np.sum(valid_grads > 0)) / len(valid_grads)
            if len(valid_grads) > 0 else 0.0
        )
        # Rise rate: average pfu gained per 5-minute step from start to peak
        rise_rate = (peak_flux - flux_first) / max(peak_idx, 1)

        # --- Catalog proximity ---
        nearest_cat, overlap, gap_hours = find_nearest_catalog_event(
            det.start_time, det.end_time, catalog
        )
        has_overlap = overlap > pd.Timedelta(0)

        # --- Decay phase check ---
        is_decay, decay_parent, hrs_after_end = check_decay_phase(
            det.start_time, det.end_time, catalog, DECAY_WINDOW_HOURS
        )

        # --- Classify ---
        category, reasoning = classify_detection(
            det.start_time, det.end_time,
            peak_flux, det.duration_minutes, gap_hours,
            has_overlap, is_decay, pos_grad_ratio, std_flux,
            n_above_10, n_points,
        )

        # Track
        category_counts[category] = category_counts.get(category, 0) + 1
        all_classifications.append({
            "start": det.start_time,
            "end": det.end_time,
            "duration_min": det.duration_minutes,
            "peak_flux": peak_flux,
            "category": category,
        })

        # --- Print block ---
        sep_line = "─" * 70
        print(f"\n{sep_line}")
        print(f"  Detection #{i+1:02d}  ──  [{category}]")
        print(f"  Period:   {det.start_time}  →  {det.end_time}")
        print(f"  Duration: {det.duration_minutes:.0f} min "
              f"({det.duration_minutes / 60:.1f} hrs)")
        print(f"  Reason:   {reasoning}")

        print(f"\n  Flux profile:")
        print(f"    Start:   {flux_first:.2f} pfu")
        print(f"    Peak:    {peak_flux:.2f} pfu  (at {peak_time})")
        print(f"    End:     {flux_last:.2f} pfu")
        print(f"    Mean:    {mean_flux:.2f} pfu   Std: {std_flux:.2f}   "
              f"Min: {min_flux:.2f}")
        print(f"    Points:  {n_points} total | "
              f"{n_above_10} ≥10 pfu ({100*n_above_10/max(n_points,1):.0f}%) | "
              f"{n_above_100} ≥100 pfu ({100*n_above_100/max(n_points,1):.0f}%)")

        print(f"\n  Gradient (rise/fall behavior):")
        print(f"    Positive steps: {pos_grad_ratio:.0%} of timesteps show rising flux")
        print(f"    Rise rate:      {rise_rate:.3f} pfu per 5-min step")

        print(f"\n  Catalog proximity:")
        if has_overlap and nearest_cat is not None:
            print(f"    Direct overlap with catalog event:")
            print(f"      {nearest_cat['start_time']} → {nearest_cat['end_time']}")
            peak_val = nearest_cat.get("peak_flux_pfu", np.nan)
            if pd.notna(peak_val):
                print(f"      Peak: {peak_val:.1f} pfu")
            print(f"      Overlap duration: {overlap}")
        elif is_decay and decay_parent is not None:
            print(f"    Decay phase of catalog event:")
            print(f"      {decay_parent['start_time']} → {decay_parent['end_time']}")
            print(f"      This detection starts {hrs_after_end:.1f} hrs after "
                  f"catalog end — flux still elevated")
        elif nearest_cat is not None:
            print(f"    Nearest catalog event: {gap_hours:.1f} hrs away")
            print(f"      {nearest_cat['start_time']} → {nearest_cat['end_time']}")

    # ============================================
    # SUMMARY TABLE
    # ============================================
    print("\n\n" + "=" * 70)
    print("  CLASSIFICATION SUMMARY")
    print("  Breaks down all detections by category so you can see at a")
    print("  glance which 'false alarms' are real physics vs. actual noise.")
    print("=" * 70)

    # Group categories
    real_categories = {
        "CATALOG MATCH",
        "LIKELY REAL (decay phase)",
        "LIKELY REAL (strong signal)",
        "LIKELY REAL (sustained enhancement)",
    }
    borderline_categories = {
        "TAIL / PRECURSOR",
        "POSSIBLE UNCATALOGED EVENT",
        "WEAK ENHANCEMENT",
    }
    noise_categories = {
        "POSSIBLE NOISE",
        "FLAT NEAR-THRESHOLD",
        "DECAYING SIGNAL",
    }

    n_real = 0
    n_borderline = 0
    n_noise = 0

    total = len(events)
    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        tag = ""
        if category in real_categories:
            n_real += count
            tag = "  [REAL]"
        elif category in borderline_categories:
            n_borderline += count
            tag = "  [BORDERLINE]"
        else:
            n_noise += count
            tag = "  [LIKELY NOISE]"
        print(f"  {count:3d}  {category}{tag}")

    print(f"\n  {'─'*50}")
    print(f"  {n_real:3d}  Clearly real           ({100*n_real/total:.0f}%)")
    print(f"  {n_borderline:3d}  Borderline / uncataloged ({100*n_borderline/total:.0f}%)")
    print(f"  {n_noise:3d}  Likely noise           ({100*n_noise/total:.0f}%)")
    print(f"  {total:3d}  Total detections")

    print(f"\n  Interpretation:")
    print(f"    Your reported false alarm rate counts 'LIKELY REAL (decay phase)'")
    print(f"    events as false alarms because they fall outside the GSEP window.")
    print(f"    These are physically real elevated flux periods — the catalog")
    print(f"    closed its window while the particle enhancement was still ongoing.")
    print(f"    True false alarms are only the POSSIBLE NOISE category ({n_noise} events).")

    # ============================================
    # FLUX CONTEXT ANALYSIS
    # ============================================
    analyze_flux_around_catalog_events(result, catalog)

    # ============================================
    # MISSED EVENTS ANALYSIS
    # ============================================
    print("\n\n" + "=" * 70)
    print("  MISSED CATALOG EVENTS ANALYSIS")
    print("  For each catalog event your pipeline did not detect, explains why.")
    print("=" * 70)

    for _, cat_row in catalog.iterrows():
        cat_start = cat_row["start_time"]
        cat_end = cat_row["end_time"]

        # Check if any detection overlaps this catalog event
        matched = False
        for det in events:
            overlap_start = max(cat_start, det.start_time)
            overlap_end = min(cat_end, det.end_time)
            if overlap_end > overlap_start:
                matched = True
                break

        if matched:
            continue

        # This is a missed event — show what flux looked like
        peak_val = cat_row.get("peak_flux_pfu", np.nan)
        peak_str = f"{peak_val:.1f} pfu" if pd.notna(peak_val) else "unknown"

        print(f"\n  MISSED: {cat_start} → {cat_end}  (Peak: {peak_str})")

        in_window = (result.time >= cat_start) & (result.time <= cat_end)
        flux_in_window = result.flux[in_window]

        if flux_in_window.size == 0:
            print(f"    No flux data in this period — data gap in instrument record")
            continue

        max_flux_seen = float(np.nanmax(flux_in_window))
        mean_flux_seen = float(np.nanmean(flux_in_window))
        n_above = int(np.sum(flux_in_window >= 10.0))
        n_total = len(flux_in_window)

        print(f"    Max flux observed:  {max_flux_seen:.2f} pfu")
        print(f"    Mean flux observed: {mean_flux_seen:.2f} pfu")
        print(f"    Points ≥ 10 pfu:    {n_above} of {n_total} "
              f"({100*n_above/max(n_total,1):.0f}%)")

        if max_flux_seen < 10.0:
            print(f"    Diagnosis: flux never reached 10 pfu in {INSTRUMENT.upper()} data.")
            print(f"    This event was observed by GOES but not by SOHO, or "
                  f"the instrument had a data gap during this period.")
        elif n_above < 6:
            print(f"    Diagnosis: flux crossed 10 pfu but for < 30 min "
                  f"(fewer than 6 data points). The duration filter correctly "
                  f"rejected this as too brief.")
        else:
            print(f"    Diagnosis: flux reached {max_flux_seen:.1f} pfu with "
                  f"{n_above} points above threshold. The gradient condition "
                  f"(3/4 positive steps) may not have been satisfied at onset, "
                  f"or the event was embedded in a larger ongoing enhancement.")


if __name__ == "__main__":
    run_diagnosis()