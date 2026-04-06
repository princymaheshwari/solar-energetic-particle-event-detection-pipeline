"""
experiments/validation/run_full_validation.py

Full multi-year validation of the SEP detection pipeline across 1995–2017.

Runs GOES detection, SOHO detection, and GOES+SOHO fusion for every year
from 1995 through 2017, evaluating each at both the pointwise and
event-level against the GSEP catalog (Papaioannou et al., Flag==1).

Why 1995–2017:
    The GSEP catalog covers solar cycles 22–24, ending around 2017.
    This is the range where validated ground truth exists. Years
    2018–2025 have no external catalog for comparison — those are
    covered separately in experiments/detection/.

Outputs:
    - Per-year metrics printed to console (pointwise + event-level)
    - Aggregated summary table printed at the end
    - Per-year event CSVs saved to output/full_validation/
    - Aggregated metrics CSV saved to output/full_validation/summary_metrics.csv

Usage:
    cd sep_detection_pipeline/
    python -m experiments.validation.run_full_validation
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
from sep_core.evaluation.metrics import evaluate_detection, print_metrics
from sep_core.evaluation.matching import (
    match_events_to_catalog, print_event_metrics,
)


# ============================================
# CONFIGURATION
# ============================================

# Year range — full GSEP-validated window
START_YEAR = 1995
END_YEAR = 2017

# GSEP catalog path
GSEP_CATALOG_PATH = PROJECT_ROOT / "GSEP_List.csv"

# Cache directories for downloaded instrument data
GOES_CACHE = str(PROJECT_ROOT / "data" / "cache" / "goes")
SOHO_CACHE = str(PROJECT_ROOT / "data" / "cache" / "soho")

# Output directory for this validation run
OUTPUT_DIR = PROJECT_ROOT / "output" / "full_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# CATALOG LOADER
# ============================================

def load_gsep_for_year(year: int) -> pd.DataFrame:
    """
    Load the GSEP catalog filtered to a single year.

    Only returns Flag==1 (significant) events — the 245 events
    that cross the SWPC threshold of >10 MeV at 10 pfu. These
    are the events the pipeline's 10 pfu threshold is designed
    to detect, so they are the correct comparison set.

    Parameters
    ----------
    year : int
        The year to filter to.

    Returns
    -------
    pd.DataFrame
        GSEP catalog rows for the requested year.
        Empty DataFrame if no events exist for that year
        (expected for quiet years like 1995–1996).
    """

    from sep_core.evaluation.gsep_catalog import load_gsep_catalog

    return load_gsep_catalog(
        str(GSEP_CATALOG_PATH),
        start_year=year,
        end_year=year,
        significant_only=True,
    )


# ============================================
# SINGLE-YEAR PIPELINE
# ============================================

def run_year(
    year: int,
    goes_adapter: GOESAdapter,
    soho_adapter: SOHOAdapter,
) -> dict:
    """
    Run the full detection + evaluation pipeline for one year.

    Executes three stages:
        1. GOES-only detection and evaluation
        2. SOHO-only detection and evaluation
        3. Fused (GOES + SOHO) detection and evaluation

    Each stage produces both pointwise metrics (precision, recall, F1)
    and event-level metrics (detection rate, false alarm rate) against
    the GSEP catalog for that year.

    If either instrument returns no data for the year (e.g., SOHO CDF
    not available), that instrument is skipped gracefully and fusion
    runs with whatever data is available.

    Parameters
    ----------
    year : int
        The year to process (1995–2017).
    goes_adapter : GOESAdapter
        Pre-initialized GOES adapter (reused across years to
        keep cache configuration consistent).
    soho_adapter : SOHOAdapter
        Pre-initialized SOHO adapter.

    Returns
    -------
    dict
        Year-level results with keys:
            year : int
            catalog_events : int — number of GSEP events in this year
            goes_pw, soho_pw, fused_pw : dict or None — pointwise metrics
            goes_ev, soho_ev, fused_ev : dict or None — event-level metrics
            goes_n_detected, soho_n_detected, fused_n_detected : int
            elapsed_seconds : float — wall-clock time for this year
    """

    t0 = time_module.time()

    print(f"\n{'#' * 70}")
    print(f"#  YEAR {year}")
    print(f"{'#' * 70}")

    # ----------------------------------------------------------
    # Load ground truth
    # ----------------------------------------------------------
    catalog = load_gsep_for_year(year)
    n_catalog = len(catalog)
    print(f"\n  GSEP catalog events (Flag==1): {n_catalog}")

    # Prepare result dict with defaults
    result = {
        "year": year,
        "catalog_events": n_catalog,
        "goes_pw": None,  "goes_ev": None,  "goes_n_detected": 0,
        "soho_pw": None,  "soho_ev": None,  "soho_n_detected": 0,
        "fused_pw": None, "fused_ev": None, "fused_n_detected": 0,
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

        if n_catalog > 0:
            result["goes_pw"] = evaluate_detection(
                goes_result.mask, goes_result.time, catalog
            )
            result["goes_ev"] = match_events_to_catalog(
                goes_events, catalog
            )
    else:
        print(f"  [GOES] No data for {year}")

    # Save GOES events
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

        if n_catalog > 0:
            result["soho_pw"] = evaluate_detection(
                soho_result.mask, soho_result.time, catalog
            )
            result["soho_ev"] = match_events_to_catalog(
                soho_events, catalog
            )
    else:
        print(f"  [SOHO] No data for {year}")

    # Save SOHO events
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
        fused_event_list = [
            Event(fe.start_time, fe.end_time, fe.duration_minutes)
            for fe in fused
        ]
        result["fused_n_detected"] = len(fused_event_list)
        print(f"  [FUSION] {len(fused_event_list)} fused events")

        # Save fused events
        fused_df = fused_events_to_dataframe(fused)
        fused_df.to_csv(
            OUTPUT_DIR / f"fused_events_{year}.csv", index=False
        )

        # Evaluate fused on common time axis
        if n_catalog > 0 and has_goes and has_soho:
            common_start = max(goes_result.time[0], soho_result.time[0])
            common_end = min(goes_result.time[-1], soho_result.time[-1])
            common_time = pd.date_range(
                common_start, common_end, freq="5min"
            )

            if len(common_time) > 0:
                fused_mask = events_to_mask(fused_event_list, common_time)
                result["fused_pw"] = evaluate_detection(
                    fused_mask, common_time, catalog
                )
                result["fused_ev"] = match_events_to_catalog(
                    fused_event_list, catalog
                )

        elif n_catalog > 0 and (has_goes != has_soho):
            # Only one instrument available — fused = that instrument
            single_result = goes_result if has_goes else soho_result
            single_events = fused_event_list
            fused_mask = events_to_mask(single_events, single_result.time)
            result["fused_pw"] = evaluate_detection(
                fused_mask, single_result.time, catalog
            )
            result["fused_ev"] = match_events_to_catalog(
                single_events, catalog
            )

    result["elapsed_seconds"] = time_module.time() - t0
    print(f"\n  Year {year} completed in "
          f"{result['elapsed_seconds']:.1f}s")

    return result


# ============================================
# SUMMARY TABLE CONSTRUCTION
# ============================================

def build_summary_table(all_results: list) -> pd.DataFrame:
    """
    Flatten per-year results into a single summary DataFrame.

    Extracts event-level metrics from each year's result dict and
    assembles them into one row per year. All metrics are event-based
    (not pointwise): TP, FP, FN, Precision, Recall, F1, EDR, FAR.

    Columns include:
        - year, catalog_events
        - goes/soho/fused TP, FP, FN (event-level confusion matrix)
        - goes/soho/fused precision, recall, F1 (event-based)
        - goes/soho/fused EDR (event detection rate), FAR (false alarm rate)
        - goes/soho/fused detected event counts

    Parameters
    ----------
    all_results : list of dict
        One dict per year, from run_year().

    Returns
    -------
    pd.DataFrame
        One row per year, sorted by year.
    """

    rows = []

    for r in all_results:
        row = {
            "year": r["year"],
            "catalog_events": r["catalog_events"],
            "goes_events": r["goes_n_detected"],
            "soho_events": r["soho_n_detected"],
            "fused_events": r["fused_n_detected"],
        }

        # Extract event-level metrics (TP, FP, FN, Precision, Recall, F1, EDR, FAR)
        for source, key in [
            ("goes", "goes_ev"),
            ("soho", "soho_ev"),
            ("fused", "fused_ev"),
        ]:
            ev = r.get(key)
            if ev is not None:
                row[f"{source}_TP"] = ev["TP"]
                row[f"{source}_FP"] = ev["FP"]
                row[f"{source}_FN"] = ev["FN"]
                row[f"{source}_precision"] = ev["event_precision"]
                row[f"{source}_recall"] = ev["event_recall"]
                row[f"{source}_f1"] = ev["event_f1"]
                row[f"{source}_edr"] = ev["event_detection_rate"]
                row[f"{source}_far"] = ev["false_alarm_rate"]
            else:
                row[f"{source}_TP"] = 0
                row[f"{source}_FP"] = 0
                row[f"{source}_FN"] = 0
                row[f"{source}_precision"] = np.nan
                row[f"{source}_recall"] = np.nan
                row[f"{source}_f1"] = np.nan
                row[f"{source}_edr"] = np.nan
                row[f"{source}_far"] = np.nan

        row["elapsed_seconds"] = r["elapsed_seconds"]
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return df


def print_summary_table(df: pd.DataFrame) -> None:
    """
    Print the aggregated summary table to the console.

    Shows a per-instrument year-by-year breakdown of event-based
    metrics: TP, FP, FN, Precision, Recall, F1, EDR, FAR.

    Each instrument block ends with a SUM row: micro-averaged
    metrics computed from aggregated TP/FP/FN across all years.

    Parameters
    ----------
    df : pd.DataFrame
        Output from build_summary_table().
    """

    print(f"\n\n{'=' * 130}")
    print(f"  FULL VALIDATION SUMMARY (EVENT-BASED) — {df['year'].min()} to {df['year'].max()}")
    print(f"{'=' * 130}")

    # Per-instrument tables
    for source, label in [("goes", "GOES"), ("soho", "SOHO"), ("fused", "FUSED")]:
        print(f"\n  --- {label} ---")
        print(f"  {'Year':>4}  {'Cat':>3}  {'Det':>3}  "
              f"{'TP':>3} {'FP':>3} {'FN':>3}  "
              f"{'Prec':>6} {'Recall':>6} {'F1':>6}  "
              f"{'EDR':>6} {'FAR':>6}")
        print(f"  {'-' * 72}")

        for _, row in df.iterrows():
            def fmt(val):
                return f"{val:.4f}" if not np.isnan(val) else "  -   "

            cat = int(row['catalog_events'])
            det = int(row[f'{source}_events'])
            tp = int(row[f'{source}_TP'])
            fp = int(row[f'{source}_FP'])
            fn = int(row[f'{source}_FN'])

            print(
                f"  {int(row['year']):>4}  {cat:>3}  {det:>3}  "
                f"{tp:>3} {fp:>3} {fn:>3}  "
                f"{fmt(row[f'{source}_precision']):>6} "
                f"{fmt(row[f'{source}_recall']):>6} "
                f"{fmt(row[f'{source}_f1']):>6}  "
                f"{fmt(row[f'{source}_edr']):>6} "
                f"{fmt(row[f'{source}_far']):>6}"
            )

        print(f"  {'-' * 72}")

        total_tp = int(df[f'{source}_TP'].sum())
        total_fp = int(df[f'{source}_FP'].sum())
        total_fn = int(df[f'{source}_FN'].sum())
        total_det = int(df[f'{source}_events'].sum())
        total_cat = int(df['catalog_events'].sum())

        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = (
            2 * prec * rec / (prec + rec)
            if (prec + rec) > 0 else 0.0
        )
        edr = total_tp / total_cat if total_cat > 0 else 0.0
        far = total_fp / total_det if total_det > 0 else 0.0

        print(
            f"  {'TOTAL':>4}  {total_cat:>3}  {total_det:>3}  "
            f"{total_tp:>3} {total_fp:>3} {total_fn:>3}  "
            f"{prec:>6.4f} {rec:>6.4f} {f1:>6.4f}  "
            f"{edr:>6.4f} {far:>6.4f}"
        )

    # Grand totals
    total_catalog = int(df["catalog_events"].sum())
    total_goes = int(df["goes_events"].sum())
    total_soho = int(df["soho_events"].sum())
    total_fused = int(df["fused_events"].sum())
    total_time = df["elapsed_seconds"].sum()

    print(f"\n  {'=' * 50}")
    print(f"  Total catalog events:  {total_catalog}")
    print(f"  Total GOES detections: {total_goes}")
    print(f"  Total SOHO detections: {total_soho}")
    print(f"  Total fused events:    {total_fused}")
    print(f"  Total wall-clock time: {total_time / 60:.1f} minutes")
    print(f"{'=' * 130}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":

    print(f"\n{'=' * 70}")
    print(f"  SEP Detection Pipeline — Full Multi-Year Validation")
    print(f"  Years:       {START_YEAR} – {END_YEAR}")
    print(f"  Ground truth: GSEP catalog (Flag==1 significant events)")
    print(f"  Catalog:     {GSEP_CATALOG_PATH}")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    # Verify catalog exists
    if not GSEP_CATALOG_PATH.exists():
        print(f"\nERROR: GSEP catalog not found at {GSEP_CATALOG_PATH}")
        print(f"Place GSEP_List.csv in: {PROJECT_ROOT}")
        sys.exit(1)

    # Print catalog overview
    from sep_core.evaluation.gsep_catalog import (
        load_gsep_catalog, gsep_catalog_summary,
    )
    full_catalog = load_gsep_catalog(
        str(GSEP_CATALOG_PATH),
        start_year=START_YEAR,
        end_year=END_YEAR,
        significant_only=True,
    )
    print(f"\n{gsep_catalog_summary(full_catalog)}")

    # Initialize adapters once (reused across all years)
    goes_adapter = GOESAdapter(cache_dir=GOES_CACHE)
    soho_adapter = SOHOAdapter(cache_dir=SOHO_CACHE)

    # Run each year
    all_results = []
    total_start = time_module.time()

    for year in range(START_YEAR, END_YEAR + 1):
        year_result = run_year(year, goes_adapter, soho_adapter)
        all_results.append(year_result)

    total_elapsed = time_module.time() - total_start

    # Build and display summary
    summary_df = build_summary_table(all_results)
    print_summary_table(summary_df)

    # Save summary CSV
    summary_path = OUTPUT_DIR / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved to: {summary_path}")

    print(f"\n  Total runtime: {total_elapsed / 60:.1f} minutes")
    print(f"\nDone.")