"""
sep_core/evaluation/gsep_catalog.py

Loader and utilities for the GSEP (GOES SEP) catalog.

Reference:
    Papaioannou et al. — A catalog of solar energetic particle events
    covering solar cycles 22, 23, and 24.
    341 visually verified SEP events, ~1986–2017.

Catalog columns used here:
    timestamp    — actual SEP event start time
    slice_end    — actual event end time (flux returned below threshold)
    slice_start  — 12 hrs before timestamp (ML slice boundary, not used)
    gsep_pf_gt10MeV — peak >10 MeV proton flux in pfu
    Flag         — quality flag: 1 = clean significant event, 0 = minor/uncertain
    noaa-sep_flag — 1 if event also appears in NOAA SWPC catalog

Why GSEP is better than the raw NOAA scraped catalog for your evaluation:
    The NOAA table's "Maximum Time" column records the peak time, NOT the
    event end time. This causes the evaluation to count the entire decay
    phase of every event as false positives. GSEP's slice_end column is
    the actual end of the proton enhancement — the last time flux was
    above the detection threshold — making pointwise evaluation valid.

Usage:
    from sep_core.evaluation.gsep_catalog import load_gsep_catalog

    # All events 1995-2017
    catalog = load_gsep_catalog("GSEP_List.csv", start_year=1995)

    # Only Flag==1 events (clean, significant)
    catalog = load_gsep_catalog("GSEP_List.csv", start_year=1995,
                                significant_only=True)
"""

import pandas as pd
import numpy as np
from typing import Optional
from pathlib import Path


def load_gsep_catalog(
    catalog_path: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    significant_only: bool = False,
    min_peak_flux: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load and filter the GSEP SEP event catalog.

    Parameters
    ----------
    catalog_path : str
        Path to the GSEP_List.csv file.
    start_year : int or None
        Keep only events whose start (timestamp) is >= this year.
    end_year : int or None
        Keep only events whose start (timestamp) is <= this year.
    significant_only : bool
        If True, keep only events where Flag == 1.
        Flag==1 events are the 245 events that cross the SWPC
        significant proton event threshold (>10 MeV @ 10 pfu).
        Flag==0 events are minor or uncertain enhancements.
        Default False (keep all 341 events).
    min_peak_flux : float or None
        If provided, keep only events with gsep_pf_gt10MeV >= this value.
        Example: min_peak_flux=10.0 keeps events >= 10 pfu.

    Returns
    -------
    pd.DataFrame
        Columns: start_time, end_time, peak_flux_pfu, sep_id,
                 flag, noaa_flag, slice_start
        start_time : pd.Timestamp — actual SEP event onset
        end_time   : pd.Timestamp — actual event end (from slice_end)
        peak_flux_pfu : float — peak >10 MeV flux in pfu
        sep_id     : str — GSEP event identifier (e.g., "gsep_334")
        flag       : int — 0 or 1 quality flag
        noaa_flag  : int — 1 if also in NOAA SWPC catalog
        slice_start: pd.Timestamp — ML slice start (12 hrs before onset)
    """

    path = Path(catalog_path)
    if not path.exists():
        raise FileNotFoundError(f"GSEP catalog not found: {catalog_path}")

    df = pd.read_csv(catalog_path, low_memory=False)

    # Parse timestamps — some rows have empty timestamps (sub-events)
    df["start_time"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["end_time"] = pd.to_datetime(df["slice_end"], errors="coerce")
    df["slice_start"] = pd.to_datetime(df["slice_start"], errors="coerce")

    # Drop rows where we can't determine start or end
    df = df.dropna(subset=["start_time", "end_time"])

    # Rename for consistency with pipeline conventions
    df["sep_id"] = df["sep_index"].fillna("").astype(str)
    df["peak_flux_pfu"] = pd.to_numeric(df["gsep_pf_gt10MeV"], errors="coerce")
    df["flag"] = pd.to_numeric(df["Flag"], errors="coerce").fillna(0).astype(int)
    df["noaa_flag"] = pd.to_numeric(
        df["noaa-sep_flag"], errors="coerce"
    ).fillna(0).astype(int)

    # Keep only the columns we need
    df = df[[
        "sep_id", "start_time", "end_time", "slice_start",
        "peak_flux_pfu", "flag", "noaa_flag"
    ]].copy()

    # Apply filters
    if start_year is not None:
        df = df[df["start_time"].dt.year >= start_year]
    if end_year is not None:
        df = df[df["start_time"].dt.year <= end_year]
    if significant_only:
        df = df[df["flag"] == 1]
    if min_peak_flux is not None:
        df = df[df["peak_flux_pfu"] >= min_peak_flux]

    df = df.sort_values("start_time").reset_index(drop=True)

    return df


def gsep_catalog_summary(catalog: pd.DataFrame) -> str:
    """
    Print a human-readable summary of a loaded GSEP catalog.

    Parameters
    ----------
    catalog : pd.DataFrame
        Output of load_gsep_catalog().

    Returns
    -------
    str
        Summary string.
    """

    if catalog.empty:
        return "GSEP catalog: empty (no events match filters)"

    n_total = len(catalog)
    n_flag1 = int((catalog["flag"] == 1).sum())
    n_noaa = int((catalog["noaa_flag"] == 1).sum())

    year_min = catalog["start_time"].dt.year.min()
    year_max = catalog["start_time"].dt.year.max()

    dur = (catalog["end_time"] - catalog["start_time"]).dt.total_seconds() / 3600
    dur_mean = dur.mean()
    dur_max = dur.max()

    peak = catalog["peak_flux_pfu"].dropna()

    lines = [
        f"GSEP Catalog Summary",
        f"  Events:           {n_total}",
        f"  Flag==1 (significant): {n_flag1}",
        f"  Also in NOAA:     {n_noaa}",
        f"  Year range:       {year_min} – {year_max}",
        f"  Avg duration:     {dur_mean:.1f} hrs",
        f"  Max duration:     {dur_max:.1f} hrs",
        f"  Peak flux range:  {peak.min():.1f} – {peak.max():.0f} pfu",
        f"  Median peak flux: {peak.median():.1f} pfu",
    ]

    # Events per year
    year_counts = catalog["start_time"].dt.year.value_counts().sort_index()
    lines.append(f"\n  Events per year:")
    for yr, cnt in year_counts.items():
        lines.append(f"    {yr}: {cnt}")

    return "\n".join(lines)