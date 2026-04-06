"""

GOES instrument adapter for SEP detection.

Handles three instrument formats across 1995-2025:
  - EPS (GOES-8, GOES-12):       1995 – 2010, CSV, p3_flux_ic
  - EPEAD cpflux (GOES-13, -15): 2010 – 2020 (Mar), CSV, ZPGT10E
  - SGPS (GOES-16):              2020 (Nov) – 2025, NetCDF, derived >10 MeV

Coverage note:
  2020 Apr–Oct has no GOES data (GOES-15 EPEAD ends Mar 2020;
  GOES-16 SGPS avg5m starts Nov 2020). Months in this gap return
  empty DetectionResults.

The adapter encapsulates ALL GOES-specific knowledge:
  - satellite-year-month mapping with transition years (2003, 2010, 2020)
  - three different file formats and column names
  - URL construction for NCEI (legacy) and NGDC (GOES-R) archives
  - header parsing with "data:" marker for legacy CSVs
  - missing value handling (-99999 for EPS/EPEAD, fill values for SGPS)
  - quality flag checking for EPEAD
  - differential-to-integral flux conversion for SGPS
  - energy threshold in keV (10,000 keV = 10 MeV) for SGPS channel selection
  - east-facing sensor selection for EPEAD and SGPS
  - SGPS version discovery (v1-0-1 for 2020-2021, v2/v3 for 2022+)
  - SGPS time variable handling (L2_SciData_TimeStamp in v1, time in v2+)
  - alternative satellite fallback for GOES-12 (broken p3_flux_ic):
    uses GOES-11 or GOES-10 pre-computed integral flux instead

External code just calls:
    adapter = GOESAdapter(cache_dir="data/cache/goes")
    result = adapter.detect(year=2003, month=10)
"""

import io
import os
import re
import time as time_module
import calendar
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import requests

from sep_core.adapters.base import BaseAdapter
from sep_core.detections import DetectionResult
from sep_core.threshold import detect_sep_events


# ==================================================================
# SATELLITE-YEAR-MONTH MAPPING
# ==================================================================
# Maps (year, month) or (year, None) to (satellite_id, instrument_type)
#
# satellite_id: used in URL/filename construction (e.g., "goes08")
# instrument_type: "eps", "epead", or "sgps" — drives which parser is used
#
# Lookup order: (year, month) first, then (year, None) as fallback.
# Transition years (2003, 2010) have month-level entries because
# the primary GOES-East satellite changed mid-year.
#
# Sources:
# - GOES-8 decommissioned April 1, 2003 (Wikipedia/GOES)
# - GOES-12 became GOES-East April 2003 (eoPortal)
# - GOES-13 replaced GOES-12 as GOES-East April 14, 2010 (eoPortal)
# - GOES-16 became GOES-East December 18, 2017 (GOES-R Series)
# - SGPS L2 avg5m data available from 2020 onward (NCEI archive)

GOES_SATELLITE_MAP = {
    # GOES-8 / EPS — GOES-East 1995 through March 2003
    (1995, None): ("goes08", "eps"),
    (1996, None): ("goes08", "eps"),
    (1997, None): ("goes08", "eps"),
    (1998, None): ("goes08", "eps"),
    (1999, None): ("goes08", "eps"),
    (2000, None): ("goes08", "eps"),
    (2001, None): ("goes08", "eps"),
    (2002, None): ("goes08", "eps"),

    # 2003 transition: GOES-8 → GOES-12 in April
    (2003, 1):  ("goes08", "eps"),
    (2003, 2):  ("goes08", "eps"),
    (2003, 3):  ("goes08", "eps"),
    (2003, 4):  ("goes12", "eps"),
    (2003, 5):  ("goes12", "eps"),
    (2003, 6):  ("goes12", "eps"),
    (2003, 7):  ("goes12", "eps"),
    (2003, 8):  ("goes12", "eps"),
    (2003, 9):  ("goes12", "eps"),
    (2003, 10): ("goes12", "eps"),
    (2003, 11): ("goes12", "eps"),
    (2003, 12): ("goes12", "eps"),

    # GOES-12 / EPS — GOES-East 2004 through April 2010
    (2004, None): ("goes12", "eps"),
    (2005, None): ("goes12", "eps"),
    (2006, None): ("goes12", "eps"),
    (2007, None): ("goes12", "eps"),
    (2008, None): ("goes12", "eps"),
    (2009, None): ("goes12", "eps"),

    # 2010 transition: GOES-12 → GOES-13 in May
    # (GOES-13 became GOES-East April 14, we use May for clean month boundary)
    (2010, 1):  ("goes12", "eps"),
    (2010, 2):  ("goes12", "eps"),
    (2010, 3):  ("goes12", "eps"),
    (2010, 4):  ("goes12", "eps"),
    (2010, 5):  ("goes13", "epead"),
    (2010, 6):  ("goes13", "epead"),
    (2010, 7):  ("goes13", "epead"),
    (2010, 8):  ("goes13", "epead"),
    (2010, 9):  ("goes13", "epead"),
    (2010, 10): ("goes13", "epead"),
    (2010, 11): ("goes13", "epead"),
    (2010, 12): ("goes13", "epead"),

    # GOES-13 / EPEAD — GOES-East 2011 through 2017
    (2011, None): ("goes13", "epead"),
    (2012, None): ("goes13", "epead"),
    (2013, None): ("goes13", "epead"),
    (2014, None): ("goes13", "epead"),
    (2015, None): ("goes13", "epead"),
    (2016, None): ("goes13", "epead"),
    (2017, None): ("goes13", "epead"),

    # GOES-15 / EPEAD — 2018 through March 2020
    (2018, None): ("goes15", "epead"),
    (2019, None): ("goes15", "epead"),

    # 2020 transition: GOES-15 EPEAD ends March, GOES-16 SGPS starts November
    # April–October 2020 has no GOES data (returns empty result)
    (2020, 1):  ("goes15", "epead"),
    (2020, 2):  ("goes15", "epead"),
    (2020, 3):  ("goes15", "epead"),
    (2020, 11): ("goes16", "sgps"),
    (2020, 12): ("goes16", "sgps"),

    # GOES-16 / SGPS — 2021 through 2025
    (2021, None): ("goes16", "sgps"),
    (2022, None): ("goes16", "sgps"),
    (2023, None): ("goes16", "sgps"),
    (2024, None): ("goes16", "sgps"),
    (2025, None): ("goes16", "sgps"),
}


def lookup_satellite(year: int, month: int) -> Optional[Tuple[str, str]]:
    """
    Look up which satellite and instrument to use for a given year/month.

    Checks (year, month) first for transition years,
    then falls back to (year, None) for normal years.

    Returns
    -------
    tuple of (satellite_id, instrument_type), or None if no data
    exists for this month (e.g., 2020 Apr-Oct gap between
    GOES-15 EPEAD and GOES-16 SGPS).

    Raises
    ------
    ValueError
        If the year is outside the supported range (1995-2025).
    """

    key = (year, month)
    if key in GOES_SATELLITE_MAP:
        return GOES_SATELLITE_MAP[key]

    key = (year, None)
    if key in GOES_SATELLITE_MAP:
        return GOES_SATELLITE_MAP[key]

    if 1995 <= year <= 2025:
        return None

    raise ValueError(
        f"No GOES satellite mapping for year={year}, month={month}. "
        f"Supported range: 1995-2025."
    )


# ==================================================================
# ALTERNATIVE SATELLITE FALLBACK FOR GOES-12 (broken p3_flux_ic)
# ==================================================================
# GOES-12 EPS files have p3_flux_ic = -99999 everywhere (NOAA never
# computed corrected integral flux for GOES-12). Instead of deriving
# integral flux from differential channels, we can use the pre-computed
# p3_flux_ic from another satellite that was operational during the
# same period. Data availability (verified):
#
#   GOES-8:  Jan 2003 – June 2003 (decommissioned)
#   GOES-10: Jan 2003 – Dec 2009  (full coverage)
#   GOES-11: June 2003 – Dec 2009 (full coverage)
#
# Priority: GOES-11 first (it was the standby GOES-East), then GOES-10.
# Both have valid p3_flux_ic with thousands of positive readings per month.

EPS_FALLBACK_SATELLITES = {
    (2003, 1):  ["goes08", "goes10"],
    (2003, 2):  ["goes08", "goes10"],
    (2003, 3):  ["goes08", "goes10"],
    (2003, 4):  ["goes11", "goes10"],
    (2003, 5):  ["goes11", "goes10"],
    (2003, 6):  ["goes11", "goes10"],
    (2003, 7):  ["goes11", "goes10"],
    (2003, 8):  ["goes11", "goes10"],
    (2003, 9):  ["goes11", "goes10"],
    (2003, 10): ["goes11", "goes10"],
    (2003, 11): ["goes11", "goes10"],
    (2003, 12): ["goes11", "goes10"],
}

# 2004–2009: GOES-11 and GOES-10 both have full-year coverage
for _y in range(2004, 2010):
    for _m in range(1, 13):
        EPS_FALLBACK_SATELLITES[(_y, _m)] = ["goes11", "goes10"]

# 2010 Jan–Apr (still GOES-12 in the primary map)
for _m in range(1, 5):
    EPS_FALLBACK_SATELLITES[(2010, _m)] = ["goes11", "goes10"]


# ==================================================================
# URL CONSTRUCTION
# ==================================================================
# Legacy archive (EPS + EPEAD) at NCEI:
#   https://www.ncei.noaa.gov/data/goes-space-environment-monitor/
#       access/avg/{year}/{month:02d}/{satellite}/csv/{filename}
#
# GOES-R archive (SGPS) at NGDC:
#   https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/
#       goes/goes16/l2/data/sgps-l2-avg5m/{year}/{month:02d}/{filename}
#
# URL structure verified: month comes BEFORE satellite in legacy archive.

NCEI_BASE_URL = (
    "https://www.ncei.noaa.gov/data/"
    "goes-space-environment-monitor/access/avg"
)

SGPS_BASE_URL = (
    "https://data.ngdc.noaa.gov/platforms/"
    "solar-space-observing-satellites/goes/goes16/"
    "l2/data/sgps-l2-avg5m"
)


def _sat_short(satellite: str) -> str:
    """
    Convert satellite ID to short form for filenames.
    "goes08" → "g08", "goes12" → "g12", "goes13" → "g13"
    """
    num = satellite.replace("goes", "")
    return f"g{num}"


def build_eps_url(satellite: str, year: int, month: int) -> str:
    """
    Build download URL for a GOES EPS 5-minute CSV file.

    Example:
        .../avg/1997/05/goes08/csv/g08_eps_5m_19970501_19970531.csv
    """
    short = _sat_short(satellite)
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year}{month:02d}01"
    end = f"{year}{month:02d}{last_day:02d}"

    return (
        f"{NCEI_BASE_URL}/{year}/{month:02d}/{satellite}/csv/"
        f"{short}_eps_5m_{start}_{end}.csv"
    )


def build_epead_url(satellite: str, year: int, month: int) -> str:
    """
    Build download URL for a GOES EPEAD cpflux 5-minute CSV file.

    Example:
        .../avg/2016/08/goes13/csv/g13_epead_cpflux_5m_20160801_20160831.csv
    """
    short = _sat_short(satellite)
    last_day = calendar.monthrange(year, month)[1]
    start = f"{year}{month:02d}01"
    end = f"{year}{month:02d}{last_day:02d}"

    return (
        f"{NCEI_BASE_URL}/{year}/{month:02d}/{satellite}/csv/"
        f"{short}_epead_cpflux_5m_{start}_{end}.csv"
    )


_SGPS_VERSION_CACHE: Dict[Tuple[int, int], Optional[str]] = {}


def _discover_sgps_version(year: int, month: int) -> Optional[str]:
    """
    Discover the SGPS file version string for a given year/month.

    The version changes over time:
        2020-2021 (early):  v1-0-1
        2021 (Dec) – 2022 (May): v2-0-0
        2022 (Jun) – 2023 (early): v3-0-0
        2023 (mid):         v3-0-1
        2024-2025:          v3-0-2

    Rather than hardcoding these, we list the NGDC directory for the
    month and extract the version from the first filename found.
    Results are cached per (year, month) to avoid repeated HTTP calls.
    """

    cache_key = (year, month)
    if cache_key in _SGPS_VERSION_CACHE:
        return _SGPS_VERSION_CACHE[cache_key]

    dir_url = f"{SGPS_BASE_URL}/{year}/{month:02d}/"
    try:
        resp = requests.get(dir_url, timeout=30)
        if resp.status_code != 200:
            _SGPS_VERSION_CACHE[cache_key] = None
            return None

        match = re.search(
            r"sci_sgps-l2-avg5m_g16_d\d{8}_(v[\d]+-[\d]+-[\d]+)\.nc",
            resp.text,
        )
        if match:
            version = match.group(1)
            _SGPS_VERSION_CACHE[cache_key] = version
            return version

    except requests.RequestException:
        pass

    _SGPS_VERSION_CACHE[cache_key] = None
    return None


def build_sgps_url(year: int, month: int, day: int,
                   version: str = "v1-0-1") -> str:
    """
    Build download URL for one daily GOES-16 SGPS L2 avg5m NetCDF file.

    SGPS files are DAILY, not monthly.

    Example:
        .../sgps-l2-avg5m/2021/03/sci_sgps-l2-avg5m_g16_d20210301_v1-0-1.nc
    """
    date_str = f"{year}{month:02d}{day:02d}"
    filename = f"sci_sgps-l2-avg5m_g16_d{date_str}_{version}.nc"
    return f"{SGPS_BASE_URL}/{year}/{month:02d}/{filename}"


# ==================================================================
# DOWNLOAD + CACHING
# ==================================================================

def download_with_retry(
    url: str,
    save_path: Path,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> bool:
    """
    Download a file with retry logic.

    Returns True on success, False on 404 (file not found).
    Raises on other HTTP errors after all retries.
    Creates parent directories automatically.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=60)

            if resp.status_code == 404:
                return False

            resp.raise_for_status()

            with open(save_path, "wb") as f:
                f.write(resp.content)
            return True

        except requests.RequestException:
            if attempt < max_retries - 1:
                time_module.sleep(retry_delay)
            else:
                raise

    return False


# ==================================================================
# LEGACY CSV PARSING (EPS + EPEAD)
# ==================================================================

def parse_legacy_csv(filepath: Path) -> pd.DataFrame:
    """
    Parse a GOES legacy CSV file (EPS or EPEAD).

    Both formats have a long metadata header (hundreds of lines)
    ending with a "data:" line. The actual CSV data starts on the
    line after "data:". This was verified in our inspection of
    both EPS (line 454) and EPEAD cpflux (line 718) files.
    """

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    # Find the "data:" marker
    data_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "data:":
            data_start = i + 1
            break

    if data_start is None:
        print(f"  WARNING: No 'data:' marker in {filepath.name}")
        return pd.DataFrame()

    csv_text = "".join(lines[data_start:])

    try:
        df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        print(f"  WARNING: Failed to parse {filepath.name}: {e}")
        return pd.DataFrame()

    return df


def extract_eps_flux(
    df: pd.DataFrame,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Extract time and >10 MeV integral flux from EPS data.

    Primary path (GOES-8, 1995–2003):
        Uses p3_flux_ic — the pre-computed corrected integral >10 MeV
        proton flux. Units: pfu (protons / cm² s sr). This column is
        populated by NOAA for GOES-8 but NOT for GOES-12.

    Fallback path (GOES-12, 2003–2009):
        GOES-12 EPS files have the p3_flux_ic column header but every
        value is -99999 (missing). NOAA never computed the corrected
        integral product for GOES-12. The raw differential channels
        (p3_flux through p7_flux) DO have valid data.

        When p3_flux_ic is entirely missing, we derive the >10 MeV
        integral flux by integrating the differential channels:

            J(>10 MeV) = Σ [ diff_flux_i × ΔE_i ]

        using documented energy boundaries from the EPS metadata:

        Channel  E_min (MeV)  E_max (MeV)  Full ΔE  Effective ΔE
        P3       8.7          14.5         5.8      4.5 (10.0–14.5)
        P4       15.0         44.0         29.0     29.0
        P5       40.0         80.0         40.0     40.0
        P6       80.0         165.0        85.0     85.0
        P7       165.0        500.0        335.0    335.0

        P3 uses an effective width of 4.5 MeV (14.5 - 10.0) instead
        of the full 5.8 MeV, applying a flat-spectrum partial-bin
        correction to exclude the sub-10 MeV portion — same approach
        used in the SOHO adapter for EPHIN channel P8.

        Units: diff_flux is p/(cm² s sr MeV) × MeV = p/(cm² s sr) = pfu.

    Missing flag: -99999 for both paths.
    """

    # Parse time — GOES CSV uses "time_tag" column
    time = pd.to_datetime(df["time_tag"], utc=True).dt.tz_localize(None)

    # Try primary path: pre-computed integral flux
    flux = _try_eps_integral(df)

    if flux is None:
        # Fallback: derive from differential channels
        flux = _derive_eps_integral_from_differential(df)

    # Build index, sort, deduplicate
    time_index = pd.DatetimeIndex(time)
    sort_idx = time_index.argsort()
    time_index = time_index[sort_idx]
    flux = flux[sort_idx]

    dup_mask = ~time_index.duplicated(keep="first")
    time_index = time_index[dup_mask]
    flux = flux[dup_mask]

    return time_index, flux


# Documented EPS differential proton channel boundaries (from file metadata).
# "description" gives nominal ranges; "long_label" gives actual measured
# ranges. We use the long_label values as they reflect real instrument response.
#
# Channel  description       long_label (actual)   units
# P3       9.0 - 15.0 MeV   8.7 - 14.5 MeV       p/(cm² s sr MeV)
# P4       15.0 - 44.0 MeV   15.0 - 44.0 MeV      p/(cm² s sr MeV)
# P5       40.0 - 80.0 MeV   40.0 - 80.0 MeV      p/(cm² s sr MeV)
# P6       80.0 - 165.0 MeV  80.0 - 165.0 MeV     p/(cm² s sr MeV)
# P7       165.0 - 500.0 MeV 165.0 - 500.0 MeV    p/(cm² s sr MeV)

EPS_DIFF_CHANNELS = {
    "p3_flux": {"e_min":   8.7, "e_max":  14.5},
    "p4_flux": {"e_min":  15.0, "e_max":  44.0},
    "p5_flux": {"e_min":  40.0, "e_max":  80.0},
    "p6_flux": {"e_min":  80.0, "e_max": 165.0},
    "p7_flux": {"e_min": 165.0, "e_max": 500.0},
}

# Effective energy widths for the >10 MeV integral proxy.
# P3 uses 10.0–14.5 = 4.5 MeV (flat-spectrum partial-bin correction,
# same approach as SOHO P8). All other channels use full width.
EPS_EFFECTIVE_WIDTHS = {
    "p3_flux":  4.5,    # 14.5 - 10.0; excludes 8.7–10 MeV sub-bin
    "p4_flux": 29.0,    # 44.0 - 15.0; full channel
    "p5_flux": 40.0,    # 80.0 - 40.0; full channel
    "p6_flux": 85.0,    # 165.0 - 80.0; full channel
    "p7_flux": 335.0,   # 500.0 - 165.0; full channel
}


def _try_eps_integral(df: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Try to use the pre-computed p3_flux_ic column.

    Returns the flux array if p3_flux_ic exists and has at least some
    valid (non-NaN) data. Returns None if the column is missing or
    entirely filled with the -99999 missing flag (as in GOES-12).
    """

    if "p3_flux_ic" not in df.columns:
        return None

    flux = df["p3_flux_ic"].to_numpy(dtype=np.float64)
    flux[flux <= -99998] = np.nan

    if np.all(np.isnan(flux)):
        return None

    return flux


def _derive_eps_integral_from_differential(
    df: pd.DataFrame,
) -> np.ndarray:
    """
    Derive >10 MeV integral flux from EPS differential channels.

    Used as fallback when p3_flux_ic is unavailable (GOES-12).
    Integrates p3_flux through p7_flux using documented energy
    widths, with a flat-spectrum partial-bin correction on P3
    to approximate the >10 MeV lower bound.

    Result:
        proxy = p3 × 4.5 + p4 × 29.0 + p5 × 40.0 + p6 × 85.0 + p7 × 335.0

    Parameters
    ----------
    df : pd.DataFrame
        EPS CSV data with p3_flux through p7_flux columns.

    Returns
    -------
    np.ndarray
        Derived integral flux in pfu. Non-positive values set to NaN.
    """

    n = len(df)
    flux = np.zeros(n, dtype=np.float64)

    for col, width in EPS_EFFECTIVE_WIDTHS.items():
        if col not in df.columns:
            print(f"  WARNING: EPS column '{col}' missing — skipping")
            continue

        ch = df[col].to_numpy(dtype=np.float64)
        ch[ch <= -99998] = np.nan
        flux += np.nan_to_num(ch, nan=0.0) * width

    # Where ALL channels were NaN, result should be NaN not 0
    all_missing = np.ones(n, dtype=bool)
    for col in EPS_EFFECTIVE_WIDTHS:
        if col in df.columns:
            ch = df[col].to_numpy(dtype=np.float64)
            ch[ch <= -99998] = np.nan
            all_missing &= np.isnan(ch)
    flux[all_missing] = np.nan

    # Non-positive values are unphysical
    flux[flux <= 0] = np.nan

    return flux


def extract_epead_flux(
    df: pd.DataFrame,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Extract time and >10 MeV integral flux from EPEAD cpflux data.

    Target column: ZPGT10E (east-facing sensor, corrected integral >10 MeV)
    Fallback: ZPGT10W (west-facing sensor)
    Quality flag: ZPGT10E_QUAL_FLAG — nonzero means bad data

    Units: pfu (protons / cm² s sr)
    Missing flag: -99999

    From our inspection: quiet-time values ~0.12-0.19 pfu.
    """

    # Select flux column — prefer east sensor
    if "ZPGT10E" in df.columns:
        flux_col = "ZPGT10E"
        qual_col = "ZPGT10E_QUAL_FLAG"
    elif "ZPGT10W" in df.columns:
        flux_col = "ZPGT10W"
        qual_col = "ZPGT10W_QUAL_FLAG"
    else:
        raise ValueError(
            f"EPEAD data missing ZPGT10E and ZPGT10W. "
            f"Columns: {list(df.columns)}"
        )

    # Parse time
    time = pd.to_datetime(df["time_tag"], utc=True).dt.tz_localize(None)

    # Extract flux
    flux = df[flux_col].to_numpy(dtype=np.float64)

    # Replace missing flags
    flux[flux <= -99998] = np.nan

    # Apply quality flags — nonzero flag means bad data
    if qual_col in df.columns:
        qual = df[qual_col].to_numpy()
        flux[qual != 0] = np.nan

    # Sort and deduplicate
    time_index = pd.DatetimeIndex(time)
    sort_idx = time_index.argsort()
    time_index = time_index[sort_idx]
    flux = flux[sort_idx]

    dup_mask = ~time_index.duplicated(keep="first")
    time_index = time_index[dup_mask]
    flux = flux[dup_mask]

    return time_index, flux


# ==================================================================
# SGPS NETCDF PARSING (GOES-R)
# ==================================================================

def extract_sgps_flux(
    filepath: Path,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Extract time and derived >10 MeV integral flux from SGPS NetCDF.

    SGPS provides 13 differential proton channels (1-500 MeV) with
    energy bounds in keV. There is NO pre-computed >10 MeV integral flux.

    We derive it by integrating differential channels above 10 MeV:
        integral_flux = sum( diff_flux[ch] × (upper_keV[ch] - lower_keV[ch]) )
    for channels where lower_energy >= 10,000 keV (= 10 MeV).

    Units: diff_flux is protons/(cm² sr keV s)
           × keV gives protons/(cm² sr s) = pfu

    Sensor: index 1 = east-facing when spacecraft upright.
    Fill value: < -1e29 (from inspection: -9.999999848243207e+30)

    Time variable varies by SGPS file version:
        v1-0-1 (2020-2021): "L2_SciData_TimeStamp"
        v2-0-0+ (2022+):    "time"
    """

    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray required for SGPS. Install: pip install xarray netcdf4"
        )

    ds = xr.open_dataset(filepath)

    # Time — variable name differs between file versions
    if "L2_SciData_TimeStamp" in ds.variables:
        time_raw = ds["L2_SciData_TimeStamp"].values
    elif "time" in ds.variables:
        time_raw = ds["time"].values
    else:
        ds.close()
        raise ValueError(
            f"SGPS file has no recognized time variable. "
            f"Variables: {list(ds.variables.keys())}"
        )

    time_index = pd.DatetimeIndex(time_raw)
    if time_index.tz is not None:
        time_index = time_index.tz_localize(None)

    # Energy bounds — shape: (2 sensors, 13 channels), units: keV
    lower_energy = ds["DiffProtonLowerEnergy"].values   # (2, 13)
    upper_energy = ds["DiffProtonUpperEnergy"].values   # (2, 13)

    # Differential flux — shape: (time, 2 sensors, 13 channels)
    # Units: protons/(cm² sr keV s)
    diff_flux = ds["AvgDiffProtonFlux"].values          # (N, 2, 13)

    ds.close()

    # Select east-facing sensor (index 1)
    sensor_idx = 1
    lower_E = lower_energy[sensor_idx, :]    # (13,) keV
    upper_E = upper_energy[sensor_idx, :]    # (13,) keV
    flux_diff = diff_flux[:, sensor_idx, :]  # (N, 13)

    # Find channels with lower energy >= 10 MeV = 10,000 keV
    # IMPORTANT: energy bounds are in keV, not MeV
    threshold_keV = 10_000.0
    above_10mev = lower_E >= threshold_keV

    if not above_10mev.any():
        print("  WARNING: No SGPS channels above 10 MeV")
        flux_out = np.full(len(time_index), np.nan)
        return time_index, flux_out

    # Select channels and compute bin widths
    delta_E = upper_E[above_10mev] - lower_E[above_10mev]  # (n_ch,) keV
    flux_above = flux_diff[:, above_10mev]                   # (N, n_ch)

    # Replace fill values with NaN
    flux_above = np.where(flux_above < -1e29, np.nan, flux_above)

    # Numerical integration: sum(diff_flux × delta_E)
    # protons/(cm² sr keV s) × keV = protons/(cm² sr s) = pfu
    integral_flux = np.nansum(flux_above * delta_E[np.newaxis, :], axis=1)

    # Where ALL channels were NaN, result should be NaN not 0
    all_nan = np.all(np.isnan(flux_above), axis=1)
    integral_flux[all_nan] = np.nan

    # Sort and deduplicate
    sort_idx = time_index.argsort()
    time_index = time_index[sort_idx]
    integral_flux = integral_flux[sort_idx]

    dup_mask = ~time_index.duplicated(keep="first")
    time_index = time_index[dup_mask]
    integral_flux = integral_flux[dup_mask]

    return time_index, integral_flux.astype(np.float64)


# ==================================================================
# THE ADAPTER CLASS
# ==================================================================

class GOESAdapter(BaseAdapter):
    """
    GOES instrument adapter for SEP detection.

    Handles EPS (1995-2010), EPEAD (2010-2020), and SGPS (2021-2025).
    One adapter for all years — satellite/instrument selection is automatic.

    Usage:
        adapter = GOESAdapter(cache_dir="data/cache/goes")

        # Single month
        result = adapter.detect(year=2003, month=10)
        print(result.summary())

        # Full year
        result = adapter.detect_year(year=2003)
    """

    def __init__(self, cache_dir: str = "data/cache/goes"):
        super().__init__(name="GOES", cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ------------------------------------------------------------------

    def fetch_data(
        self, year: int, month: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Download or load cached GOES data for a given year/month.

        For EPS/EPEAD: downloads one monthly CSV file.
        For SGPS: downloads all daily NetCDF files for the month,
        extracts and concatenates them.

        Returns empty DataFrame if data is unavailable (including
        the 2020 Apr-Oct gap between GOES-15 and GOES-16).
        """

        if month is None:
            raise ValueError(
                "GOES adapter requires a month. "
                "Use detect(year, month) or detect_year(year)."
            )

        mapping = lookup_satellite(year, month)
        if mapping is None:
            print(f"  No GOES data available for {year}-{month:02d} "
                  f"(data gap)")
            return pd.DataFrame()

        satellite, instrument = mapping

        if instrument in ("eps", "epead"):
            return self._fetch_legacy(satellite, instrument, year, month)
        elif instrument == "sgps":
            return self._fetch_sgps(year, month)
        else:
            raise ValueError(f"Unknown instrument: {instrument}")

    def parse_flux(
        self, raw_data: pd.DataFrame
    ) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Extract clean time and flux from raw GOES data.

        Detects instrument type from columns present:
        - "p3_flux_ic" → EPS
        - "ZPGT10E" or "ZPGT10W" → EPEAD
        - "sgps_integral_flux" → SGPS (already extracted)
        """

        if "p3_flux_ic" in raw_data.columns or "p3_flux" in raw_data.columns:
            return extract_eps_flux(raw_data)

        elif "ZPGT10E" in raw_data.columns or "ZPGT10W" in raw_data.columns:
            return extract_epead_flux(raw_data)

        elif "sgps_integral_flux" in raw_data.columns:
            return self._parse_sgps_dataframe(raw_data)

        else:
            raise ValueError(
                f"Cannot determine instrument from columns: "
                f"{list(raw_data.columns)}"
            )

    def get_threshold_params(self) -> Dict[str, Any]:
        """
        Return GOES detection parameters.

        All GOES instruments use the same detection logic:
        - Entry threshold: 10 pfu (flux must rise above 10 to start)
        - Exit threshold: 5 pfu (hysteresis — event continues until
          flux drops below 5 pfu)
        - Quiet period: 2 hours (24 × 5-min points — flux must stay
          below 5 pfu for 2 continuous hours to confirm event end)
        - Rising gradient: 3 out of 4 positive steps
        - Minimum duration: 30 minutes (6 × 5-min points)
        """

        return {
            "type": "fixed",
            "value": 10.0,
            "exit_threshold": 5.0,
            "quiet_period_points": 24,
            "gradient_window": 4,
            "allow_negative_inside": 1,
            "min_duration_points": 6,
        }

    # ------------------------------------------------------------------
    # DETECT (overrides base class to use detect_sep_events)
    # ------------------------------------------------------------------

    def detect(
        self, year: int, month: Optional[int] = None
    ) -> DetectionResult:
        """
        Run full GOES SEP detection for one month.

        Overrides base class detect() to use detect_sep_events()
        from threshold.py, which returns (mask, info) with the
        three-rule detection logic.

        For GOES-12 months (Apr 2003 – Apr 2010) where the primary
        satellite's p3_flux_ic is broken, tries to use integral flux
        from an alternative satellite (GOES-11 or GOES-10) before
        falling back to differential-to-integral derivation.

        Returns an empty result for months with no data (e.g., the
        2020 Apr-Oct gap).
        """

        if month is None:
            raise ValueError(
                "GOES adapter requires a month. "
                "Use detect(year, month) or detect_year(year)."
            )

        mapping = lookup_satellite(year, month)
        if mapping is None:
            return self._empty_result(year, month)

        satellite, instrument = mapping

        # Step 1: Fetch primary satellite data
        raw_data = self.fetch_data(year, month)

        if raw_data.empty:
            return self._empty_result(year, month)

        # Step 2: Parse flux — with fallback satellite logic for EPS
        flux_source = satellite
        if instrument == "eps":
            time, flux, flux_source = self._parse_eps_with_fallback(
                raw_data, year, month, satellite
            )
        else:
            time, flux = self.parse_flux(raw_data)

        self.validate_time_series(time, flux)

        instrument_name = f"{flux_source.upper()}/{instrument.upper()}"

        # Step 3: Detection with hysteresis
        params = self.get_threshold_params()
        sep_mask, detection_info = detect_sep_events(
            flux,
            threshold=params["value"],
            gradient_window=params["gradient_window"],
            allow_negative_inside=params["allow_negative_inside"],
            min_duration_points=params["min_duration_points"],
            exit_threshold=params.get("exit_threshold"),
            quiet_period_points=params.get("quiet_period_points", 0),
        )

        # Step 4: Build result
        return DetectionResult(
            instrument=instrument_name,
            time=time,
            flux=flux,
            mask=sep_mask,
            metadata={
                "year": year,
                "month": month,
                "satellite": satellite,
                "flux_source_satellite": flux_source,
                "instrument_type": instrument,
                "threshold": params,
                "detection_info": {
                    k: v for k, v in detection_info.items()
                    if not isinstance(v, np.ndarray)
                },
                "n_raw_rows": len(raw_data),
                "n_valid_flux": int(np.sum(~np.isnan(flux))),
                "n_detected": int(sep_mask.sum()),
                "status": "ok",
            }
        )

    def detect_year(self, year: int) -> DetectionResult:
        """
        Run detection for all 12 months and concatenate.

        Handles transition years automatically — each month
        uses the correct satellite via the mapping.
        """

        all_times = []
        all_fluxes = []
        all_masks = []
        monthly_meta = []

        for month in range(1, 13):
            result = self.detect(year, month)

            if result.n_timestamps > 0:
                all_times.append(result.time)
                all_fluxes.append(result.flux)
                all_masks.append(result.mask)
                monthly_meta.append(result.metadata)

        if not all_times:
            return self._empty_result(year, None)

        # Concatenate all months
        combined_time = all_times[0].append(all_times[1:])
        combined_flux = np.concatenate(all_fluxes)
        combined_mask = np.concatenate(all_masks)

        # Sort (handles month boundaries)
        sort_idx = combined_time.argsort()
        combined_time = combined_time[sort_idx]
        combined_flux = combined_flux[sort_idx]
        combined_mask = combined_mask[sort_idx]

        # Deduplicate
        dup_mask = ~combined_time.duplicated(keep="first")
        combined_time = combined_time[dup_mask]
        combined_flux = combined_flux[dup_mask]
        combined_mask = combined_mask[dup_mask]

        # Instrument name from first available month
        satellite, instrument = lookup_satellite(
            year, monthly_meta[0]["month"]
        )
        instrument_name = f"{satellite.upper()}/{instrument.upper()}"

        return DetectionResult(
            instrument=instrument_name,
            time=combined_time,
            flux=combined_flux,
            mask=combined_mask,
            metadata={
                "year": year,
                "months_with_data": len(monthly_meta),
                "monthly_metadata": monthly_meta,
                "status": "ok",
            }
        )

    # ------------------------------------------------------------------
    # INTERNAL: EPS FALLBACK TO ALTERNATIVE SATELLITE
    # ------------------------------------------------------------------

    def _parse_eps_with_fallback(
        self,
        raw_data: pd.DataFrame,
        year: int,
        month: int,
        primary_satellite: str,
    ) -> Tuple[pd.DatetimeIndex, np.ndarray, str]:
        """
        Parse EPS flux, falling back to an alternative satellite's
        pre-computed integral flux when the primary satellite's
        p3_flux_ic is broken.

        Returns (time, flux, source_satellite).

        Fallback order:
        1. Primary satellite p3_flux_ic (if valid)
        2. Alternative satellite p3_flux_ic (GOES-11, then GOES-10)
        3. Differential-to-integral derivation from primary satellite
        """
        time = pd.to_datetime(
            raw_data["time_tag"], utc=True
        ).dt.tz_localize(None)
        time_index = pd.DatetimeIndex(time)

        # Try primary satellite's integral flux first
        primary_flux = _try_eps_integral(raw_data)
        if primary_flux is not None:
            sort_idx = time_index.argsort()
            time_index = time_index[sort_idx]
            primary_flux = primary_flux[sort_idx]
            dup_mask = ~time_index.duplicated(keep="first")
            return time_index[dup_mask], primary_flux[dup_mask], primary_satellite

        # Primary is broken — try alternative satellites
        fallback_sats = EPS_FALLBACK_SATELLITES.get((year, month), [])

        for alt_sat in fallback_sats:
            print(
                f"  {primary_satellite} p3_flux_ic broken for "
                f"{year}-{month:02d}, trying {alt_sat}..."
            )
            alt_df = self._fetch_legacy(alt_sat, "eps", year, month)

            if alt_df.empty:
                print(f"    {alt_sat}: no data available")
                continue

            alt_flux = _try_eps_integral(alt_df)
            if alt_flux is None:
                print(f"    {alt_sat}: p3_flux_ic also broken")
                continue

            # Got valid integral flux from alternative satellite.
            # Use the primary satellite's time grid and align.
            alt_time = pd.to_datetime(
                alt_df["time_tag"], utc=True
            ).dt.tz_localize(None)
            alt_time_index = pd.DatetimeIndex(alt_time)

            # Sort and deduplicate alt data
            sort_idx = alt_time_index.argsort()
            alt_time_index = alt_time_index[sort_idx]
            alt_flux = alt_flux[sort_idx]
            dup_mask = ~alt_time_index.duplicated(keep="first")
            alt_time_index = alt_time_index[dup_mask]
            alt_flux = alt_flux[dup_mask]

            valid_count = int(np.sum(~np.isnan(alt_flux)))
            print(
                f"    {alt_sat}: OK — {valid_count} valid flux values"
            )
            return alt_time_index, alt_flux, alt_sat

        # No alternative worked — fall back to differential derivation
        print(
            f"  No alternative satellite available for "
            f"{year}-{month:02d}, deriving from differential channels"
        )
        time_index, flux = extract_eps_flux(raw_data)
        return time_index, flux, primary_satellite

    # ------------------------------------------------------------------
    # INTERNAL: LEGACY DATA (EPS + EPEAD)
    # ------------------------------------------------------------------

    def _fetch_legacy(
        self, satellite: str, instrument: str, year: int, month: int
    ) -> pd.DataFrame:
        """
        Fetch one monthly EPS or EPEAD CSV file.
        Checks cache first, downloads if needed.
        """

        # Build filename
        short = _sat_short(satellite)
        last_day = calendar.monthrange(year, month)[1]
        start = f"{year}{month:02d}01"
        end = f"{year}{month:02d}{last_day:02d}"

        if instrument == "eps":
            filename = f"{short}_eps_5m_{start}_{end}.csv"
            url = build_eps_url(satellite, year, month)
        else:
            filename = f"{short}_epead_cpflux_5m_{start}_{end}.csv"
            url = build_epead_url(satellite, year, month)

        cache_path = Path(self.cache_dir) / str(year) / filename

        # Check cache
        if cache_path.exists():
            return parse_legacy_csv(cache_path)

        # Download
        print(f"  Downloading {filename}...")
        success = download_with_retry(url, cache_path)

        if not success:
            print(f"  WARNING: 404 for {filename}")
            return pd.DataFrame()

        return parse_legacy_csv(cache_path)

    # ------------------------------------------------------------------
    # INTERNAL: SGPS DATA (GOES-R)
    # ------------------------------------------------------------------

    def _fetch_sgps(self, year: int, month: int) -> pd.DataFrame:
        """
        Fetch GOES-16 SGPS daily NetCDF files for an entire month.

        Downloads each day's file, extracts time + flux using
        extract_sgps_flux(), concatenates into a DataFrame with
        columns: time_tag, sgps_integral_flux.

        Handles version discovery automatically — file version strings
        change across years (v1-0-1, v2-0-0, v3-0-0, v3-0-1, v3-0-2).
        """

        # Discover the version string for this month
        version = _discover_sgps_version(year, month)
        if version is None:
            print(f"  WARNING: Cannot discover SGPS version for "
                  f"{year}-{month:02d}, trying v3-0-2")
            version = "v3-0-2"

        last_day = calendar.monthrange(year, month)[1]
        all_times = []
        all_fluxes = []

        for day in range(1, last_day + 1):
            date_str = f"{year}{month:02d}{day:02d}"
            filename = f"sci_sgps-l2-avg5m_g16_d{date_str}_{version}.nc"
            cache_path = (
                Path(self.cache_dir) / "sgps" / str(year) / filename
            )

            # Download if not cached
            if not cache_path.exists():
                url = build_sgps_url(year, month, day, version=version)
                success = download_with_retry(url, cache_path)
                if not success:
                    continue

            # Extract time + flux from NetCDF
            try:
                t, f = extract_sgps_flux(cache_path)
                all_times.append(t)
                all_fluxes.append(f)
            except Exception as e:
                print(f"  WARNING: Failed {filename}: {e}")
                continue

        if not all_times:
            return pd.DataFrame()

        # Concatenate daily files into one DataFrame
        combined_time = all_times[0].append(all_times[1:])
        combined_flux = np.concatenate(all_fluxes)

        return pd.DataFrame({
            "time_tag": combined_time,
            "sgps_integral_flux": combined_flux,
        })

    def _parse_sgps_dataframe(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Parse the SGPS DataFrame created by _fetch_sgps().

        The flux was already derived from differential channels
        during fetch. This just cleans time and returns arrays.
        """

        time = pd.to_datetime(df["time_tag"])
        if time.dt.tz is not None:
            time = time.dt.tz_localize(None)

        time_index = pd.DatetimeIndex(time)
        flux = df["sgps_integral_flux"].to_numpy(dtype=np.float64)

        # Sort and deduplicate
        sort_idx = time_index.argsort()
        time_index = time_index[sort_idx]
        flux = flux[sort_idx]

        dup_mask = ~time_index.duplicated(keep="first")
        time_index = time_index[dup_mask]
        flux = flux[dup_mask]

        return time_index, flux