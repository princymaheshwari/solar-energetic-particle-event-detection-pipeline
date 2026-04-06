"""

SOHO COSTEP/EPHIN adapter for SEP detection.

Handles SOHO EPHIN Level-3 Intensities 5-minute data (1995-2025).
Annual CDF files from the CDAWeb archive.

Data structure:
  - 4 proton channels with documented energy ranges
    (from COSTEP-EPHIN L3 documentation, Table 3):
      P4:  4.3 – 7.8  MeV  (ΔE = 3.5 MeV)   — excluded (below 10 MeV)
      P8:  7.8 – 25.0 MeV  (ΔE = 17.2 MeV)   — partially included
      P25: 25.0 – 40.9 MeV (ΔE = 15.9 MeV)   — fully included
      P41: 40.9 – 53.0 MeV (ΔE = 12.1 MeV)   — fully included

  - P_int values are differential intensities in (cm² s sr MeV)⁻
    (confirmed by CDF metadata: UNITS = '(cm^2 s sr MeV)^-1').

  - To approximate >10 MeV integral flux, we multiply each channel's
    differential intensity by its effective energy width and sum:

      proxy ≈ I_P8 × 15.0 + I_P25 × 15.9 + I_P41 × 12.1

    P8 uses effective width 15.0 MeV (10.0–25.0) instead of full
    17.2 MeV (7.8–25.0) to exclude the sub-10 MeV portion under
    a flat-spectrum approximation within the channel.

  - This proxy covers 10–53 MeV, not 10 MeV to infinity. It is
    labeled as a proxy, not a true NOAA-style >10 MeV integral flux.
    This assumes constant differential flux within the P8 channel; SEP spectra typically decrease with energy, so this may slightly underestimate the contribution from 10–25 MeV


Detection uses the same three-rule logic as GOES:
  - Fixed threshold: 10 pfu
  - Rising gradient: 3/4 positive steps
  - Duration filter: 30 minutes minimum

External code just calls:
    adapter = SOHOAdapter(cache_dir="data/cache/soho")
    result = adapter.detect(year=2003)
"""

import os
import time as time_module
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import requests

from sep_core.adapters.base import BaseAdapter
from sep_core.detections import DetectionResult
from sep_core.threshold import detect_sep_events


# ==================================================================
# DOCUMENTED CHANNEL DEFINITIONS
# ==================================================================
# From COSTEP-EPHIN L3 documentation (Table 3) and CDF metadata.
#
# Channel  | E_min (MeV) | E_max (MeV) | ΔE (MeV) | Column index
# ---------+-------------+-------------+-----------+-------------
# P4  (E1) |    4.3      |    7.8      |    3.5    |     0
# P8  (E2) |    7.8      |   25.0      |   17.2    |     1
# P25 (E3) |   25.0      |   40.9      |   15.9    |     2
# P41 (E4) |   40.9      |   53.0      |   12.1    |     3
#
# P_int units: (cm² s sr MeV)⁻¹  — differential intensity.
# To get integral flux (pfu), multiply by ΔE and sum.

CHANNEL_EDGES = {
    "P4":  {"index": 0, "e_min":  4.3, "e_max":  7.8},
    "P8":  {"index": 1, "e_min":  7.8, "e_max": 25.0},
    "P25": {"index": 2, "e_min": 25.0, "e_max": 40.9},
    "P41": {"index": 3, "e_min": 40.9, "e_max": 53.0},
}

# Effective energy widths for the >10 MeV proxy.
# P4 is excluded entirely (fully below 10 MeV).
# P8 uses 10.0–25.0 = 15.0 MeV (flat-spectrum partial-bin correction).
# P25 and P41 use their full documented widths.
PROXY_EFFECTIVE_WIDTHS = {
    "P4":  0.0,    # excluded: entirely below 10 MeV
    "P8":  15.0,   # 25.0 - 10.0; excludes 7.8–10 MeV sub-bin
    "P25": 15.9,   # 40.9 - 25.0; full channel
    "P41": 12.1,   # 53.0 - 40.9; full channel
}


# ==================================================================
# ARCHIVE URL
# ==================================================================
# SOHO EPHIN L3I 5-min CDF files are hosted at CDAWeb.
# One file per year, organized as:
#   .../ephin_l3i-5min/{year}/soho_costep-ephin_l3i-5min_{year}0101_v{version}.cdf
#
# The version string varies by year (v01.01, v01.23, etc.),
# so we list the directory to find the actual filename.

CDAWEB_BASE_URL = (
    "https://cdaweb.gsfc.nasa.gov/pub/data/"
    "soho/costep/ephin_l3i-5min"
)


# ==================================================================
# HELPER FUNCTIONS
# ==================================================================

def find_cdf_filename(year: int) -> Optional[str]:
    """
    Find the actual CDF filename for a given year by listing
    the CDAWeb directory.

    The version string in the filename varies per year, so we
    can't construct the filename directly. Instead we fetch the
    directory listing and find the matching file.

    Parameters
    ----------
    year : int

    Returns
    -------
    str or None
        The filename if found, None otherwise.
    """

    dir_url = f"{CDAWEB_BASE_URL}/{year}/"

    try:
        resp = requests.get(dir_url, timeout=30)
        if resp.status_code != 200:
            return None

        # Parse the directory listing for the CDF filename
        # Files match: soho_costep-ephin_l3i-5min_{year}0101_v*.cdf
        prefix = f"soho_costep-ephin_l3i-5min_{year}0101"
        for line in resp.text.split('"'):
            if line.startswith(prefix) and line.endswith(".cdf"):
                return line

    except requests.RequestException:
        return None

    return None


def build_soho_url(year: int, filename: str) -> str:
    """Build full download URL for a SOHO CDF file."""
    return f"{CDAWEB_BASE_URL}/{year}/{filename}"


def download_with_retry(
    url: str,
    save_path: Path,
    max_retries: int = 3,
    retry_delay: float = 2.0
) -> bool:
    """
    Download a file with retry logic.
    Returns True on success, False on 404.
    """

    save_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(max_retries):
        try:
            resp = requests.get(url, timeout=120)

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
# CDF PARSING
# ==================================================================

def parse_soho_cdf(
    filepath: Path,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Parse a SOHO EPHIN Level-3 annual CDF file.

    Reads Epoch and P_int (proton differential intensities).
    Channel energy definitions are taken from the documentation
    (hardcoded in CHANNEL_EDGES), not inferred from the CDF.

    Parameters
    ----------
    filepath : Path
        Path to the CDF file.

    Returns
    -------
    tuple of (time, p_int)
        time : pd.DatetimeIndex — timestamps
        p_int : np.ndarray — proton differential intensities,
                shape (N, 4), units: (cm² s sr MeV)⁻¹
    """

    import cdflib
    from cdflib import epochs

    with cdflib.CDF(str(filepath)) as cdf:
        # Time — CDF epoch to datetime
        raw_time = cdf.varget("Epoch")
        time = epochs.CDFepoch.to_datetime(raw_time)

        # Proton intensities — shape (N, 4)
        # Columns: P4 (4.3-7.8), P8 (7.8-25), P25 (25-40.9), P41 (40.9-53)
        # Units: (cm² s sr MeV)⁻¹  [differential intensity]
        p_int = cdf.varget("P_int")

    # Build DatetimeIndex
    time_index = pd.DatetimeIndex(pd.to_datetime(time))
    if time_index.tz is not None:
        time_index = time_index.tz_localize(None)

    return time_index, np.asarray(p_int, dtype=float)


def compute_gt10mev_proxy(p_int: np.ndarray) -> np.ndarray:
    """
    Build an approximate >10 MeV integral flux proxy from SOHO channels.

    Multiplies each channel's differential intensity by its effective
    energy width and sums, using documented channel boundaries from
    the COSTEP-EPHIN L3 documentation.

    The P8 channel (7.8–25 MeV) uses an effective width of 15.0 MeV
    (10.0–25.0 MeV) instead of the full 17.2 MeV, applying a
    flat-spectrum approximation to exclude the sub-10 MeV portion.

    P25 and P41 use their full documented widths (15.9 and 12.1 MeV).
    P4 is excluded entirely (below 10 MeV).

    Result:
        proxy ≈ I_P8 × 15.0 + I_P25 × 15.9 + I_P41 × 12.1

    This covers approximately 10–53 MeV, not 10 MeV to infinity.

    Parameters
    ----------
    p_int : np.ndarray
        Proton differential intensities, shape (N, 4).
        Columns: P4, P8, P25, P41.
        Units: (cm² s sr MeV)⁻¹.

    Returns
    -------
    np.ndarray
        Proxy flux values, shape (N,).
        Units: (cm² s sr)⁻¹, i.e. pfu-comparable.
        Non-positive values replaced with NaN.
    """

    proxy = (
        p_int[:, 1] * PROXY_EFFECTIVE_WIDTHS["P8"]   # P8:  10.0–25.0 MeV
        + p_int[:, 2] * PROXY_EFFECTIVE_WIDTHS["P25"]  # P25: 25.0–40.9 MeV
        + p_int[:, 3] * PROXY_EFFECTIVE_WIDTHS["P41"]  # P41: 40.9–53.0 MeV
    )

    # Non-positive values are unphysical — replace with NaN
    proxy[proxy <= 0] = np.nan

    return proxy


# ==================================================================
# THE ADAPTER CLASS
# ==================================================================

class SOHOAdapter(BaseAdapter):
    """
    SOHO COSTEP/EPHIN adapter for SEP detection.

    Handles annual CDF files from 1995-2025.
    Builds a 10–53 MeV proton flux proxy from channels P8, P25, P41.
    Uses same three-rule detection as GOES (threshold=10 pfu).

    The proxy approximates NOAA-style >10 MeV integral proton flux
    using documented channel boundaries and a flat-spectrum partial-bin
    correction for P8. See compute_gt10mev_proxy() for details.

    Usage:
        adapter = SOHOAdapter(cache_dir="data/cache/soho")

        # Full year (natural unit for SOHO — annual CDF files)
        result = adapter.detect(year=2003)

        # Single month (extracts from annual data)
        result = adapter.detect(year=2003, month=10)
    """

    def __init__(self, cache_dir: str = "data/cache/soho"):
        super().__init__(name="SOHO/EPHIN", cache_dir=cache_dir)

    # ------------------------------------------------------------------
    # ABSTRACT METHOD IMPLEMENTATIONS
    # ------------------------------------------------------------------

    def fetch_data(
        self, year: int, month: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Download or load cached SOHO CDF data for a given year.

        SOHO data is annual — one CDF file per year. If month is
        specified, the full year is still fetched (and cached), but
        parse_flux() will filter to the requested month.

        Returns empty DataFrame if data is unavailable.
        """

        # Check cache for any matching file
        cache_dir = Path(self.cache_dir) / str(year)
        cached_file = self._find_cached_cdf(cache_dir, year)

        if cached_file is not None:
            return self._cdf_to_dataframe(cached_file, month)

        # Not cached — find filename from directory listing
        print(f"  Looking up SOHO CDF filename for {year}...")
        filename = find_cdf_filename(year)

        if filename is None:
            print(f"  WARNING: No SOHO CDF found for {year}")
            return pd.DataFrame()

        cache_path = cache_dir / filename

        # Download
        url = build_soho_url(year, filename)
        print(f"  Downloading {filename}...")
        success = download_with_retry(url, cache_path)

        if not success:
            print(f"  WARNING: Failed to download {filename}")
            return pd.DataFrame()

        return self._cdf_to_dataframe(cache_path, month)

    def parse_flux(
        self, raw_data: pd.DataFrame
    ) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Extract clean time and proxy flux arrays from SOHO data.

        The DataFrame was created by _cdf_to_dataframe() and contains
        pre-computed proxy flux in the "soho_gt10_proxy" column.
        """

        if "soho_gt10_proxy" not in raw_data.columns:
            raise ValueError(
                f"SOHO data missing 'soho_gt10_proxy'. "
                f"Columns: {list(raw_data.columns)}"
            )

        time = pd.to_datetime(raw_data["time"])
        if time.dt.tz is not None:
            time = time.dt.tz_localize(None)

        time_index = pd.DatetimeIndex(time)
        flux = raw_data["soho_gt10_proxy"].to_numpy(dtype=np.float64)

        # Sort and deduplicate
        sort_idx = time_index.argsort()
        time_index = time_index[sort_idx]
        flux = flux[sort_idx]

        dup_mask = ~time_index.duplicated(keep="first")
        time_index = time_index[dup_mask]
        flux = flux[dup_mask]

        return time_index, flux

    def get_threshold_params(self) -> Dict[str, Any]:
        """
        Return SOHO detection parameters.

        Same detection logic as GOES:
        - Entry threshold: 10 pfu (proxy reaches pfu-comparable values)
        - Exit threshold: 5 pfu (hysteresis)
        - Quiet period: 2 hours (24 × 5-min points)
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
    # DETECT (overrides base class)
    # ------------------------------------------------------------------

    def detect(
        self, year: int, month: Optional[int] = None
    ) -> DetectionResult:
        """
        Run full SOHO SEP detection for a year or single month.

        SOHO data is annual, so a full year is always fetched.
        If month is specified, the result is filtered to that month.
        """

        # Step 1: Fetch (downloads annual CDF, filters to month if needed)
        raw_data = self.fetch_data(year, month)

        if raw_data.empty:
            return self._empty_result(year, month)

        # Step 2: Parse
        time, flux = self.parse_flux(raw_data)
        self.validate_time_series(time, flux)

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
            instrument=self.name,
            time=time,
            flux=flux,
            mask=sep_mask,
            metadata={
                "year": year,
                "month": month,
                "threshold": params,
                "proxy_method": (
                    "Flat-spectrum partial-bin correction. "
                    "P8: 15.0 MeV (10–25), P25: 15.9 MeV, P41: 12.1 MeV. "
                    "Covers ~10–53 MeV."
                ),
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

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    def _find_cached_cdf(
        self, cache_dir: Path, year: int
    ) -> Optional[Path]:
        """
        Check if a CDF file for this year already exists in the cache.

        Since the version string varies, we search for any file
        matching the expected prefix.
        """

        if not cache_dir.exists():
            return None

        prefix = f"soho_costep-ephin_l3i-5min_{year}0101"
        for f in cache_dir.iterdir():
            if f.name.startswith(prefix) and f.name.endswith(".cdf"):
                return f

        return None

    def _cdf_to_dataframe(
        self, filepath: Path, month: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Read a SOHO CDF file, compute the >10 MeV proxy, and return
        as a DataFrame.

        If month is specified, filters to that month only.

        Parameters
        ----------
        filepath : Path
            Path to the CDF file.
        month : int or None
            If specified, filter to this month (1-12).

        Returns
        -------
        pd.DataFrame
            Columns: time, soho_gt10_proxy, plus individual channels.
        """

        try:
            time_index, p_int = parse_soho_cdf(filepath)
        except Exception as e:
            print(f"  WARNING: Failed to parse {filepath.name}: {e}")
            return pd.DataFrame()

        # Compute >10 MeV proxy using documented channel widths
        proxy = compute_gt10mev_proxy(p_int)

        # Build DataFrame
        df = pd.DataFrame({
            "time": time_index,
            "soho_gt10_proxy": proxy,
            "p_int_P4": p_int[:, 0],
            "p_int_P8": p_int[:, 1],
            "p_int_P25": p_int[:, 2],
            "p_int_P41": p_int[:, 3],
        })

        # Filter to requested month if specified
        if month is not None:
            df = df[pd.to_datetime(df["time"]).dt.month == month].copy()

        if df.empty:
            return pd.DataFrame()

        return df