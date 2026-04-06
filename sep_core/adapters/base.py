"""
Defines the abstract base class that every instrument adapter must follow.
This is the interface contract — it tells any developer exactly what methods
they need to implement and what those methods must return.

If DetectionResult is "what the output looks like",
then BaseAdapter is "what the adapter must do to produce it."
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from sep_core.detections import DetectionResult


class BaseAdapter(ABC):
    """
    Abstract base class for all instrument adapters.

    Every adapter (GOES, SOHO, future STEREO, etc.) must inherit from this
    and implement all abstract methods.

    The adapter's job is to:
    1. Fetch raw data (from an API, archive URL, or local cache)
    2. Parse it into a clean time series (time + flux)
    3. Apply instrument-specific thresholding to produce a detection mask
    4. Return a DetectionResult

    The adapter encapsulates ALL instrument-specific knowledge:
    - file formats, column names, missing value conventions
    - sensor selection, proxy construction, satellite mapping
    - threshold logic (fixed for GOES and SOHO)

    Nothing outside the adapter needs to know any of this.

    Parameters
    ----------
    name : str
        Human-readable instrument identifier.
        Examples: "GOES-8/EPS", "GOES-13/EPEAD", "SOHO/EPHIN"
        Stored in every DetectionResult this adapter produces.
    cache_dir : str
        Local directory for caching downloaded files.
        If a file already exists here, the adapter skips the download.
    """

    def __init__(self, name: str, cache_dir: str = "data/cache"):
        self.name = name
        self.cache_dir = cache_dir

    # ------------------------------------------------------------------
    # ABSTRACT METHODS — every subclass MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_data(self, year: int, month: Optional[int] = None) -> pd.DataFrame:
        """
        Download or load raw instrument data for a given time period.

        Handles:
        - Checking if data is already cached locally
        - If not cached: downloading from the remote archive
        - Saving downloaded data to cache_dir
        - Reading the data into a pandas DataFrame

        The returned DataFrame contains raw instrument data with
        instrument-specific column names. It does NOT need to be
        fully cleaned — that happens in parse_flux().

        Parameters
        ----------
        year : int
            The year to fetch data for (e.g., 2003).
        month : int or None
            The month to fetch (1-12). None means fetch the full year.
            GOES adapters use month (monthly CSV files).
            SOHO adapter may use None (annual CDF files).

        Returns
        -------
        pd.DataFrame
            Raw instrument data. Returns empty DataFrame if unavailable.
        """
        pass

    @abstractmethod
    def parse_flux(self, raw_data: pd.DataFrame) -> Tuple[pd.DatetimeIndex, np.ndarray]:
        """
        Extract and clean the time and flux arrays from raw data.

        Handles all instrument-specific parsing:
        - Selecting the correct flux column (p3_flux_ic, ZPGT10E, etc.)
        - Selecting the correct sensor (east/west for EPEAD)
        - Replacing missing value flags (-99999) with NaN
        - Filtering by quality flags where applicable
        - Building a flux proxy from differential channels (SGPS, SOHO)
        - Converting timestamps to pd.DatetimeIndex (UTC, naive, sorted)

        Parameters
        ----------
        raw_data : pd.DataFrame
            The raw DataFrame returned by fetch_data().

        Returns
        -------
        tuple of (pd.DatetimeIndex, np.ndarray)
            time : cleaned, sorted, UTC, timezone-naive timestamps
            flux : float64 flux values, NaN for missing. Same length as time.
        """
        pass

    @abstractmethod
    def get_threshold_params(self) -> Dict[str, Any]:
        """
        Return the thresholding parameters for this instrument.

        Returns
        -------
        dict
            Must contain at least:
            - "type": "fixed" or "dynamic"

            For "fixed":
            - "value": float (e.g., 10.0 for GOES)

            For "dynamic":
            - "method": str (e.g., "mad")
            - "k": float (multiplier)
            - "log_space": bool

            Optional:
            - "persistence_minutes": int (minimum detection duration)
            - "smoothing_window": int (rolling median window size)
        """
        pass

    # ------------------------------------------------------------------
    # CONCRETE METHODS — shared logic, same for every adapter
    # ------------------------------------------------------------------

    def detect(
        self, year: int, month: Optional[int] = None
    ) -> DetectionResult:
        """
        Run the full detection pipeline for one time period.

        This is the main entry point that experiment scripts call.
        It orchestrates the abstract methods in sequence:
        1. fetch_data()  — get raw data (download or cache)
        2. parse_flux()  — extract clean time and flux
        3. threshold      — produce boolean detection mask
        4. return DetectionResult

        This method is NOT abstract — the overall algorithm is fixed.
        Instrument-specific behavior lives in the abstract methods.
        This is the Template Method pattern.

        Parameters
        ----------
        year : int
            Year to run detection for.
        month : int or None
            Month (1-12), or None for full year.

        Returns
        -------
        DetectionResult
            Standardized detection output, ready for fusion/evaluation.
        """

        # Step 1: Get raw data
        raw_data = self.fetch_data(year, month)

        if raw_data.empty:
            return self._empty_result(year, month)

        # Step 2: Parse into clean time + flux
        time, flux = self.parse_flux(raw_data)

        # Validate before proceeding
        self.validate_time_series(time, flux)

        # Step 3: Apply threshold to produce detection mask
        threshold_params = self.get_threshold_params()
        mask = self._apply_threshold(flux, threshold_params)

        # Step 4: Build and return DetectionResult
        return DetectionResult(
            instrument=self.name,
            time=time,
            flux=flux,
            mask=mask,
            metadata={
                "year": year,
                "month": month,
                "threshold": threshold_params,
                "n_raw_rows": len(raw_data),
                "n_valid_flux": int(np.sum(~np.isnan(flux))),
                "status": "ok"
            }
        )

    def _apply_threshold(
        self, flux: np.ndarray, params: Dict[str, Any]
    ) -> np.ndarray:
        """
        Apply thresholding logic to produce a boolean detection mask.

        Handles the "fixed" threshold case directly.
        For "dynamic" thresholds, subclasses should override this method.

        Parameters
        ----------
        flux : np.ndarray
            Cleaned flux array. May contain NaN values.
        params : dict
            Threshold parameters from get_threshold_params().

        Returns
        -------
        np.ndarray
            Boolean mask. True where flux exceeds threshold.
            NaN flux values produce False.
        """

        threshold_type = params.get("type", "fixed")

        if threshold_type == "fixed":
            value = params["value"]
            # NaN >= value evaluates to False, which is correct
            mask = flux >= value

        elif threshold_type == "dynamic":
            # Subclasses (e.g., SOHO) override this method.
            # If they forget, this error reminds them.
            raise NotImplementedError(
                "Dynamic thresholding not implemented in base adapter. "
                "Override _apply_threshold() in your adapter subclass."
            )
        else:
            raise ValueError(f"Unknown threshold type: {threshold_type}")

        # Apply persistence filter if specified
        persistence = params.get("persistence_minutes")
        if persistence is not None:
            mask = self._apply_persistence(mask, persistence)

        return mask

    def _apply_persistence(
        self,
        mask: np.ndarray,
        min_minutes: int,
        cadence_minutes: int = 5
    ) -> np.ndarray:
        """
        Filter out detections shorter than a minimum duration.

        A single 5-minute spike above threshold is likely noise.
        Real SEP events are sustained enhancements. This removes
        short runs of True values in the mask.

        Parameters
        ----------
        mask : np.ndarray
            Boolean detection mask from thresholding.
        min_minutes : int
            Minimum duration in minutes for a detection to be kept.
        cadence_minutes : int
            Time resolution of the data in minutes. Default 5.

        Returns
        -------
        np.ndarray
            Filtered mask. Short True-runs set to False.
        """

        min_points = max(1, min_minutes // cadence_minutes)
        filtered = mask.copy()

        # Find where the mask transitions between False and True
        changes = np.diff(mask.astype(int))
        starts = np.where(changes == 1)[0] + 1   # False → True
        ends = np.where(changes == -1)[0] + 1     # True → False

        # Edge case: mask starts with True
        if mask[0]:
            starts = np.insert(starts, 0, 0)
        # Edge case: mask ends with True
        if mask[-1]:
            ends = np.append(ends, len(mask))

        # Remove runs shorter than min_points
        for s, e in zip(starts, ends):
            if (e - s) < min_points:
                filtered[s:e] = False

        return filtered

    def _empty_result(
        self, year: int, month: Optional[int]
    ) -> DetectionResult:
        """
        Create a valid but empty DetectionResult when no data is available.

        This prevents the pipeline from crashing when a particular
        month has no data (e.g., before a satellite launched, or
        a gap in the archive).

        Parameters
        ----------
        year : int
            The year that was requested.
        month : int or None
            The month that was requested.

        Returns
        -------
        DetectionResult
            Empty but valid result with status "no_data" in metadata.
        """
        return DetectionResult(
            instrument=self.name,
            time=pd.DatetimeIndex([], dtype="datetime64[ns]"),
            flux=np.array([], dtype=np.float64),
            mask=np.array([], dtype=np.bool_),
            metadata={
                "year": year,
                "month": month,
                "status": "no_data"
            }
        )

    # ------------------------------------------------------------------
    # SHARED HELPERS
    # ------------------------------------------------------------------

    def validate_time_series(
        self, time: pd.DatetimeIndex, flux: np.ndarray
    ) -> None:
        """
        Validate time and flux arrays before detection.

        Called inside detect() after parse_flux() returns.
        Catches problems early — before they reach DetectionResult
        validation or, worse, silently corrupt fusion results.

        Parameters
        ----------
        time : pd.DatetimeIndex
            The time axis to validate.
        flux : np.ndarray
            The flux array to validate.

        Raises
        ------
        TypeError
            If time is not a DatetimeIndex.
        ValueError
            If time is unsorted, has duplicates, or length mismatches flux.
        """

        if not isinstance(time, pd.DatetimeIndex):
            raise TypeError(
                f"time must be pd.DatetimeIndex, got {type(time)}"
            )

        if not time.is_monotonic_increasing:
            raise ValueError("time must be sorted in ascending order")

        if time.has_duplicates:
            raise ValueError("time contains duplicate timestamps")

        if len(time) != len(flux):
            raise ValueError(
                f"time length ({len(time)}) != flux length ({len(flux)})"
            )

    def __repr__(self) -> str:
        """Readable string representation for debugging and logging."""
        return f"{self.__class__.__name__}(name='{self.name}')"