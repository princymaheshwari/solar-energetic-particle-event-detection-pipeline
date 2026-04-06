"""
sep_core/detections.py

Defines the standardized data contract for the SEP detection pipeline.
Every adapter must produce a DetectionResult. The fusion engine, event builder,
and evaluation code all consume DetectionResults. This is the single agreement
that keeps the entire system source-independent.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


@dataclass
class DetectionResult:
    """
    The standardized output that every instrument adapter must return.

    This is the boundary between instrument-specific code (adapters) and
    instrument-agnostic code (fusion, events, evaluation). Nothing outside
    of an adapter should ever need to know whether the data came from
    GOES EPS, GOES EPEAD, GOES SGPS, or SOHO EPHIN.

    Attributes
    ----------
    instrument : str
        Human-readable identifier for the source instrument.
        Examples: "GOES-8/EPS", "GOES-13/EPEAD", "GOES-16/SGPS", "SOHO/EPHIN"
        Used for labeling in plots, logs, and fusion support tags.
        Not used for any logic — just identification.

    time : pd.DatetimeIndex
        The time axis. Must be UTC, timezone-naive, sorted ascending,
        no duplicates. This is the pipeline's internal time standard.

    flux : np.ndarray
        Proton flux time series, one value per timestamp.
        For GOES: >10 MeV integral proton flux in pfu.
        For SOHO: >10 MeV proxy flux (different scale).
        NaN indicates missing or flagged data.
        Shape: (len(time),), dtype: float64

    mask : np.ndarray
        Pointwise boolean detection mask.
        True = "this timestamp is part of an SEP enhancement."
        Output of threshold + persistence filtering inside the adapter.
        Shape: (len(time),), dtype: bool

    metadata : Dict[str, Any]
        Flexible dictionary recording what the adapter did.
        Not consumed by downstream logic — exists for reproducibility.
        Typical keys: threshold_type, threshold_value, persistence_minutes,
        smoothing_window, satellite, file_type, flux_column, sensor.
    """

    instrument: str
    time: pd.DatetimeIndex
    flux: np.ndarray
    mask: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Runs automatically right after the object is created.
        Coerces inputs to correct types where safe, then validates
        that everything meets the contract. If anything is wrong,
        raises an error immediately — at the adapter boundary —
        rather than letting bad data propagate into fusion or evaluation.
        """

        # --- Coerce flux and mask to numpy arrays ---
        # This is a convenience: if an adapter accidentally passes a list
        # or a pandas Series, we convert it rather than crashing.
        self.flux = np.asarray(self.flux, dtype=np.float64)
        self.mask = np.asarray(self.mask, dtype=np.bool_)

        # --- Now validate everything ---
        self._validate()

    def _validate(self):
        """
        Check that all fields meet the contract requirements.
        Raises TypeError or ValueError with a clear message if anything
        is wrong. Grouped by category for readability.
        """

        # ---- Type checks ----

        if not isinstance(self.instrument, str) or len(self.instrument.strip()) == 0:
            raise TypeError(
                f"instrument must be a non-empty string, "
                f"got: {repr(self.instrument)}"
            )

        if not isinstance(self.time, pd.DatetimeIndex):
            raise TypeError(
                f"time must be a pd.DatetimeIndex, "
                f"got: {type(self.time)}"
            )

        # ---- Time axis checks ----

        if self.time.tz is not None:
            raise ValueError(
                f"time must be timezone-naive (UTC assumed), "
                f"but has tzinfo: {self.time.tz}. "
                f"Use time.tz_localize(None) to remove timezone."
            )

        if not self.time.is_monotonic_increasing:
            raise ValueError(
                "time must be sorted in ascending order. "
                "Use time.sort_values() before creating DetectionResult."
            )

        if self.time.has_duplicates:
            raise ValueError(
                "time must not contain duplicate timestamps. "
                "Use time.drop_duplicates() before creating DetectionResult."
            )

        # ---- Shape consistency ----

        n = len(self.time)

        if self.flux.shape != (n,):
            raise ValueError(
                f"flux shape {self.flux.shape} does not match "
                f"time length {n}. Expected shape: ({n},)"
            )

        if self.mask.shape != (n,):
            raise ValueError(
                f"mask shape {self.mask.shape} does not match "
                f"time length {n}. Expected shape: ({n},)"
            )

    # ---- Convenience properties ----

    @property
    def n_timestamps(self) -> int:
        """Total number of timestamps in this detection."""
        return len(self.time)

    @property
    def n_detected(self) -> int:
        """Number of timestamps flagged as SEP detections."""
        return int(self.mask.sum())

    @property
    def detection_fraction(self) -> float:
        """Fraction of timestamps flagged as detections (0.0 to 1.0)."""
        if self.n_timestamps == 0:
            return 0.0
        return self.n_detected / self.n_timestamps

    @property
    def time_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """The (start, end) timestamps of this detection's time axis."""
        if self.n_timestamps == 0:
            return (pd.NaT, pd.NaT)
        return (self.time[0], self.time[-1])

    @property
    def has_detections(self) -> bool:
        """Whether any timestamps were flagged as SEP detections."""
        return self.n_detected > 0

    @property
    def valid_flux_fraction(self) -> float:
        """
        Fraction of flux values that are not NaN.
        This is a data quality indicator. 
        If it's 95%, you know 5% of the data was missing or flagged.
        """
        if self.n_timestamps == 0:
            return 0.0
        return 1.0 - (np.isnan(self.flux).sum() / self.n_timestamps)

    # ---- Output methods ----

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame for debugging, plotting, or CSV export.

        Returns a DataFrame with columns: time, flux, mask
        where time is set as the index for easy time-based operations.
        """
        return pd.DataFrame(
            {"flux": self.flux, "mask": self.mask},
            index=self.time
        )

    def summary(self) -> str:
        """
        Return a human-readable summary string for quick inspection.
        Useful for printing during experiment runs and debugging.

        Example output:
            DetectionResult(
              instrument    = GOES-8/EPS
              time_range    = 1997-05-01 00:00:00 to 1997-05-31 23:55:00
              n_timestamps  = 8928
              n_detected    = 42 (0.5%)
              valid_flux    = 99.5%
              flux_range    = 0.1060 to 1.6500
              metadata_keys = ['threshold_type', 'threshold_value']
            )
        """
        if self.n_timestamps == 0:
            return (
                f"DetectionResult(\n"
                f"  instrument    = {self.instrument}\n"
                f"  EMPTY — no timestamps\n"
                f")"
            )

        start, end = self.time_range
        return (
            f"DetectionResult(\n"
            f"  instrument    = {self.instrument}\n"
            f"  time_range    = {start} to {end}\n"
            f"  n_timestamps  = {self.n_timestamps}\n"
            f"  n_detected    = {self.n_detected} "
            f"({self.detection_fraction:.1%})\n"
            f"  valid_flux    = {self.valid_flux_fraction:.1%}\n"
            f"  flux_range    = {np.nanmin(self.flux):.4f} to "
            f"{np.nanmax(self.flux):.4f}\n"
            f"  metadata_keys = {list(self.metadata.keys())}\n"
            f")"
        )