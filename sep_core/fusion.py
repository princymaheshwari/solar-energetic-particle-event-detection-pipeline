"""

Multi-instrument fusion engine for SEP event detection.

Takes event intervals from two or more instruments and produces
a unified event catalog. This module is completely source-independent —
it never knows whether the intervals came from GOES, SOHO, STEREO,
or any other instrument. It only sees Event objects with start/end times.

Handles the three interval-level edge cases:
1. Overlapping intervals — partially overlapping detections from
   different instruments. Merged into one fused event.
2. Close intervals — detections separated by a small gap
   (<= gap_minutes). Merged into one fused event.
3. Nested intervals — one instrument detects a long event, another
   detects a shorter interval fully inside it. One fused event.

All three are handled by a single merge condition: sort by start time,
then merge any interval whose start falls within
(current_end + gap_tolerance) of the current interval.

Flow in the pipeline:
    Adapter A → DetectionResult → events.py → List[Event]
    Adapter B → DetectionResult → events.py → List[Event]
        → fusion.py (merge across instruments)
            → List[FusedEvent] (unified catalog)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

from sep_core.events import Event


@dataclass
class FusedEvent:
    """
    A single fused SEP event from multiple instruments.

    Attributes
    ----------
    start_time : pd.Timestamp
        Start time of the fused event (earliest across instruments).
    end_time : pd.Timestamp
        End time of the fused event (latest across instruments).
    duration_minutes : float
        Duration of the fused event in minutes.
    instruments : List[str]
        Sorted list of instruments that contributed to this event.
        Examples: ["GOES-8/EPS"], ["GOES-8/EPS", "SOHO/EPHIN"]
    """

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_minutes: float
    instruments: List[str]


def fuse_events(
    events_by_instrument: dict,
    gap_minutes: float = 30.0
) -> List[FusedEvent]:
    """
    Merge event intervals from multiple instruments into a unified catalog.

    Takes events from each instrument, tags them with their source,
    pools them together, sorts by start time, and merges intervals
    that overlap, are close, or are nested.

    The algorithm is identical to your original script's
    merge_and_label_intervals(), generalized to handle any number
    of instruments.

    Parameters
    ----------
    events_by_instrument : dict
        Keys are instrument names (str), values are List[Event].
        Example:
            {
                "GOES-8/EPS": [Event(...), Event(...)],
                "SOHO/EPHIN": [Event(...)]
            }
    gap_minutes : float
        Maximum gap in minutes between two intervals for them
        to be merged. Default 30.0 minutes.

    Returns
    -------
    List[FusedEvent]
        Unified event catalog sorted by start time.
        Each FusedEvent records which instruments contributed.
        Empty list if no events from any instrument.
    """

    # Step 1: Pool all events, tagged with source instrument
    all_events = []
    for instrument, events in events_by_instrument.items():
        for ev in events:
            all_events.append({
                "start": ev.start_time,
                "end": ev.end_time,
                "instrument": instrument
            })

    if not all_events:
        return []

    # Step 2: Sort by start time, then by end time
    all_events.sort(key=lambda x: (x["start"], x["end"]))

    # Step 3: Walk through and merge
    max_gap = pd.Timedelta(minutes=gap_minutes)
    fused = []

    current_start = all_events[0]["start"]
    current_end = all_events[0]["end"]
    current_instruments = {all_events[0]["instrument"]}

    for i in range(1, len(all_events)):
        ev = all_events[i]

        # This single condition handles all 3 edge cases:
        #
        # Case 1 — Overlap:
        #   GOES [10:00–11:00], SOHO [10:30–11:30]
        #   gap = 10:30 - 11:00 = -30min → negative → merge
        #
        # Case 2 — Close gap:
        #   GOES [10:00–11:00], SOHO [11:20–12:00]
        #   gap = 11:20 - 11:00 = 20min → ≤ 30min → merge
        #
        # Case 3 — Nested:
        #   GOES [10:00–14:00], SOHO [11:00–12:00]
        #   gap = 11:00 - 14:00 = -3h → negative → merge
        #   current_end stays at max(14:00, 12:00) = 14:00

        if ev["start"] <= current_end + max_gap:
            # Merge: extend end, add instrument
            current_end = max(current_end, ev["end"])
            current_instruments.add(ev["instrument"])
        else:
            # Gap too large — finalize current event
            duration = (
                (current_end - current_start).total_seconds() / 60.0
            )
            fused.append(FusedEvent(
                start_time=current_start,
                end_time=current_end,
                duration_minutes=duration,
                instruments=sorted(current_instruments)
            ))

            # Start new event
            current_start = ev["start"]
            current_end = ev["end"]
            current_instruments = {ev["instrument"]}

    # Finalize the last event
    duration = (current_end - current_start).total_seconds() / 60.0
    fused.append(FusedEvent(
        start_time=current_start,
        end_time=current_end,
        duration_minutes=duration,
        instruments=sorted(current_instruments)
    ))

    return fused


def fused_events_to_dataframe(
    fused: List[FusedEvent]
) -> pd.DataFrame:
    """
    Convert fused events to a pandas DataFrame.

    The instruments list is joined into a comma-separated string
    for clean display and CSV export.

    Parameters
    ----------
    fused : List[FusedEvent]
        Fused events from fuse_events().

    Returns
    -------
    pd.DataFrame
        Columns: start_time, end_time, instruments, duration_minutes,
                 n_instruments.
    """

    if not fused:
        return pd.DataFrame(
            columns=["start_time", "end_time", "instruments",
                     "duration_minutes", "n_instruments"]
        )

    return pd.DataFrame([
        {
            "start_time": fe.start_time,
            "end_time": fe.end_time,
            "instruments": ",".join(fe.instruments),
            "duration_minutes": fe.duration_minutes,
            "n_instruments": len(fe.instruments),
        }
        for fe in fused
    ])


def compute_support_labels(
    fused: List[FusedEvent],
    time: pd.DatetimeIndex,
    instrument_names: List[str]
) -> pd.Series:
    """
    Create a pointwise instrument support label for each timestamp.

    For each timestamp, indicates which instruments detected an event.
    Possible values:
    - "none" — no event at this timestamp
    - "{name}_only" — only one instrument detected (e.g., "GOES-8/EPS_only")
    - "both" — all instruments detected
    - comma-separated names — subset of instruments (for 3+ instruments)

    Equivalent to your original script's instrument_support column.

    Parameters
    ----------
    fused : List[FusedEvent]
        Fused events from fuse_events().
    time : pd.DatetimeIndex
        Common time axis.
    instrument_names : List[str]
        All instrument names involved.

    Returns
    -------
    pd.Series
        String labels indexed by time.
    """

    labels = pd.Series("none", index=time, dtype=object)

    for fe in fused:
        inside = (time >= fe.start_time) & (time <= fe.end_time)

        if len(fe.instruments) == 1:
            label = f"{fe.instruments[0]}_only"
        elif len(fe.instruments) >= len(instrument_names):
            label = "both"
        else:
            label = ",".join(fe.instruments)

        labels[inside] = label

    return labels