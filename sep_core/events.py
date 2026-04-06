"""

Converts pointwise boolean detection masks into event intervals,
and event intervals back into pointwise masks.

This module sits between the adapters (which produce pointwise masks)
and the fusion engine (which merges intervals from multiple instruments).

Interval construction lives HERE, not in adapters. This means every
instrument's detections are converted to intervals using the same logic.

Flow in the pipeline:
    Adapter → DetectionResult (pointwise mask)
        → events.py (mask → intervals)
            → fusion.py (merge intervals across instruments)
                → events.py (intervals → mask, for evaluation)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List


@dataclass
class Event:
    """
    A single detected SEP event interval.

    Attributes
    ----------
    start_time : pd.Timestamp
        Start time of the event (first detected timestamp).
    end_time : pd.Timestamp
        End time of the event (last detected timestamp).
    duration_minutes : float
        Duration of the event in minutes.
    """

    start_time: pd.Timestamp
    end_time: pd.Timestamp
    duration_minutes: float


def extract_events(
    time: pd.DatetimeIndex,
    mask: np.ndarray,
    cadence_minutes: int = 5
) -> List[Event]:
    """
    Convert a boolean detection mask into event intervals.

    Scans through the mask, finds contiguous runs of True values,
    and creates one Event for each run.

    This is the equivalent of your original script's
    boolean_runs_to_intervals() function.

    Parameters
    ----------
    time : pd.DatetimeIndex
        Time axis aligned with the mask.
    mask : np.ndarray
        Boolean detection mask from a DetectionResult.
    cadence_minutes : int
        Time resolution of the data in minutes. Default 5.
        Used to compute duration from point count.

    Returns
    -------
    List[Event]
        One Event per contiguous run of True values.
        Empty list if no detections exist.
    """

    if len(mask) == 0 or not mask.any():
        return []

    # Find transitions using diff on integer mask
    # +1 = False→True (event starts), -1 = True→False (event ends)
    changes = np.diff(mask.astype(int))

    starts = np.where(changes == 1)[0] + 1    # +1 corrects diff offset
    ends = np.where(changes == -1)[0] + 1

    # Edge case: mask starts with True — no False→True transition
    if mask[0]:
        starts = np.insert(starts, 0, 0)

    # Edge case: mask ends with True — no True→False transition
    if mask[-1]:
        ends = np.append(ends, len(mask))

    # Build one Event per run
    events = []
    for s, e in zip(starts, ends):
        start_time = time[s]
        end_time = time[e - 1]       # e is exclusive; last True is e-1
        n_points = e - s
        duration = n_points * cadence_minutes

        events.append(Event(
            start_time=start_time,
            end_time=end_time,
            duration_minutes=duration
        ))

    return events


def merge_close_events(
    events: List[Event],
    gap_minutes: float = 30.0,
    cadence_minutes: int = 5
) -> List[Event]:
    """
    Merge events separated by a gap smaller than gap_minutes.

    SEP events sometimes show brief dips below threshold before
    the flux rises again. This produces two separate events that
    are physically one event. Merging close events fixes this.

    This is a single-instrument operation. Multi-instrument
    fusion is handled separately in fusion.py.

    Equivalent to the merge logic in your original script's
    merge_and_label_intervals() but for one instrument only.

    Parameters
    ----------
    events : List[Event]
        Events from one instrument, from extract_events().
    gap_minutes : float
        Maximum gap in minutes to merge across. Default 30.0.
    cadence_minutes : int
        Time resolution. Default 5. Used for duration recalculation.

    Returns
    -------
    List[Event]
        Merged events. Count <= input count.
    """

    if len(events) <= 1:
        return list(events)

    # Sort by start time
    sorted_events = sorted(events, key=lambda ev: ev.start_time)
    max_gap = pd.Timedelta(minutes=gap_minutes)

    merged = []
    current = sorted_events[0]

    for i in range(1, len(sorted_events)):
        next_ev = sorted_events[i]

        gap = next_ev.start_time - current.end_time

        if gap <= max_gap:
            # Merge: extend current to cover next
            new_end = max(current.end_time, next_ev.end_time)
            new_duration = (
                (new_end - current.start_time).total_seconds() / 60.0
                + cadence_minutes  # include the end timestamp itself
            )

            current = Event(
                start_time=current.start_time,
                end_time=new_end,
                duration_minutes=new_duration
            )
        else:
            # Gap too large — finalize current, move to next
            merged.append(current)
            current = next_ev

    # Don't forget the last event
    merged.append(current)

    return merged


def events_to_dataframe(events: List[Event]) -> pd.DataFrame:
    """
    Convert a list of Events to a pandas DataFrame.

    Useful for saving to CSV, displaying, and passing to fusion.py.

    Parameters
    ----------
    events : List[Event]
        Events from extract_events() or merge_close_events().

    Returns
    -------
    pd.DataFrame
        Columns: start_time, end_time, duration_minutes.
        Empty DataFrame with correct columns if input is empty.
    """

    if not events:
        return pd.DataFrame(
            columns=["start_time", "end_time", "duration_minutes"]
        )

    return pd.DataFrame([
        {
            "start_time": ev.start_time,
            "end_time": ev.end_time,
            "duration_minutes": ev.duration_minutes,
        }
        for ev in events
    ])


def events_to_mask(
    events: List[Event],
    time: pd.DatetimeIndex
) -> np.ndarray:
    """
    Expand event intervals back into a pointwise boolean mask.

    The inverse of extract_events(). Needed for:
    - Creating fused pointwise labels after fusion
    - Comparing interval-level results with pointwise ground truth
    - Evaluation against NOAA SEP catalog

    Equivalent to your original script's intervals_to_point_labels().

    Parameters
    ----------
    events : List[Event]
        Event intervals to expand.
    time : pd.DatetimeIndex
        The time axis to create the mask on.

    Returns
    -------
    np.ndarray
        Boolean mask. True for timestamps inside any event.
    """

    mask = np.zeros(len(time), dtype=bool)

    if not events:
        return mask

    for ev in events:
        inside = (time >= ev.start_time) & (time <= ev.end_time)
        mask |= inside

    return mask