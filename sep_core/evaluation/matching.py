"""
sep_core/evaluation/matching.py

Event-level evaluation for SEP detection.

Instead of comparing at the 5-minute timestamp level, this module
compares at the event level:
- For each catalog event, did the pipeline detect it?
- For each pipeline detection, does it match a catalog event?

This answers different questions than pointwise metrics:
- Event detection rate: "Of all real SEP events, how many did I find?"
- False alarm rate: "Of all my detections, how many have no catalog match?"

These are more forgiving than pointwise metrics because a detection
doesn't need perfect temporal overlap — it just needs to overlap
at all with the catalog event.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from sep_core.events import Event


def match_events_to_catalog(
    detected_events: List[Event],
    catalog: pd.DataFrame,
    min_overlap_minutes: float = 0.0
) -> Dict[str, Any]:
    """
    Match detected events against catalog events.

    For each catalog event, checks if ANY detected event overlaps with it.
    For each detected event, checks if ANY catalog event overlaps with it.

    An overlap is defined as: the detected event's time range intersects
    the catalog event's time range by at least min_overlap_minutes.
    With min_overlap_minutes=0, any intersection counts (even 1 shared
    timestamp).

    Parameters
    ----------
    detected_events : List[Event]
        Events detected by the pipeline (from events.py).
    catalog : pd.DataFrame
        NOAA SEP catalog with start_time, end_time columns.
    min_overlap_minutes : float
        Minimum overlap in minutes for a match. Default 0.

    Returns
    -------
    dict with keys:
        n_catalog_events : int — total catalog events
        n_detected_events : int — total detected events
        n_catalog_hits : int — catalog events matched by a detection
        n_catalog_misses : int — catalog events with no detection
        n_detected_hits : int — detected events matching a catalog event
        n_false_alarms : int — detected events with no catalog match
        event_detection_rate : float — n_catalog_hits / n_catalog_events
        false_alarm_rate : float — n_false_alarms / n_detected_events
        matched_catalog : list of dict — details per catalog event
        matched_detected : list of dict — details per detected event
    """

    min_overlap = pd.Timedelta(minutes=min_overlap_minutes)

    # Match catalog events → detected events
    matched_catalog = []
    for _, cat_row in catalog.iterrows():
        cat_start = cat_row["start_time"]
        cat_end = cat_row["end_time"]
        hit = False

        for det in detected_events:
            # Compute overlap
            overlap_start = max(cat_start, det.start_time)
            overlap_end = min(cat_end, det.end_time)
            overlap = overlap_end - overlap_start

            if overlap >= min_overlap:
                hit = True
                break

        matched_catalog.append({
            "start_time": cat_start,
            "end_time": cat_end,
            "peak_flux": cat_row.get("peak_flux_pfu", None),
            "detected": hit,
        })

    # Match detected events → catalog events
    matched_detected = []
    for det in detected_events:
        hit = False

        for _, cat_row in catalog.iterrows():
            cat_start = cat_row["start_time"]
            cat_end = cat_row["end_time"]

            overlap_start = max(cat_start, det.start_time)
            overlap_end = min(cat_end, det.end_time)
            overlap = overlap_end - overlap_start

            if overlap >= min_overlap:
                hit = True
                break

        matched_detected.append({
            "start_time": det.start_time,
            "end_time": det.end_time,
            "duration_minutes": det.duration_minutes,
            "has_catalog_match": hit,
        })

    # Compute summary statistics
    n_catalog = len(catalog)
    n_detected = len(detected_events)
    n_catalog_hits = sum(1 for m in matched_catalog if m["detected"])
    n_catalog_misses = n_catalog - n_catalog_hits
    n_detected_hits = sum(1 for m in matched_detected if m["has_catalog_match"])
    n_false_alarms = n_detected - n_detected_hits

    # Event-based confusion matrix:
    #   TP = catalog events that were detected
    #   FN = catalog events that were missed
    #   FP = detections that don't match any catalog event
    TP = n_catalog_hits
    FN = n_catalog_misses
    FP = n_false_alarms

    event_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    event_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    event_f1 = (
        2 * event_precision * event_recall / (event_precision + event_recall)
        if (event_precision + event_recall) > 0 else 0.0
    )

    event_detection_rate = (
        n_catalog_hits / n_catalog if n_catalog > 0 else 0.0
    )
    false_alarm_rate = (
        n_false_alarms / n_detected if n_detected > 0 else 0.0
    )

    return {
        "n_catalog_events": n_catalog,
        "n_detected_events": n_detected,
        "n_catalog_hits": n_catalog_hits,
        "n_catalog_misses": n_catalog_misses,
        "n_detected_hits": n_detected_hits,
        "n_false_alarms": n_false_alarms,
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "event_detection_rate": event_detection_rate,
        "false_alarm_rate": false_alarm_rate,
        "matched_catalog": matched_catalog,
        "matched_detected": matched_detected,
    }


def print_event_metrics(
    results: Dict[str, Any],
    title: str = "Event-Level Evaluation"
) -> None:
    """
    Print event-level metrics in a clean formatted block.

    Parameters
    ----------
    results : dict
        Output from match_events_to_catalog().
    title : str
        Header for the output block.
    """

    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
    print(f"  Catalog events:       {results['n_catalog_events']}")
    print(f"  Detected events:      {results['n_detected_events']}")
    print(f"  ---")
    print(f"  TP (catalog hits):    {results['TP']}")
    print(f"  FN (catalog misses):  {results['FN']}")
    print(f"  FP (false alarms):    {results['FP']}")
    print(f"  ---")
    print(f"  Event Precision:      {results['event_precision']:.4f}")
    print(f"  Event Recall (EDR):   {results['event_recall']:.4f}")
    print(f"  Event F1:             {results['event_f1']:.4f}")
    print(f"  FAR:                  {results['false_alarm_rate']:.4f}")
    print(f"{'=' * 50}")

    # Show missed catalog events
    missed = [m for m in results["matched_catalog"] if not m["detected"]]
    if missed:
        print(f"\n  Missed catalog events ({len(missed)}):")
        for m in missed:
            flux = m.get("peak_flux", "?")
            print(f"    {m['start_time']} — {m['end_time']}  "
                  f"({flux} pfu)")

    # Show false alarms
    false_alarms = [
        m for m in results["matched_detected"]
        if not m["has_catalog_match"]
    ]
    if false_alarms and len(false_alarms) <= 20:
        print(f"\n  False alarm detections ({len(false_alarms)}):")
        for m in false_alarms:
            print(f"    {m['start_time']} — {m['end_time']}  "
                  f"({m['duration_minutes']:.0f} min)")


def full_evaluation(
    predicted_mask: np.ndarray,
    time: pd.DatetimeIndex,
    detected_events: List[Event],
    catalog: pd.DataFrame,
    title: str = "Pipeline Evaluation"
) -> Dict[str, Any]:
    """
    Run both pointwise and event-level evaluation and print results.

    Convenience function that combines metrics.py and matching.py.

    Parameters
    ----------
    predicted_mask : np.ndarray
        Boolean detection mask from the pipeline.
    time : pd.DatetimeIndex
        Time axis aligned with the mask.
    detected_events : List[Event]
        Event intervals from events.py.
    catalog : pd.DataFrame
        NOAA SEP catalog.
    title : str
        Header for output.

    Returns
    -------
    dict with keys:
        pointwise : dict — pointwise metrics
        event_level : dict — event-level metrics
    """

    from sep_core.evaluation.metrics import (
        evaluate_detection,
        print_metrics
    )

    # Pointwise evaluation
    pointwise = evaluate_detection(predicted_mask, time, catalog)
    print_metrics(pointwise, title=f"{title} — Pointwise")

    # Event-level evaluation
    event_level = match_events_to_catalog(detected_events, catalog)
    print_event_metrics(event_level, title=f"{title} — Event Level")

    return {
        "pointwise": pointwise,
        "event_level": event_level,
    }