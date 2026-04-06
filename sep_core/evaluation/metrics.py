"""
sep_core/evaluation/metrics.py

Pointwise evaluation metrics for SEP detection.

Compares pipeline detections against a ground truth catalog
at the 5-minute timestamp level. Computes precision, recall,
F1 score, and confusion matrix elements.

Works with both:
  - GSEP catalog (load via gsep_catalog.py) — RECOMMENDED
  - Legacy NOAA scraped catalog (load via load_noaa_catalog())

For GSEP: use start_time and end_time (actual event end, not peak).
For NOAA scraped: end_time is peak time — evaluation will be pessimistic.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path


# ==================================================================
# CATALOG LOADERS
# ==================================================================

def load_catalog(
    catalog_path: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load any catalog CSV that has start_time, end_time, peak_flux_pfu columns.

    This is the generic loader. It works with:
    - The GSEP catalog after it has been loaded via gsep_catalog.py
      and saved/passed as a DataFrame
    - The legacy NOAA scraped CSV (noaa_sep_catalog_1995_2025.csv)

    For GSEP, prefer loading with load_gsep_catalog() directly since
    it handles the column renaming and flag filtering.

    Parameters
    ----------
    catalog_path : str
        Path to a CSV with columns: start_time, end_time, peak_flux_pfu
    start_year : int or None
        Filter to events starting >= this year.
    end_year : int or None
        Filter to events starting <= this year.

    Returns
    -------
    pd.DataFrame
        Columns: start_time, end_time, peak_flux_pfu, region, location.
        Times are pd.Timestamp.
    """

    df = pd.read_csv(catalog_path)
    df["start_time"] = pd.to_datetime(df["start_time"])
    df["end_time"] = pd.to_datetime(df["end_time"])

    if start_year is not None:
        df = df[df["start_time"].dt.year >= start_year]
    if end_year is not None:
        df = df[df["start_time"].dt.year <= end_year]

    return df.reset_index(drop=True)


def load_noaa_catalog(
    catalog_path: str,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load the legacy NOAA scraped catalog.

    WARNING: The end_time in this catalog is the peak time, NOT the
    actual event end. Using this for pointwise evaluation will give
    artificially low precision because the decay phase of each event
    counts as false positives. Use load_gsep_catalog() instead.

    Kept here for backward compatibility and comparison.
    """
    return load_catalog(catalog_path, start_year, end_year)


# ==================================================================
# MASK CONSTRUCTION
# ==================================================================

def catalog_to_mask(
    catalog: pd.DataFrame,
    time: pd.DatetimeIndex
) -> np.ndarray:
    """
    Convert catalog event intervals into a pointwise boolean mask.

    For each catalog event, marks all timestamps between start_time
    and end_time as True (inclusive).

    Works with both GSEP and legacy NOAA catalogs since both have
    start_time and end_time columns after loading.

    Parameters
    ----------
    catalog : pd.DataFrame
        SEP catalog with start_time and end_time columns.
    time : pd.DatetimeIndex
        The 5-minute time axis to create the mask on.

    Returns
    -------
    np.ndarray
        Boolean ground truth mask. True = real SEP event timestamp.
    """

    mask = np.zeros(len(time), dtype=bool)

    for _, row in catalog.iterrows():
        inside = (time >= row["start_time"]) & (time <= row["end_time"])
        mask |= inside

    return mask


# ==================================================================
# METRICS COMPUTATION
# ==================================================================

def compute_pointwise_metrics(
    predicted: np.ndarray,
    ground_truth: np.ndarray
) -> Dict[str, Any]:
    """
    Compute pointwise detection metrics (precision, recall, F1).

    These answer:
    - Precision: of all timestamps I flagged, how many were real?
    - Recall: of all real SEP timestamps, how many did I catch?
    - F1: harmonic mean of precision and recall.
    - Accuracy: misleading due to class imbalance — use F1 instead.

    Parameters
    ----------
    predicted : np.ndarray
        Boolean mask from pipeline detection.
    ground_truth : np.ndarray
        Boolean mask from catalog (via catalog_to_mask).

    Returns
    -------
    dict with keys:
        tp, fp, fn, tn : int — confusion matrix elements
        precision, recall, f1, accuracy : float — metrics
        n_total : int — total timestamps evaluated
    """

    pred = predicted.astype(bool)
    true = ground_truth.astype(bool)

    tp = int(np.sum(pred & true))
    fp = int(np.sum(pred & ~true))
    fn = int(np.sum(~pred & true))
    tn = int(np.sum(~pred & ~true))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "n_total": len(pred),
    }


def evaluate_detection(
    predicted_mask: np.ndarray,
    time: pd.DatetimeIndex,
    catalog: pd.DataFrame
) -> Dict[str, Any]:
    """
    Full pointwise evaluation against a catalog.

    Combines catalog_to_mask() + compute_pointwise_metrics() in one call.

    Parameters
    ----------
    predicted_mask : np.ndarray
        Boolean detection mask from the pipeline.
    time : pd.DatetimeIndex
        Time axis aligned with the mask.
    catalog : pd.DataFrame
        SEP catalog (GSEP recommended, NOAA legacy also works).

    Returns
    -------
    dict — all pointwise metrics (see compute_pointwise_metrics).
    """

    ground_truth = catalog_to_mask(catalog, time)
    return compute_pointwise_metrics(predicted_mask, ground_truth)


# ==================================================================
# PRINTING
# ==================================================================

def print_metrics(
    metrics: Dict[str, Any],
    title: str = "Evaluation",
    catalog_note: str = ""
) -> None:
    """
    Print metrics in a clean formatted block.

    Parameters
    ----------
    metrics : dict
        Output from compute_pointwise_metrics() or evaluate_detection().
    title : str
        Header label.
    catalog_note : str
        Optional note about the catalog used, e.g.,
        "GSEP catalog, Flag==1 events, 1995–2017"
    """

    print(f"\n{'=' * 55}")
    print(f"  {title}")
    if catalog_note:
        print(f"  Catalog: {catalog_note}")
    print(f"{'=' * 55}")
    print(f"  Total timestamps:  {metrics['n_total']:,}")
    print(f"  True Positives:    {metrics['tp']:,}")
    print(f"  False Positives:   {metrics['fp']:,}")
    print(f"  False Negatives:   {metrics['fn']:,}")
    print(f"  True Negatives:    {metrics['tn']:,}")
    print(f"  ---")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall:            {metrics['recall']:.4f}")
    print(f"  F1 Score:          {metrics['f1']:.4f}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}  "
          f"(misleading — see F1)")
    print(f"{'=' * 55}")