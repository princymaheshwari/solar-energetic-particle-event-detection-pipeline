"""

Reusable thresholding and detection utilities for SEP detection.

Implements the detection logic with hysteresis:
1. Threshold: flux >= enter_threshold (e.g., 10 pfu) to START
2. Rising gradient: in a sliding window of N gradients, allow at most
   K negative ones (equivalently, require N-K positive ones) to START
3. Hysteresis exit: flux must drop below exit_threshold (e.g., 5 pfu)
   AND stay below it for quiet_period consecutive points to END
4. Duration filter: events shorter than min_duration are discarded

Rules 1+2 must both be true to START an event.
Rule 3 determines when an event ENDS (with hysteresis).
Rule 4 is applied at the end — short events are erased.

These are building blocks that adapters call. All functions are pure —
they take arrays in and return arrays out. No side effects, no file I/O.
"""

import numpy as np
import pandas as pd


# ==================================================================
# SMOOTHING (optional — available if needed, not used by default)
# ==================================================================

def smooth_flux(
    flux: np.ndarray,
    window: int = 3,
    method: str = "median"
) -> np.ndarray:
    """
    Smooth a flux time series to reduce noise before thresholding.

    This is OPTIONAL — the current pipeline does not use smoothing,
    but it's available for future experimentation.

    Parameters
    ----------
    flux : np.ndarray
        Raw flux values. May contain NaN.
    window : int
        Rolling window size in data points.
        Default 3 = 15 minutes at 5-minute cadence.
    method : str
        "median" — robust to outliers (recommended).
        "mean"  — sensitive to outliers.

    Returns
    -------
    np.ndarray
        Smoothed flux array. Same length as input.
    """

    if window < 1:
        return flux.copy()

    series = pd.Series(flux)

    if method == "median":
        smoothed = series.rolling(
            window=window, center=True, min_periods=1
        ).median()
    elif method == "mean":
        smoothed = series.rolling(
            window=window, center=True, min_periods=1
        ).mean()
    else:
        raise ValueError(
            f"Unknown smoothing method: '{method}'. Use 'median' or 'mean'."
        )

    return smoothed.to_numpy()


# ==================================================================
# RULE 1: THRESHOLD CHECK
# ==================================================================

def apply_threshold(
    flux: np.ndarray,
    threshold: float
) -> np.ndarray:
    """
    Flag timestamps where flux exceeds a threshold.

    Rule 1 of the detection logic. A timestamp must have
    flux >= threshold to be part of an SEP event.

    Parameters
    ----------
    flux : np.ndarray
        Flux values. May contain NaN.
    threshold : float
        The threshold value. Default for both GOES and SOHO: 10.0 pfu.

    Returns
    -------
    np.ndarray
        Boolean mask. True where flux >= threshold.
        NaN values produce False.
    """

    return flux >= threshold


# ==================================================================
# RULE 2: RISING GRADIENT CHECK
# ==================================================================

def compute_gradient(flux: np.ndarray) -> np.ndarray:
    """
    Compute point-to-point gradient (first difference) of flux.

    gradient[i] = flux[i] - flux[i-1]

    Positive gradient = flux is rising.
    Negative gradient = flux is falling.
    First element is NaN (no previous point).

    Parameters
    ----------
    flux : np.ndarray
        Flux values. May contain NaN.

    Returns
    -------
    np.ndarray
        Gradient array. Same length as flux.
    """

    return np.diff(flux, prepend=np.nan)


def check_rising_gradient(
    flux: np.ndarray,
    window: int = 4,
    allow_negative_inside: int = 1
) -> np.ndarray:
    """
    Check for sustained rising flux in a sliding window.

    Rule 2 of the detection logic. Ensures flux is actively
    increasing — not just sitting above threshold at a flat level.
    SEP onsets are characterized by a rapid rise in proton flux.

    Logic: in a sliding window of `window` gradient points,
    count positive gradients. Allow at most `allow_negative_inside`
    negative ones. Equivalently, require at least
    (window - allow_negative_inside) positive gradients.

    With defaults (window=4, allow_negative_inside=1):
    "Allow at most 1 negative gradient in 4 steps"
    = "Require at least 3 out of 4 gradients to be positive"

    Parameters
    ----------
    flux : np.ndarray
        Flux values. May contain NaN.
    window : int
        Size of the sliding window. Default 4.
    allow_negative_inside : int
        How many negative gradients are tolerated in the window.
        Default 1.

    Returns
    -------
    np.ndarray
        Boolean mask. True where the rising gradient condition is met.
    """

    grad = compute_gradient(flux)
    grad_positive = grad > 0

    # Count positive gradients in rolling window
    positive_count = (
        pd.Series(grad_positive.astype(float))
        .rolling(window=window, min_periods=window)
        .sum()
        .to_numpy()
    )

    required = window - allow_negative_inside
    return positive_count >= required


# ==================================================================
# COMBINED START SIGNAL
# ==================================================================

def compute_start_signal(
    flux: np.ndarray,
    threshold: float,
    gradient_window: int = 4,
    allow_negative_inside: int = 1
) -> np.ndarray:
    """
    Combine Rule 1 (threshold) and Rule 2 (gradient) into
    a single start signal.

    A timestamp is a valid event START point only if BOTH:
    - flux >= threshold
    - rising gradient condition is met

    This is exactly your script's:
        SEP_start_signal = above_thresh & grad_condition

    Parameters
    ----------
    flux : np.ndarray
        Flux values.
    threshold : float
        Flux threshold (e.g., 10.0 pfu).
    gradient_window : int
        Window size for gradient check. Default 4.
    allow_negative_inside : int
        Allowed negative gradients in window. Default 1.

    Returns
    -------
    np.ndarray
        Boolean mask. True at valid event start points.
    """

    above_thresh = apply_threshold(flux, threshold)
    grad_condition = check_rising_gradient(
        flux,
        window=gradient_window,
        allow_negative_inside=allow_negative_inside
    )

    return above_thresh & grad_condition


# ==================================================================
# STATE MACHINE + DURATION FILTER (Rule 3)
# ==================================================================

def track_events(
    flux: np.ndarray,
    start_signal: np.ndarray,
    threshold: float,
    min_duration_points: int = 6,
    exit_threshold: float = None,
    quiet_period_points: int = 0,
) -> np.ndarray:
    """
    Stateful event tracking with hysteresis, quiet period, and
    duration filtering.

    The state machine walks through the time series point by point:

    STATE 1 — Not in event:
        If start_signal[i] is True → begin event, switch to State 2.

    STATE 2 — Inside event:
        The event continues as long as flux >= exit_threshold.
        When flux drops below exit_threshold, a quiet-period counter
        starts. The event only truly ends when flux stays below
        exit_threshold for quiet_period_points consecutive points.
        If flux rises back above exit_threshold before the quiet
        period completes, the counter resets and the event continues.

        Once the event ends, check duration:
            If duration >= min_duration_points → accept.
            If duration < min_duration_points → reject (erase).

    Parameters
    ----------
    flux : np.ndarray
        Flux values. May contain NaN.
    start_signal : np.ndarray
        Boolean array from compute_start_signal().
    threshold : float
        Flux threshold for event entry (used only for backward
        compatibility; entry is controlled by start_signal).
    min_duration_points : int
        Minimum event length. Default 6 = 30 min at 5-min cadence.
    exit_threshold : float or None
        Flux level below which the event begins to end.
        If None, defaults to threshold (no hysteresis).
        Typical: 5.0 pfu when threshold is 10.0 pfu.
    quiet_period_points : int
        Number of consecutive points flux must stay below
        exit_threshold to confirm event end. Default 0 (immediate).
        Typical: 24 = 2 hours at 5-min cadence.

    Returns
    -------
    np.ndarray
        Boolean SEP event mask. True for detected timestamps.
    """

    if exit_threshold is None:
        exit_threshold = threshold

    n = len(flux)
    sep = np.zeros(n, dtype=bool)

    in_event = False
    event_start_idx = 0
    below_exit_count = 0

    for i in range(n):
        val = flux[i]
        is_nan = np.isnan(val) if not isinstance(val, bool) else False

        if not in_event:
            if start_signal[i]:
                in_event = True
                event_start_idx = i
                below_exit_count = 0
                sep[i] = True

        else:
            if is_nan or val < exit_threshold:
                below_exit_count += 1

                if below_exit_count >= quiet_period_points:
                    # Event confirmed ended — the event's last real point
                    # was quiet_period_points ago.
                    event_end_idx = i - quiet_period_points + 1
                    duration = event_end_idx - event_start_idx

                    if duration < min_duration_points:
                        sep[event_start_idx:event_end_idx] = False

                    in_event = False
                    below_exit_count = 0
                else:
                    # Still in quiet-period countdown — mark as part of event
                    sep[i] = True
            else:
                # Flux is back above exit_threshold — reset counter
                below_exit_count = 0
                sep[i] = True

    # Handle event open at end of time series
    if in_event:
        duration = n - event_start_idx
        if duration < min_duration_points:
            sep[event_start_idx:n] = False

    return sep


# ==================================================================
# STANDALONE DURATION FILTER (reusable utility)
# ==================================================================

def apply_duration_filter(
    mask: np.ndarray,
    min_duration_points: int = 6
) -> np.ndarray:
    """
    Remove contiguous True runs shorter than a minimum length.

    This is a standalone version of the duration filter, separate
    from the state machine. Useful as a general-purpose tool —
    for example, base.py's _apply_persistence() does the same thing.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask.
    min_duration_points : int
        Minimum run length to keep. Default 6.

    Returns
    -------
    np.ndarray
        Filtered mask. Short runs set to False.
    """

    filtered = mask.copy()

    changes = np.diff(mask.astype(int))
    starts = np.where(changes == 1)[0] + 1
    ends = np.where(changes == -1)[0] + 1

    if mask[0]:
        starts = np.insert(starts, 0, 0)
    if mask[-1]:
        ends = np.append(ends, len(mask))

    for s, e in zip(starts, ends):
        if (e - s) < min_duration_points:
            filtered[s:e] = False

    return filtered


# ==================================================================
# MAIN DETECTION FUNCTION
# ==================================================================

def detect_sep_events(
    flux: np.ndarray,
    threshold: float = 10.0,
    gradient_window: int = 4,
    allow_negative_inside: int = 1,
    min_duration_points: int = 6,
    exit_threshold: float = None,
    quiet_period_points: int = 0,
) -> tuple:
    """
    Run the full SEP detection on a flux time series.

    This is the main function that adapters call. It combines:
    - Rule 1: flux >= threshold (entry)
    - Rule 2: rising gradient condition (entry)
    - Rule 3: hysteresis exit (flux < exit_threshold for quiet_period)
    - Rule 4: minimum duration filter

    EVENT START requires Rule 1 AND Rule 2.
    EVENT CONTINUATION: flux stays above exit_threshold.
    EVENT END: flux drops below exit_threshold and stays there
              for quiet_period_points consecutive points.
    EVENT ACCEPTANCE requires Rule 4 (minimum duration).

    Parameters
    ----------
    flux : np.ndarray
        Flux values (e.g., >10 MeV integral proton flux in pfu).
    threshold : float
        Entry threshold. Default 10.0 pfu.
    gradient_window : int
        Sliding window for gradient check. Default 4.
    allow_negative_inside : int
        Allowed negative gradients in window. Default 1.
    min_duration_points : int
        Minimum event length in data points. Default 6 (= 30 min).
    exit_threshold : float or None
        Exit threshold for hysteresis. Default None = same as threshold.
        Typical: 5.0 pfu (event continues until flux drops below 5 pfu).
    quiet_period_points : int
        Consecutive points below exit_threshold needed to end event.
        Default 0 (immediate exit). Typical: 24 = 2 hours at 5-min.

    Returns
    -------
    tuple of (np.ndarray, dict)
        mask : boolean array, True for SEP-detected timestamps.
        info : dictionary with intermediate results for debugging.
    """

    # Rule 1: threshold (for entry)
    above_threshold = apply_threshold(flux, threshold)

    # Rule 2: rising gradient
    gradient_condition = check_rising_gradient(
        flux,
        window=gradient_window,
        allow_negative_inside=allow_negative_inside
    )

    # Combined start signal
    start_signal = above_threshold & gradient_condition

    # State machine with hysteresis + quiet period + duration filter
    sep_mask = track_events(
        flux,
        start_signal,
        threshold,
        min_duration_points,
        exit_threshold=exit_threshold,
        quiet_period_points=quiet_period_points,
    )

    # Package debugging info
    info = {
        "above_threshold": above_threshold,
        "gradient_condition": gradient_condition,
        "start_signal": start_signal,
        "threshold": threshold,
        "exit_threshold": exit_threshold if exit_threshold is not None else threshold,
        "quiet_period_points": quiet_period_points,
        "gradient_window": gradient_window,
        "allow_negative_inside": allow_negative_inside,
        "min_duration_points": min_duration_points,
    }

    return sep_mask, info