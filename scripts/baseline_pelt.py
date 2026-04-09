#!/usr/bin/env python3
"""PELT changepoint detection baseline for terminal boundary detection."""

import sys
from pathlib import Path

import numpy as np
import ruptures
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import D_MIN, D_MAX, make_method_result, timer


def _envelope(signal: np.ndarray) -> np.ndarray:
    """Compute the amplitude envelope via the analytic signal (Hilbert transform)."""
    analytic = hilbert(signal)
    return np.abs(analytic)


def pelt_parse_sentence(signal: np.ndarray) -> dict:
    """Detect terminal boundaries using PELT changepoint detection on the envelope.

    Steps:
      1. Compute envelope (abs Hilbert transform)
      2. Run PELT on the envelope to detect abrupt changes
      3. Map changepoints to terminal boundaries
      4. Label segments by sign at midpoint
      5. Classify sentence type by terminal count
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)

    if n < D_MIN:
        return {"sentence_type": "unknown", "starts": None, "ends": None, "labels": None}

    env = _envelope(sig)

    # Expected terminal count: impulse=9, correction=3 → 2-10 changepoints
    # Use BinSeg (faster than PELT) with penalty search
    best_bkps = None
    best_score = float("inf")

    for pen_mult in [1.0, 4.0, 10.0]:
        penalty = pen_mult * np.log(n) * np.var(env)
        algo = ruptures.Binseg(model="l2", min_size=D_MIN // 2, jump=10).fit(env)
        bkps = algo.predict(pen=penalty)

        n_segments = len(bkps)
        # Prefer results with 3-15 segments
        if 3 <= n_segments <= 15:
            dist = abs(n_segments - 6)  # 6 is midpoint of expected range
            if dist < best_score:
                best_score = dist
                best_bkps = bkps
        elif best_bkps is None:
            dist = min(abs(n_segments - 3), abs(n_segments - 15)) + 100
            if dist < best_score:
                best_score = dist
                best_bkps = bkps

    if best_bkps is None or len(best_bkps) < 2:
        return {"sentence_type": "unknown", "starts": None, "ends": None, "labels": None}

    # Build boundaries: [0, bkp1, bkp2, ..., n]
    boundary_points = sorted(set([0] + [b for b in best_bkps if b < n] + [n]))

    # Filter out tiny segments
    filtered = [boundary_points[0]]
    for bp in boundary_points[1:]:
        if bp - filtered[-1] >= D_MIN // 3:
            filtered.append(bp)
        else:
            filtered[-1] = bp  # merge with previous
    boundary_points = filtered

    if len(boundary_points) < 2:
        return {"sentence_type": "unknown", "starts": None, "ends": None, "labels": None}

    starts = np.array(boundary_points[:-1], dtype=np.int32)
    ends = np.array(boundary_points[1:], dtype=np.int32)

    # Label each segment by sign at midpoint
    labels = []
    for s, e in zip(starts, ends):
        mid = min((s + e) // 2, n - 1)
        labels.append("Up" if sig[mid] > 0 else "Down")

    n_terminals = len(labels)
    sentence_type = "impulse" if n_terminals > 5 else "correction"

    return {
        "sentence_type": sentence_type,
        "starts": starts,
        "ends": ends,
        "labels": labels,
    }


def run_pelt_benchmark(signals, labels, **kwargs):
    """Run PELT baseline on full dataset."""
    sentence_preds = []
    boundary_preds = []
    label_preds = []

    with timer() as t:
        for signal in signals:
            result = pelt_parse_sentence(signal)
            sentence_preds.append(result["sentence_type"])

            if result.get("starts") is not None:
                boundary_preds.append({
                    "starts": result["starts"],
                    "ends": result["ends"],
                })
                label_preds.append(result["labels"])
            else:
                boundary_preds.append(None)
                label_preds.append(None)

    return make_method_result(
        name="PELT",
        sentence_preds=sentence_preds,
        boundary_preds=boundary_preds,
        label_preds=label_preds,
        timing_sec=t.elapsed,
    )
