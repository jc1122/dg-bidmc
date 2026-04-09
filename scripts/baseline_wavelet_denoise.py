#!/usr/bin/env python3
"""Wavelet denoising + zero-crossing baseline for terminal boundary detection.

Applies VisuShrink wavelet soft-thresholding to denoise the signal, then detects
terminal boundaries via zero-crossings on the denoised signal. This is the fairest
comparison to DG-Hybrid: both filter noise before zero-crossing detection, but DG
uses graph structure while this uses frequency-domain filtering.
"""

import sys
from pathlib import Path

import numpy as np
import pywt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import D_MIN, make_method_result, timer


def _wavelet_denoise(signal: np.ndarray, wavelet: str = "db4",
                     level: int | None = None) -> np.ndarray:
    """Denoise signal using VisuShrink wavelet soft-thresholding.

    threshold = sigma_hat * sqrt(2 * log(n))
    where sigma_hat is estimated from the finest-level detail coefficients
    using the MAD estimator.
    """
    n = len(signal)
    if level is None:
        level = min(pywt.dwt_max_level(n, pywt.Wavelet(wavelet).dec_len), 6)
    if level < 1:
        return signal.copy()

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # Estimate noise sigma from finest detail coefficients (MAD estimator)
    detail_finest = coeffs[-1]
    sigma = np.median(np.abs(detail_finest)) / 0.6745

    # VisuShrink universal threshold
    threshold = sigma * np.sqrt(2 * np.log(n))

    # Soft-threshold all detail coefficients (keep approximation intact)
    denoised_coeffs = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised_coeffs.append(pywt.threshold(detail, value=threshold, mode="soft"))

    return pywt.waverec(denoised_coeffs, wavelet)[:n]


def _detect_zero_crossings(signal: np.ndarray, min_segment: int = 0) -> list[int]:
    """Find zero-crossing indices in the signal."""
    signs = np.sign(signal)
    nonzero_mask = signs != 0
    nonzero_signs = signs[nonzero_mask]
    nonzero_idx = np.where(nonzero_mask)[0]

    if len(nonzero_signs) < 2:
        return []

    flips = np.where(np.diff(nonzero_signs) != 0)[0]
    crossings = [(int(nonzero_idx[i]) + int(nonzero_idx[i + 1])) // 2 for i in flips]

    if min_segment <= 0:
        return crossings

    # Merge crossings that are too close
    filtered = [crossings[0]]
    for c in crossings[1:]:
        if c - filtered[-1] >= min_segment:
            filtered.append(c)
    return filtered


def wavelet_denoise_parse_sentence(signal: np.ndarray) -> dict:
    """Parse a sentence by denoising + zero-crossing detection.

    Steps:
      1. Apply wavelet soft-thresholding to denoise
      2. Detect zero-crossings on denoised signal
      3. Build terminal boundaries from crossings
      4. Label by sign at midpoint
      5. Classify by terminal count
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)

    if n < D_MIN:
        return {"sentence_type": "unknown", "starts": None, "ends": None, "labels": None}

    denoised = _wavelet_denoise(sig)
    crossings = _detect_zero_crossings(denoised, min_segment=D_MIN // 3)

    # Build boundaries
    boundary_points = sorted(set([0] + crossings + [n]))
    boundary_points = [b for b in boundary_points if 0 <= b <= n]

    # Filter out tiny segments
    filtered = [boundary_points[0]]
    for bp in boundary_points[1:]:
        if bp - filtered[-1] >= D_MIN // 3:
            filtered.append(bp)
    if filtered[-1] < n:
        filtered.append(n)
    boundary_points = filtered

    if len(boundary_points) < 2:
        return {"sentence_type": "unknown", "starts": None, "ends": None, "labels": None}

    starts = np.array(boundary_points[:-1], dtype=np.int32)
    ends = np.array(boundary_points[1:], dtype=np.int32)

    # Label each segment
    labels = []
    for s, e in zip(starts, ends):
        mid = min((s + e) // 2, n - 1)
        labels.append("Up" if denoised[mid] > 0 else "Down")

    n_terminals = len(labels)
    sentence_type = "impulse" if n_terminals > 5 else "correction"

    return {
        "sentence_type": sentence_type,
        "starts": starts,
        "ends": ends,
        "labels": labels,
    }


def run_wavelet_denoise_benchmark(signals, labels, **kwargs):
    """Run wavelet-denoise baseline on full dataset."""
    sentence_preds = []
    boundary_preds = []
    label_preds = []

    with timer() as t:
        for signal in signals:
            result = wavelet_denoise_parse_sentence(signal)
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
        name="WaveletDenoise",
        sentence_preds=sentence_preds,
        boundary_preds=boundary_preds,
        label_preds=label_preds,
        timing_sec=t.elapsed,
    )
