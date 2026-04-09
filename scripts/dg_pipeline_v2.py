#!/usr/bin/env python3
"""Enhanced DG pipeline with self-gating multi-stage preprocessing.

Each preprocessing stage runs on every signal but only modifies it
when its specific degradation type is detected. No routing needed.

Pipeline:
  1. Burst suppression: MAD-based outlier clipping (activates on impulsive noise)
  2. Harmonic removal: FFT notch filtering (activates on tonal interference)
  3. Wavelet denoising: two-tier VisuShrink (activates on broadband noise)
  4. Drift removal: conditional linear detrend (activates on DC drift)
  5. Standard DG: build_graph → assign_levels → recover_parse_tree
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dg_pipeline import build_graph, assign_levels, recover_parse_tree


# ── Signal Analysis ───────────────────────────────────────────────────────────


def _estimate_snr(signal, wavelet="db4"):
    """Wavelet-domain SNR estimation — robust to structural amplitude variation."""
    import pywt

    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)
    if n < 16:
        return 100.0

    wav = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, wav.dec_len)
    level = min(max_level, 3)
    if level < 1:
        return 100.0

    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma_noise = np.median(np.abs(coeffs[-1])) / 0.6745
    noise_power = sigma_noise**2
    if noise_power < 1e-12:
        return 100.0
    signal_power = np.mean(sig**2)
    return 10 * np.log10(signal_power / noise_power)


def _has_monotonic_drift(signal):
    """Detect DC drift via first-third vs last-third mean comparison."""
    n = len(signal)
    if n < 30:
        return False
    third = n // 3
    mean_start = np.mean(signal[:third])
    mean_end = np.mean(signal[-third:])
    drift = abs(mean_end - mean_start)
    sig_rms = np.sqrt(np.mean(signal**2)) + 1e-12
    return drift > 0.10 * sig_rms


# ── Self-Gating Preprocessing Stages ─────────────────────────────────────────


def _suppress_bursts(signal, mad_multiplier=6.0, window=7):
    """Stage 1: Replace impulsive outliers with local median.

    Only modifies samples that exceed MAD_MULTIPLIER × local MAD.
    Self-gating: no-op on clean/colored-noise signals where no outliers exist.
    """
    sig = signal.copy()
    n = len(sig)
    half = window // 2
    modified = 0

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        local = signal[lo:hi]
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        sigma_est = mad / 0.6745 if mad > 1e-12 else 1e-12
        if abs(signal[i] - med) > mad_multiplier * sigma_est:
            sig[i] = med
            modified += 1

    return sig


def _suppress_harmonics(signal, fs=2000, max_notches=6):
    """Stage 2: Remove tonal interference via adaptive FFT notching.

    Uses median-based spectral floor for robust peak detection.
    Self-gating: no-op if no prominent spectral peaks are found.
    """
    n = len(signal)
    if n < 64:
        return signal.copy()

    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    spectrum = np.fft.rfft(signal)
    magnitude = np.abs(spectrum).astype(np.float64)
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

    # Median-based local floor (robust to peaks, narrow window)
    from scipy.ndimage import median_filter as med_filt

    floor_width = max(int(3.0 / freq_res), 7)
    noise_floor = med_filt(magnitude, size=floor_width)

    notch_width = max(1, int(0.8 / freq_res))
    notched = 0

    for i in range(2, len(freqs) - 2):
        if freqs[i] < 3.0 or freqs[i] > 50.0:
            continue
        if noise_floor[i] < 1e-12:
            continue
        # Must be a local maximum
        if magnitude[i] <= magnitude[i - 1] or magnitude[i] <= magnitude[i + 1]:
            continue
        ratio = magnitude[i] / noise_floor[i]
        if ratio < 5.0:
            continue
        if notched < max_notches:
            lo = max(0, i - notch_width)
            hi = min(len(spectrum), i + notch_width + 1)
            # Tapered suppression to reduce Gibbs ringing
            for j in range(lo, hi):
                dist = abs(j - i) / max(notch_width, 1)
                suppress = max(0.0, 1.0 - dist)
                spectrum[j] *= (1.0 - 0.8 * suppress)
            notched += 1

    if notched == 0:
        return signal.copy()
    return np.fft.irfft(spectrum, n=n)


def _denoise_wavelet(signal, wavelet="db4"):
    """Stage 3: Two-tier VisuShrink wavelet denoising.

    Self-gating: bypasses clean signals (SNR > 15 dB).
    Moderate noise (5-15 dB): reduced 0.6× threshold.
    Heavy noise (< 5 dB): full universal threshold.
    """
    import pywt

    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)

    snr_db = _estimate_snr(sig, wavelet)

    if snr_db > 15.0:
        return sig.copy()

    wav = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, wav.dec_len)
    level = min(max_level, 6)

    if level < 1:
        return sig.copy()

    coeffs = pywt.wavedec(sig, wavelet, level=level)
    sigma_noise = np.median(np.abs(coeffs[-1])) / 0.6745
    universal_thresh = sigma_noise * np.sqrt(2 * np.log(n))

    scale = 0.6 if snr_db >= 5.0 else 1.0
    threshold = universal_thresh * scale

    denoised = [coeffs[0]]
    for detail in coeffs[1:]:
        denoised.append(pywt.threshold(detail, value=threshold, mode="soft"))

    return pywt.waverec(denoised, wavelet)[:n]


def _remove_drift(signal):
    """Stage 4: Conditional linear detrend.

    Self-gating: only applies when monotonic DC drift detected.
    """
    if not _has_monotonic_drift(signal):
        return signal.copy()
    sig = signal.copy()
    n = len(sig)
    trend = np.polyfit(np.arange(n), sig, deg=1)
    return sig - np.polyval(trend, np.arange(n))


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def preprocess_signal(
    signal: np.ndarray,
    wavelet: str = "db4",
    **_kwargs,
) -> np.ndarray:
    """Multi-stage self-gating preprocessing pipeline.

    Each stage only modifies the signal when its degradation type is detected.
    Order is critical: bursts bias everything, drift biases frequency analysis,
    harmonics are coherent (invisible to wavelet), broadband noise is last.
    """
    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)

    if n < 8:
        return sig.copy()
    if np.std(sig) < 1e-12:
        return sig.copy()

    # Stage 1: Burst suppression — gated on kurtosis OR fooled-wavelet-SNR
    # Burst noise has kurtosis>3 (heavy tails) or fools wavelet SNR estimator (>50dB)
    mean_val = np.mean(sig)
    std_val = np.std(sig)
    kurtosis = np.mean(((sig - mean_val) / std_val) ** 4) - 3.0 if std_val > 1e-12 else 0.0
    wavelet_snr = _estimate_snr(sig)
    if kurtosis > 3.0 or wavelet_snr > 50.0:
        sig = _suppress_bursts(sig)

    # Stage 2: Harmonic removal (self-gating via peak detection)
    sig = _suppress_harmonics(sig)

    # Stage 3: Wavelet denoising (handles broadband noise)
    sig = _denoise_wavelet(sig, wavelet)

    # Stage 4: Drift removal (last — drift detection more accurate on cleaned signal)
    sig = _remove_drift(sig)

    return np.ascontiguousarray(sig)


def _compute_amp_cv(signal: np.ndarray) -> float:
    """Amplitude coefficient of variation in D_MIN-sized windows."""
    from common import D_MIN

    n = len(signal)
    n_windows = n // D_MIN
    if n_windows < 2:
        return 1.0
    windows = signal[: n_windows * D_MIN].reshape(n_windows, D_MIN)
    window_amps = np.max(np.abs(windows), axis=1)
    mean_amp = np.mean(window_amps)
    if mean_amp < 1e-12:
        return 1.0
    return float(np.std(window_amps) / mean_amp)


def _run_dg_core(signal: np.ndarray) -> dict:
    """Core DG pipeline: preprocess → build graph → parse tree."""
    cleaned = preprocess_signal(signal)
    graph = build_graph(cleaned)
    node_level, edge_level = assign_levels(graph, n_levels=3)
    result = recover_parse_tree(graph, node_level, cleaned)
    result["diagnostics"]["edge_level_counts"] = {
        int(k): int(v)
        for k, v in zip(*np.unique(edge_level, return_counts=True))
    }
    result["diagnostics"]["preprocessing"] = "multistage_selfgating"
    return result


def _run_wd_fallback(signal: np.ndarray) -> dict:
    """WaveletDenoise fallback for signals where WD outperforms DG.

    Converts WD output to DG-compatible format so the adversarial suite
    normalizer handles it correctly. Falls back to DG core if WD fails.
    """
    from baseline_wavelet_denoise import wavelet_denoise_parse_sentence

    try:
        wd = wavelet_denoise_parse_sentence(signal)
    except (IndexError, ValueError):
        return _run_dg_core(signal)

    starts = wd.get("starts", [])
    ends = wd.get("ends", [])
    labels = wd.get("labels", [])

    boundaries = list(zip(starts, ends))
    return {
        "terminal_boundaries_pred": [(int(s), int(e)) for s, e in boundaries],
        "terminal_labels_pred": list(labels),
        "phrase_boundaries_pred": [],
        "sentence_type_pred": wd.get("sentence_type", "unknown"),
        "diagnostics": {},
    }


# Amplitude-CV routing threshold: below this, WD is generally better
_AMP_CV_THRESHOLD = 0.23
# Minimum signal length for WD routing (avoids short-segment signals)
_MIN_LEN_FOR_WD = 500
# Secondary rule thresholds for variable-amplitude signals
_SNR_THRESHOLD = 22.0
_AMP_RANGE_THRESHOLD = 18.0


def _should_use_wd(signal: np.ndarray) -> tuple[bool, str]:
    """Determine whether to route signal to WD instead of DG.

    Two rules:
    1. Low amplitude CV → clean/ratio signal where WD merges better
    2. Low SNR + high amplitude range → variable-amplitude signal

    Returns (should_use_wd, reason).
    """
    n = len(signal)
    if n < _MIN_LEN_FOR_WD:
        return False, "short_signal"

    amp_cv = _compute_amp_cv(signal)
    if amp_cv < _AMP_CV_THRESHOLD:
        return True, f"low_cv={amp_cv:.3f}"

    snr = _estimate_snr(signal)
    if snr < _SNR_THRESHOLD:
        abs_sig = np.abs(signal)
        p10 = np.percentile(abs_sig, 10)
        amp_range = np.max(abs_sig) / (p10 + 1e-10)
        if amp_range > _AMP_RANGE_THRESHOLD:
            return True, f"varamp(snr={snr:.0f},ar={amp_range:.0f})"

    return False, f"dg(cv={amp_cv:.3f})"


def run_dg_enhanced_pipeline(signal: np.ndarray) -> dict:
    """Enhanced DG pipeline with per-signal method routing.

    Routes to WaveletDenoise for signals where WD outperforms DG:
    - Low amplitude CV (clean/ratio conditions)
    - Variable amplitude with moderate noise
    Uses DG's structural analysis for complex/noisy signals.
    """
    sig = np.asarray(signal, dtype=np.float64)
    use_wd, reason = _should_use_wd(sig)

    if use_wd:
        result = _run_wd_fallback(sig)
        result.setdefault("diagnostics", {})
        result["diagnostics"]["routing"] = f"wavelet_denoise:{reason}"
    else:
        result = _run_dg_core(sig)
        result["diagnostics"]["routing"] = f"dg_structural:{reason}"

    return result


def _direct_zc_parse(signal: np.ndarray) -> dict:
    """Parse sentence using direct zero-crossing detection on clean signal.

    Uses a two-phase merge: first cluster close crossings (preserving the
    midpoint of each cluster), then filter by minimum segment length.
    This preserves more boundaries than the iterative smallest-segment merge.
    """
    from common import D_MIN

    sig = np.asarray(signal, dtype=np.float64)
    n = len(sig)

    # Find all zero crossings
    signs = np.sign(sig)
    nonzero_mask = signs != 0
    nonzero_signs = signs[nonzero_mask]
    nonzero_idx = np.where(nonzero_mask)[0]

    crossings = []
    if len(nonzero_signs) > 1:
        flips = np.where(np.diff(nonzero_signs) != 0)[0]
        for f in flips:
            zc = (int(nonzero_idx[f]) + int(nonzero_idx[f + 1])) // 2
            crossings.append(zc)

    if not crossings:
        terminal_boundaries = [(0, n)]
    else:
        # Phase 1: cluster close crossings (< D_MIN//8) by midpoint
        CLUSTER_GAP = max(D_MIN // 8, 1)
        clustered = [crossings[0]]
        cluster_start = crossings[0]
        for c in crossings[1:]:
            if c - clustered[-1] < CLUSTER_GAP:
                # Replace with running midpoint of this cluster
                clustered[-1] = (cluster_start + c) // 2
            else:
                clustered.append(c)
                cluster_start = c

        # Phase 2: sequential skip merge (like WaveletDenoise)
        MIN_SEG = D_MIN // 3
        boundary_points = sorted(set([0] + clustered + [n]))
        filtered = [boundary_points[0]]
        for bp in boundary_points[1:]:
            if bp - filtered[-1] >= MIN_SEG:
                filtered.append(bp)
        if filtered[-1] < n:
            filtered.append(n)

        terminal_boundaries = [(filtered[j], filtered[j + 1])
                               for j in range(len(filtered) - 1)
                               if filtered[j] < filtered[j + 1]]

    # Labels from signal sign at midpoint
    terminal_labels = []
    for start, end in terminal_boundaries:
        mid = min((start + end) // 2, n - 1)
        terminal_labels.append("Up" if sig[mid] > 0 else "Down")

    # Sentence classification
    n_terminals = len(terminal_labels)
    if n_terminals > 5:
        sentence_type = "impulse"
    elif n_terminals >= 2:
        sentence_type = "correction"
    else:
        sentence_type = "unknown"

    return {
        "terminal_boundaries_pred": terminal_boundaries,
        "terminal_labels_pred": terminal_labels,
        "phrase_boundaries_pred": [],
        "sentence_type_pred": sentence_type,
        "diagnostics": {
            "n_terminals_pred": n_terminals,
            "signal_length": n,
        },
    }
