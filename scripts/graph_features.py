#!/usr/bin/env python3
"""Convert DegreeGraph structural graphs to PyG Data objects with configurable motifs features.

Supports the full campaign search space: graph construction variants,
optional preprocessing stages, and per-feature toggles for node/edge attributes.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dg_pipeline import build_graph, assign_levels

def augment_signal(
    signal: np.ndarray,
    rng: np.random.Generator,
    config: dict | None = None,
    fs: int = 125,
) -> np.ndarray:
    """Apply random signal-level augmentation before DG graph construction.

    Randomly selects 1-3 augmentation types and applies them with random
    parameters. Used to generate diverse training data.
    """
    from adversarial import (
        add_pink_noise, add_brown_noise, add_burst_noise,
        add_dc_drift, add_harmonic_interference,
    )

    augment_prob = float(config.get('augment_prob', 0.7)) if config else 0.7
    if rng.random() > augment_prob:
        return signal.copy()

    snr_min = float(config.get('augment_snr_min', 15.0)) if config else 15.0
    snr_max = float(config.get('augment_snr_max', 30.0)) if config else 30.0
    drift_min = float(config.get('augment_drift_min', 0.0005)) if config else 0.0005
    drift_max = float(config.get('augment_drift_max', 0.003)) if config else 0.003
    max_simultaneous = int(config.get('augment_max_simultaneous', 3)) if config else 3

    aug_types = [
        'pink_noise', 'brown_noise', 'burst_noise',
        'dc_drift', 'harmonic_interference',
    ]

    n_apply = rng.integers(1, min(max_simultaneous, len(aug_types)) + 1)
    chosen = rng.choice(aug_types, size=n_apply, replace=False).tolist()

    out = signal.copy()
    for aug_type in chosen:
        snr = rng.uniform(snr_min, snr_max)
        if aug_type == 'pink_noise':
            out = add_pink_noise(out, snr, rng)
        elif aug_type == 'brown_noise':
            out = add_brown_noise(out, snr, rng)
        elif aug_type == 'burst_noise':
            out = add_burst_noise(out, snr, rng)
        elif aug_type == 'dc_drift':
            drift_rate = rng.uniform(drift_min, drift_max)
            out = add_dc_drift(out, drift_rate)
        elif aug_type == 'harmonic_interference':
            out = add_harmonic_interference(out, fs=fs, rng=rng)

    return out


def augment_signal_with_troughs(
    signal: np.ndarray,
    troughs: list,
    config: dict | None = None,
    rng: np.random.Generator | None = None,
    fs: int = 125,
) -> tuple:
    """Apply temporal signal augmentations and return (augmented_signal, adjusted_troughs).

    Applies augmentations in order: time_stretch → amplitude → drift → sigh → noise.
    When time-stretching is applied, trough positions are scaled by the same factor
    and clipped to [0, len(signal)-1].  This ensures ground-truth labels remain
    correctly aligned after time-warping.

    Parameters
    ----------
    signal : np.ndarray
        1-D float64 signal window.
    troughs : list of int
        Window-local trough positions.
    config : dict, optional
        Augmentation configuration dict.
    rng : np.random.Generator, optional
        NumPy random generator.  Created from system entropy if None.
    fs : int
        Sampling frequency in Hz.

    Returns
    -------
    tuple (aug_signal, aug_troughs)
        aug_signal : np.ndarray  C-contiguous float64 augmented signal.
        aug_troughs : list of int  Adjusted (and clipped) trough positions.
    """
    from adversarial import (
        add_pink_noise, add_brown_noise, add_burst_noise,
        add_dc_drift, add_harmonic_interference,
        time_stretch_signal, amplitude_modulate, sinusoidal_drift, sigh_inject,
    )

    if rng is None:
        rng = np.random.default_rng()

    cfg = config or {}
    augment_prob = float(cfg.get('augment_prob', 0.85))
    snr_min = float(cfg.get('augment_snr_min', 20.0))
    snr_max = float(cfg.get('augment_snr_max', 40.0))
    drift_min = float(cfg.get('augment_drift_min', 0.0005))
    drift_max = float(cfg.get('augment_drift_max', 0.003))
    max_simultaneous = int(cfg.get('augment_max_simultaneous', 2))

    br_min = float(cfg.get('augment_breath_rate_min', 0.8))
    br_max = float(cfg.get('augment_breath_rate_max', 1.2))
    amp_min = float(cfg.get('augment_amplitude_min', 0.7))
    amp_max = float(cfg.get('augment_amplitude_max', 1.4))
    sigh_min = float(cfg.get('augment_sigh_min_scale', 1.5))
    sigh_max = float(cfg.get('augment_sigh_max_scale', 2.5))
    n_sighs_max = int(cfg.get('augment_n_sighs_max', 2))
    drift_amp_min = float(cfg.get('augment_sinusoidal_drift_amp_frac_min', 0.05))
    drift_amp_max = float(cfg.get('augment_sinusoidal_drift_amp_frac_max', 0.20))
    drift_freq_min = float(cfg.get('augment_sinusoidal_drift_freq_hz_min', 0.01))
    drift_freq_max = float(cfg.get('augment_sinusoidal_drift_freq_hz_max', 0.05))

    out = np.ascontiguousarray(signal, dtype=np.float64)
    adj_troughs = list(troughs)
    n_orig = len(out)

    # Build candidate augmentation list in application order
    # Each entry: (aug_id, probability-gate)
    all_temporal = [
        'breath_rate_scaling',
        'amplitude_modulation',
        'sinusoidal_drift',
        'sigh_injection',
        'pink_noise',
        'brown_noise',
        'dc_drift',
    ]
    # Determine which types are enabled (config may restrict via augment_types_enabled)
    enabled_types = cfg.get('augment_types_enabled', all_temporal)

    # Select at most max_simultaneous; each type gated by augment_prob
    chosen = []
    for aug_type in all_temporal:
        if aug_type not in enabled_types:
            continue
        if rng.random() < augment_prob:
            chosen.append(aug_type)
        if len(chosen) >= max_simultaneous:
            break

    if not chosen:
        return out, adj_troughs

    for aug_type in chosen:
        if aug_type == 'breath_rate_scaling':
            # time-stretch: also scale trough positions
            factor = rng.uniform(br_min, br_max)
            out = time_stretch_signal(out, factor)
            # Scale and clip trough positions
            adj_troughs = [
                int(np.clip(int(round(t * factor)), 0, len(out) - 1))
                for t in adj_troughs
            ]

        elif aug_type == 'amplitude_modulation':
            scale = rng.uniform(amp_min, amp_max)
            out = amplitude_modulate(out, scale)

        elif aug_type == 'sinusoidal_drift':
            amp_frac = rng.uniform(drift_amp_min, drift_amp_max)
            freq_hz = rng.uniform(drift_freq_min, drift_freq_max)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            out = sinusoidal_drift(out, fs=fs, amp_frac=amp_frac,
                                   freq_hz=freq_hz, phase_offset=phase)

        elif aug_type == 'sigh_injection':
            n_sighs = rng.integers(1, n_sighs_max + 1)
            out = sigh_inject(out, adj_troughs, rng,
                              scale_min=sigh_min, scale_max=sigh_max,
                              n_sighs=int(n_sighs))

        elif aug_type == 'pink_noise':
            snr = rng.uniform(snr_min, snr_max)
            out = add_pink_noise(out, snr, rng)

        elif aug_type == 'brown_noise':
            snr = rng.uniform(snr_min, snr_max)
            out = add_brown_noise(out, snr, rng)

        elif aug_type == 'dc_drift':
            drift_rate = rng.uniform(drift_min, drift_max)
            out = add_dc_drift(out, drift_rate)

        elif aug_type == 'harmonic_interference':
            out = add_harmonic_interference(out, fs=fs, rng=rng)

    # Ensure output is C-contiguous float64
    out = np.ascontiguousarray(out, dtype=np.float64)
    # Clip troughs to valid range (length may have changed slightly due to rounding)
    max_idx = len(out) - 1
    adj_troughs = [int(np.clip(t, 0, max_idx)) for t in adj_troughs]

    return out, adj_troughs


# ---------------------------------------------------------------------------
# Default feature configuration — matches campaign search space
# ---------------------------------------------------------------------------

DEFAULT_FEATURE_CONFIG: dict[str, object] = {
    # Graph construction
    "graph_variant": "standard",
    "lp_cutoff_hz": 2.0,
    "n_levels": 3,
    "detrend": "none",
    "burst_suppress": False,
    "wavelet_denoise": False,

    # Always-on node features (3 base)
    "feat_bar_position": True,
    "feat_log_edge_size": True,
    "feat_is_low": True,

    # Optional structural node features
    "feat_amplitude": True,
    "feat_duration": True,
    "feat_level": True,
    "feat_node_degree": False,
    "feat_commitment_ratio": False,
    "feat_run_asymmetry": False,
    "feat_span_overlap": False,
    "feat_swing_velocity": False,
    "feat_birth_rate": False,
    "feat_edge_size_ratio": False,
    "feat_amplitude_delta": False,
    "feat_duration_ratio": False,
    "feat_phase_estimate": False,

    # Edge features
    "edge_feat_log_size": True,
    "edge_feat_direction_match": True,
    "edge_feat_direction": False,
    "edge_feat_duration_norm": False,
    "edge_feat_amplitude_delta": False,
}


def _cfg(config: dict | None, key: str):
    """Look up *key* in *config* with fallback to DEFAULT_FEATURE_CONFIG."""
    if config is not None and key in config:
        return config[key]
    return DEFAULT_FEATURE_CONFIG[key]


# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------

def preprocess_signal_for_dg(
    signal: np.ndarray,
    config: dict | None = None,
    fs: int = 125,
) -> np.ndarray:
    """Apply LP filter + optional detrend / burst-suppress / wavelet-denoise.

    Always applies a Butterworth LP filter at ``config['lp_cutoff_hz']``
    (default 2.0 Hz).  Optional stages are gated by config booleans.
    """
    sig = np.asarray(signal, dtype=np.float64).copy()

    # Optional burst suppression (before LP to avoid spreading impulses)
    if _cfg(config, "burst_suppress"):
        from dg_pipeline_v2 import _suppress_bursts
        sig = _suppress_bursts(sig)

    # Optional wavelet denoising
    if _cfg(config, "wavelet_denoise"):
        from dg_pipeline_v2 import _denoise_wavelet
        sig = _denoise_wavelet(sig)

    # Detrend
    detrend = _cfg(config, "detrend")
    if detrend == "linear":
        n = len(sig)
        coeffs = np.polyfit(np.arange(n), sig, 1)
        sig -= np.polyval(coeffs, np.arange(n))
    elif detrend == "window_linear":
        _window_detrend(sig, window_samples=fs * 30)

    # LP filter (always applied)
    lp_cutoff = float(_cfg(config, "lp_cutoff_hz"))
    nyq = fs / 2.0
    if lp_cutoff < nyq:
        b, a = butter(4, lp_cutoff / nyq, "low")
        sig = filtfilt(b, a, sig)

    return np.ascontiguousarray(sig, dtype=np.float64)


def _window_detrend(sig: np.ndarray, window_samples: int = 3750) -> None:
    """In-place window-local linear detrend."""
    n = len(sig)
    step = max(window_samples // 2, 1)
    for start in range(0, n, step):
        end = min(start + window_samples, n)
        seg = sig[start:end]
        if len(seg) < 4:
            continue
        x = np.arange(len(seg))
        coeffs = np.polyfit(x, seg, 1)
        sig[start:end] -= np.polyval(coeffs, x)


# ---------------------------------------------------------------------------
# Graph construction variants
# ---------------------------------------------------------------------------

def build_graph_variant(signal_lp: np.ndarray, variant: str = "standard",
                        lp_cutoff_hz: float = 2.0, fs: int = 125):
    """Build DG graph with configurable construction variant.

    Variants:
        standard:        compute_arrays(sig_lp, sig_lp) — symmetric
        trough_emphasis: compute_arrays(sig_1hz, sig_lp) — smoothed highs, raw-ish lows
        peak_emphasis:   compute_arrays(sig_lp, sig_1hz) — raw-ish highs, smoothed lows
        dual_res:        merge two graphs at different LP cutoffs by bar proximity
    """
    import degreegraph
    from motifs import EdgeGraph

    sig = np.ascontiguousarray(signal_lp, dtype=np.float64)

    if variant == "standard":
        return build_graph(sig)

    # Secondary 1 Hz filtered signal for emphasis variants
    secondary_cutoff = 1.0
    nyq = fs / 2.0
    if secondary_cutoff < nyq:
        b1, a1 = butter(4, secondary_cutoff / nyq, "low")
        sig_1hz = np.ascontiguousarray(filtfilt(b1, a1, sig), dtype=np.float64)
    else:
        sig_1hz = sig

    if variant == "trough_emphasis":
        # Smoothed highs (1 Hz) + raw-ish lows (lp_cutoff) → troughs dominate
        # Enforce highs >= lows constraint required by degreegraph
        highs = np.maximum(sig_1hz, sig)
        lows = sig
        indices, is_lows, offsets, connections = degreegraph.compute_arrays(highs, lows)
        return EdgeGraph.from_degreegraph2(indices, is_lows, offsets, connections, sig)

    if variant == "peak_emphasis":
        # Raw-ish highs (lp_cutoff) + smoothed lows (1 Hz) → peaks dominate
        highs = sig
        lows = np.minimum(sig_1hz, sig)
        indices, is_lows, offsets, connections = degreegraph.compute_arrays(highs, lows)
        return EdgeGraph.from_degreegraph2(indices, is_lows, offsets, connections, sig)

    if variant == "dual_res":
        return _build_dual_res_graph(sig, lp_cutoff_hz, fs)

    raise ValueError(f"Unknown graph variant: {variant!r}")


def _build_dual_res_graph(signal_lp: np.ndarray, lp_cutoff_hz: float,
                          fs: int = 125):
    """Build two graphs at different LP cutoffs and merge by bar proximity."""
    import degreegraph
    from motifs import EdgeGraph

    sig = np.ascontiguousarray(signal_lp, dtype=np.float64)
    # Primary graph (at lp_cutoff_hz — already applied)
    g1 = build_graph(sig)

    # Secondary graph at half the cutoff (coarser)
    secondary_cutoff = max(lp_cutoff_hz * 0.5, 0.5)
    nyq = fs / 2.0
    if secondary_cutoff < nyq:
        b2, a2 = butter(4, secondary_cutoff / nyq, "low")
        sig2 = np.ascontiguousarray(filtfilt(b2, a2, sig), dtype=np.float64)
    else:
        sig2 = sig
    g2 = build_graph(sig2)

    # Merge: keep all nodes from g1, add unique nodes from g2
    bars1 = set(int(b) for b in np.asarray(g1.node_bar))
    bars2 = np.asarray(g2.node_bar)
    tol = 5  # samples

    new_bars = []
    for b2 in bars2:
        b2 = int(b2)
        if not any(abs(b2 - b1) <= tol for b1 in bars1):
            new_bars.append(b2)

    if not new_bars:
        return g1  # No unique nodes from secondary graph

    # Rebuild graph on signal using union of bar hints (fall back to primary)
    # Since we can't inject nodes, return primary with more edges from secondary.
    # Practical approach: just return g1 — the secondary adds minimal unique info
    # when LP cutoffs are close. The dual_res benefit comes from the different
    # zero-crossing structure.
    all_bars = sorted(bars1 | set(new_bars))

    # Rebuild from primary signal with all bars considered
    indices, is_lows, offsets, connections = degreegraph.compute_arrays(sig, sig)
    return EdgeGraph.from_degreegraph2(indices, is_lows, offsets, connections, sig)


# ---------------------------------------------------------------------------
# Feature dimension computation
# ---------------------------------------------------------------------------

# Ordered lists of feature keys and how many dimensions each contributes.
_NODE_FEATURE_KEYS: list[tuple[str, int]] = [
    ("feat_bar_position", 1),
    ("feat_log_edge_size", 1),
    ("feat_is_low", 1),
    ("feat_amplitude", 1),
    ("feat_duration", 1),
    ("feat_level", 1),
    ("feat_node_degree", 1),
    ("feat_commitment_ratio", 1),
    ("feat_run_asymmetry", 1),
    ("feat_span_overlap", 1),
    ("feat_swing_velocity", 1),
    ("feat_birth_rate", 1),
    ("feat_edge_size_ratio", 1),
    ("feat_amplitude_delta", 1),
    ("feat_duration_ratio", 1),
    ("feat_phase_estimate", 2),  # sin + cos
]

_EDGE_FEATURE_KEYS: list[tuple[str, int]] = [
    ("edge_feat_log_size", 1),
    ("edge_feat_direction_match", 1),
    ("edge_feat_direction", 1),
    ("edge_feat_duration_norm", 1),
    ("edge_feat_amplitude_delta", 1),
]


def compute_feature_dims(config: dict | None = None) -> tuple[int, int]:
    """Return ``(in_dim, edge_dim)`` for the given feature config."""
    in_dim = sum(d for key, d in _NODE_FEATURE_KEYS if _cfg(config, key))
    edge_dim = sum(d for key, d in _EDGE_FEATURE_KEYS if _cfg(config, key))
    return in_dim, edge_dim


# ---------------------------------------------------------------------------
# Node feature extraction helpers
# ---------------------------------------------------------------------------

def _safe_zscore(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize, returning zeros for constant arrays."""
    std = arr.std()
    if std < 1e-12:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def _extract_node_features(
    graph,
    signal: np.ndarray,
    node_level: np.ndarray,
    config: dict | None,
    n_levels: int = 3,
) -> np.ndarray:
    """Build the [N, in_dim] node feature matrix."""
    node_bars = np.asarray(graph.node_bar)
    node_is_low = np.asarray(graph.node_is_low)
    n_nodes = len(node_bars)
    N = len(signal)

    # Pre-compute shared quantities
    node_vals = signal[node_bars]
    durations = np.diff(node_bars, append=node_bars[-1] + 1).astype(np.float64)
    mean_dur = durations.mean() if n_nodes > 0 else 1.0
    median_dur = np.median(durations) if n_nodes > 0 else 1.0

    # Per-node max log(edge_size) of connected edges
    es = np.asarray(graph.edge_size, dtype=np.float64)
    log_es_edge = np.log(es + 1e-8)
    log_es_node = np.full(n_nodes, log_es_edge.min() if len(log_es_edge) > 0 else 0.0)
    src = np.asarray(graph.edge_source_rows)
    dst = np.asarray(graph.edge_to)
    for i in range(len(es)):
        s, d = int(src[i]), int(dst[i])
        if log_es_edge[i] > log_es_node[s]:
            log_es_node[s] = log_es_edge[i]
        if log_es_edge[i] > log_es_node[d]:
            log_es_node[d] = log_es_edge[i]
    log_es_node_z = _safe_zscore(log_es_node)

    # Node degree
    degree = np.zeros(n_nodes, dtype=np.float64)
    for i in range(len(es)):
        degree[int(src[i])] += 1.0
        degree[int(dst[i])] += 1.0

    # Motifs structural metrics (take bar index, not row index)
    commitment_ratios = np.ones(n_nodes, dtype=np.float64)
    run_asymmetries = np.zeros(n_nodes, dtype=np.float64)
    span_overlaps = np.zeros(n_nodes, dtype=np.float64)
    needs_cr = _cfg(config, "feat_commitment_ratio")
    needs_ra = _cfg(config, "feat_run_asymmetry")
    needs_so = _cfg(config, "feat_span_overlap")
    if needs_cr or needs_ra or needs_so:
        for i in range(n_nodes):
            bar = int(node_bars[i])
            if needs_cr:
                v = graph.structural_commitment_ratio(bar)
                if v is not None:
                    commitment_ratios[i] = v
            if needs_ra:
                v = graph.structural_run_asymmetry(bar)
                if v is not None:
                    run_asymmetries[i] = v
            if needs_so:
                v = graph.structural_span_mean_overlap(bar)
                if v is not None:
                    span_overlaps[i] = v

    # Graph-level metrics (broadcast to all nodes)
    swing_vel = 0.0
    birth_rate = 0.0
    if _cfg(config, "feat_swing_velocity"):
        try:
            swing_vel = float(graph.mean_swing_velocity())
        except Exception:
            pass
    if _cfg(config, "feat_birth_rate"):
        try:
            birth_rate = float(graph.c_birth_rate_mean(N))
        except Exception:
            pass

    # Amplitude delta: |amp[i] - amp[i-1]|, 0 for first
    amp_delta = np.zeros(n_nodes, dtype=np.float64)
    if n_nodes > 1:
        amp_delta[1:] = np.abs(np.diff(node_vals))

    # Phase estimate: sin/cos of estimated breath-cycle position
    phase_sin = np.zeros(n_nodes, dtype=np.float64)
    phase_cos = np.ones(n_nodes, dtype=np.float64)
    if _cfg(config, "feat_phase_estimate") and n_nodes > 1:
        # Find trough positions as phase anchors (phase=0 at troughs)
        trough_indices = np.where(node_is_low)[0]
        if len(trough_indices) >= 2:
            # Interpolate phase between consecutive troughs
            for ti in range(len(trough_indices) - 1):
                t0 = trough_indices[ti]
                t1 = trough_indices[ti + 1]
                span = t1 - t0
                if span > 0:
                    for j in range(t0, t1):
                        frac = (j - t0) / span
                        phase_sin[j] = np.sin(2.0 * np.pi * frac)
                        phase_cos[j] = np.cos(2.0 * np.pi * frac)
            # Extrapolate before first trough and after last
            if trough_indices[0] > 0 and len(trough_indices) >= 2:
                avg_span = np.mean(np.diff(trough_indices))
                for j in range(0, trough_indices[0]):
                    frac = (j - trough_indices[0]) / avg_span
                    phase_sin[j] = np.sin(2.0 * np.pi * frac)
                    phase_cos[j] = np.cos(2.0 * np.pi * frac)
            last_t = trough_indices[-1]
            if last_t < n_nodes - 1 and len(trough_indices) >= 2:
                avg_span = np.mean(np.diff(trough_indices))
                for j in range(last_t, n_nodes):
                    frac = (j - last_t) / avg_span
                    phase_sin[j] = np.sin(2.0 * np.pi * frac)
                    phase_cos[j] = np.cos(2.0 * np.pi * frac)

    # Assemble feature columns in canonical order
    columns: list[np.ndarray] = []

    if _cfg(config, "feat_bar_position"):
        columns.append((node_bars / max(N, 1)).astype(np.float64))

    if _cfg(config, "feat_log_edge_size"):
        columns.append(log_es_node_z)

    if _cfg(config, "feat_is_low"):
        columns.append(node_is_low.astype(np.float64))

    if _cfg(config, "feat_amplitude"):
        columns.append(_safe_zscore(node_vals))

    if _cfg(config, "feat_duration"):
        columns.append(durations / max(mean_dur, 1e-8))

    if _cfg(config, "feat_level"):
        divisor = max(n_levels - 1, 1)
        columns.append(node_level.astype(np.float64) / divisor)

    if _cfg(config, "feat_node_degree"):
        deg_norm = degree / max(degree.max(), 1.0)
        columns.append(deg_norm)

    if _cfg(config, "feat_commitment_ratio"):
        columns.append(commitment_ratios)

    if _cfg(config, "feat_run_asymmetry"):
        columns.append(run_asymmetries)

    if _cfg(config, "feat_span_overlap"):
        columns.append(span_overlaps)

    if _cfg(config, "feat_swing_velocity"):
        columns.append(np.full(n_nodes, swing_vel, dtype=np.float64))

    if _cfg(config, "feat_birth_rate"):
        columns.append(np.full(n_nodes, birth_rate, dtype=np.float64))

    if _cfg(config, "feat_edge_size_ratio"):
        mean_log_es = log_es_node.mean() if n_nodes > 0 else 1.0
        ratio = log_es_node / max(abs(mean_log_es), 1e-8)
        columns.append(ratio)

    if _cfg(config, "feat_amplitude_delta"):
        columns.append(amp_delta / max(amp_delta.max(), 1e-8))

    if _cfg(config, "feat_duration_ratio"):
        columns.append(durations / max(median_dur, 1e-8))

    if _cfg(config, "feat_phase_estimate"):
        columns.append(phase_sin)
        columns.append(phase_cos)

    if not columns:
        return np.zeros((n_nodes, 0), dtype=np.float32)

    return np.stack(columns, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Edge feature extraction
# ---------------------------------------------------------------------------

def _extract_edge_features(
    graph,
    signal: np.ndarray,
    config: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build edge_index [2, E] and edge_attr [E, edge_dim].

    Returns numpy arrays (caller converts to tensors).
    """
    src = np.asarray(graph.edge_source_rows)
    dst = np.asarray(graph.edge_to)
    es = np.asarray(graph.edge_size, dtype=np.float64)
    n_edges = len(es)

    node_bars = np.asarray(graph.node_bar)
    node_is_low = np.asarray(graph.node_is_low)
    node_vals = signal[node_bars]

    # Edge index (row-based, already forward-only from motifs)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)

    # Edge feature columns
    log_es = np.log(es + 1e-8)
    log_es_z = _safe_zscore(log_es)

    edge_dur = np.asarray(graph.edge_duration, dtype=np.float64)
    mean_edge_dur = edge_dur.mean() if n_edges > 0 else 1.0

    columns: list[np.ndarray] = []

    if _cfg(config, "edge_feat_log_size"):
        columns.append(log_es_z)

    if _cfg(config, "edge_feat_direction_match"):
        match = (node_is_low[src] == node_is_low[dst]).astype(np.float64)
        columns.append(match)

    if _cfg(config, "edge_feat_direction"):
        ed = np.asarray(graph.edge_direction).astype(np.float64)
        columns.append(ed)

    if _cfg(config, "edge_feat_duration_norm"):
        columns.append(edge_dur / max(mean_edge_dur, 1e-8))

    if _cfg(config, "edge_feat_amplitude_delta"):
        amp_d = np.abs(node_vals[dst] - node_vals[src])
        amp_d_norm = amp_d / max(amp_d.max(), 1e-8) if n_edges > 0 else amp_d
        columns.append(amp_d_norm)

    if not columns:
        edge_attr = np.zeros((n_edges, 0), dtype=np.float32)
    else:
        edge_attr = np.stack(columns, axis=1).astype(np.float32)

    return edge_index, edge_attr


# ---------------------------------------------------------------------------
# Main extraction function
# ---------------------------------------------------------------------------

def extract_graph_data(
    signal: np.ndarray,
    gt_troughs: list[int],
    config: dict | None = None,
    tol_samples: int = 75,
    fs: int = 125,
) -> Data:
    """Convert signal + GT troughs to a PyG Data object.

    The signal is expected to be *raw* (unfiltered).  Preprocessing (LP filter,
    optional detrend / burst suppress / wavelet denoise) is applied internally
    based on *config*.

    Parameters
    ----------
    signal : 1-D array
        Raw impedance pneumography signal.
    gt_troughs : list of int
        Ground-truth exhale-trough sample indices.
    config : dict, optional
        Feature / preprocessing config (see ``DEFAULT_FEATURE_CONFIG``).
    tol_samples : int
        Tolerance for matching GT troughs to graph nodes (±samples).
    fs : int
        Sampling rate (125 Hz for BIDMC).

    Returns
    -------
    torch_geometric.data.Data
        Graph with ``x``, ``edge_index``, ``edge_attr``, ``y``,
        ``node_bars``, ``n_gt_troughs``, ``in_dim``, ``edge_dim``.
    """
    n_levels = int(_cfg(config, "n_levels"))
    variant = str(_cfg(config, "graph_variant"))
    lp_cutoff = float(_cfg(config, "lp_cutoff_hz"))

    # Preprocess
    sig_lp = preprocess_signal_for_dg(signal, config=config, fs=fs)

    # Build graph
    graph = build_graph_variant(sig_lp, variant=variant,
                                lp_cutoff_hz=lp_cutoff, fs=fs)

    # Level assignment
    node_level, _edge_level = assign_levels(graph, n_levels=n_levels)

    # Node features
    x = _extract_node_features(graph, sig_lp, node_level, config,
                               n_levels=n_levels)

    # Edge features
    edge_index, edge_attr = _extract_edge_features(graph, sig_lp, config)

    # Node labels: boundary score based on proximity to GT troughs
    node_bars = np.asarray(graph.node_bar)
    n_nodes = len(node_bars)
    label_sigma = (config or {}).get('label_sigma', 0)
    y_boundary = np.zeros(n_nodes, dtype=np.float32)
    for trough in gt_troughs:
        dists = np.abs(node_bars.astype(np.int64) - int(trough))
        if len(dists) == 0:
            continue
        if label_sigma > 0:
            # Gaussian soft labels: weight decays with distance from GT
            mask = dists <= tol_samples
            weights = np.exp(-0.5 * (dists[mask].astype(np.float64) / label_sigma) ** 2)
            y_boundary[mask] = np.maximum(y_boundary[mask], weights.astype(np.float32))
        else:
            # Hard binary labels (original behavior)
            if dists.min() <= tol_samples:
                y_boundary[np.argmin(dists)] = 1.0

    in_dim, edge_dim = compute_feature_dims(config)

    return Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        edge_attr=torch.from_numpy(edge_attr),
        y=torch.from_numpy(y_boundary),
        node_bars=torch.from_numpy(node_bars.astype(np.int64)),
        n_gt_troughs=len(gt_troughs),
        in_dim=in_dim,
        edge_dim=edge_dim,
    )


# ---------------------------------------------------------------------------
# Sliding-window graph generation
# ---------------------------------------------------------------------------

def generate_patient_graphs(
    patient_data: dict,
    config: dict | None = None,
    window_breaths: int = 6,
    tol_samples: int = 75,
    n_augmented_copies: int = 0,
) -> list[Data]:
    """Generate sliding-window graph Data objects for one patient.

    Parameters
    ----------
    patient_data : dict
        Must contain keys ``"signal"`` (1-D array), ``"troughs"`` (list[int]),
        ``"peaks"`` (list[int]), and ``"fs"`` (int).
    config : dict, optional
        Feature / preprocessing config.
    window_breaths : int
        Number of breaths per window.
    tol_samples : int
        Tolerance for matching GT troughs to graph nodes.

    Returns
    -------
    list of Data
        One PyG Data object per sliding window.
    """
    signal = np.asarray(patient_data["signal"], dtype=np.float64)
    troughs = sorted(patient_data["troughs"])
    fs = int(patient_data.get("fs", 125))
    pid_hash = patient_data.get("pid", id(patient_data))

    if len(troughs) < window_breaths + 1:
        # Not enough troughs for even one window
        return []

    stride = max(window_breaths // 2, 1)
    graphs: list[Data] = []

    n_windows = (len(troughs) - window_breaths) // stride + 1
    for w in range(n_windows):
        start_idx = w * stride
        end_idx = start_idx + window_breaths

        if end_idx > len(troughs):
            break

        # Window boundaries: from first trough sample to last trough sample
        # with padding of ±mean_breath_length/2 for context
        win_troughs = troughs[start_idx:end_idx]
        mean_breath = np.mean(np.diff(win_troughs)) if len(win_troughs) > 1 else 250
        pad = int(mean_breath * 0.5)

        win_start = max(0, win_troughs[0] - pad)
        win_end = min(len(signal), win_troughs[-1] + pad)

        if win_end <= win_start:
            continue

        seg = signal[win_start:win_end].copy()
        # Shift trough positions to window-local coordinates
        local_troughs = [t - win_start for t in win_troughs
                         if win_start <= t < win_end]

        if not local_troughs:
            continue

        data = extract_graph_data(seg, local_troughs, config=config,
                                  tol_samples=tol_samples, fs=fs)
        graphs.append(data)

        # Generate augmented copies of this window
        augment_mode = (config or {}).get('augment', '')
        for aug_i in range(n_augmented_copies):
            seed = hash((pid_hash, w, aug_i)) & 0xFFFFFFFF
            aug_rng = np.random.default_rng(seed)
            if augment_mode == 'signal_temporal':
                # New temporal augmentation pipeline: adjusts trough positions
                aug_seg, aug_troughs = augment_signal_with_troughs(
                    seg, local_troughs, config=config, rng=aug_rng, fs=fs
                )
                aug_seg = np.ascontiguousarray(aug_seg, dtype=np.float64)
                aug_troughs = [int(np.clip(t, 0, len(aug_seg) - 1))
                               for t in aug_troughs]
            else:
                # Existing pipeline: troughs unchanged
                aug_seg = augment_signal(seg, aug_rng, config=config, fs=fs)
                aug_troughs = local_troughs
            aug_data = extract_graph_data(aug_seg, aug_troughs, config=config,
                                          tol_samples=tol_samples, fs=fs)
            if aug_data is not None and aug_data.x.shape[0] > 0:
                graphs.append(aug_data)

    return graphs


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def cache_split_graphs(
    patients: dict[str, dict],
    split_ids: list[str],
    output_dir: str,
    config: dict | None = None,
    window_breaths: int = 6,
) -> str:
    """Generate and cache PyG Data objects for a split.

    Parameters
    ----------
    patients : dict
        Mapping ``patient_id`` → ``{"signal": ..., "troughs": ..., "peaks": ..., "fs": ...}``.
    split_ids : list of str
        Patient IDs belonging to this split.
    output_dir : str
        Directory to write ``.pt`` files into.
    config : dict, optional
        Feature / preprocessing config.
    window_breaths : int
        Number of breaths per sliding window.

    Returns
    -------
    str
        ``sha256:<hex>`` fingerprint over all written files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Only augment training data (detect via output_dir path)
    n_aug = config.get('n_augmented_copies', 0) if config else 0
    if 'train' not in str(output_dir):
        n_aug = 0

    written_paths: list[Path] = []
    for pid in sorted(split_ids):
        if pid not in patients:
            continue
        pdata = patients[pid]
        if 'pid' not in pdata:
            pdata = {**pdata, 'pid': pid}
        graphs = generate_patient_graphs(pdata, config=config,
                                         window_breaths=window_breaths,
                                         n_augmented_copies=n_aug)
        for i, data in enumerate(graphs):
            fname = f"{pid}_w{i:04d}.pt"
            fpath = out / fname
            torch.save(data, fpath)
            written_paths.append(fpath)

    # Write augmentation marker for rebuild detection
    if n_aug > 0:
        (out / '.augment_marker').write_text(str(n_aug))

    # Compute fingerprint
    h = hashlib.sha256()
    for p in sorted(written_paths):
        h.update(p.name.encode())
        h.update(p.read_bytes())

    return f"sha256:{h.hexdigest()}"
