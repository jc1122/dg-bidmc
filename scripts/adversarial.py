"""Adversarial noise, distortion, and signal generators for stress-testing DG and baselines."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfilt

# Allow imports from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (
    FS, D_MIN, D_MAX, A_UP, A_DOWN, F_FINE, SNR_DB,
    IMPULSE_LABELS, CORRECTION_LABELS,
)
from generate_scfg import make_terminal, make_sentence


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scale_noise_to_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale *noise* so that the resulting SNR equals *snr_db* (dB)."""
    sig_rms = np.sqrt(np.mean(signal ** 2))
    if sig_rms < 1e-12:
        return noise
    noise_rms = np.sqrt(np.mean(noise ** 2))
    if noise_rms < 1e-12:
        return noise
    target_noise_rms = sig_rms / (10 ** (snr_db / 20))
    return noise * (target_noise_rms / noise_rms)


# ── 1. Colored noise ─────────────────────────────────────────────────────────

def add_pink_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add 1/f (pink) noise at the specified SNR."""
    n = len(signal)
    white = rng.normal(0, 1, n)
    freqs = np.fft.rfftfreq(n, d=1.0)
    spectrum = np.fft.rfft(white)
    # Scale by 1/sqrt(f); skip DC to avoid division by zero
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * scale, n=n)
    return signal + _scale_noise_to_snr(signal, pink, snr_db)


def add_brown_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add 1/f² (Brownian) noise at the specified SNR."""
    white = rng.normal(0, 1, len(signal))
    brown = np.cumsum(white)
    brown -= brown.mean()
    return signal + _scale_noise_to_snr(signal, brown, snr_db)


def add_burst_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    burst_prob: float = 0.01,
    burst_amp_factor: float = 10.0,
) -> np.ndarray:
    """Add impulsive burst noise at the specified SNR."""
    mask = rng.random(len(signal)) < burst_prob
    bursts = mask.astype(np.float64) * rng.normal(0, burst_amp_factor, len(signal))
    return signal + _scale_noise_to_snr(signal, bursts, snr_db)


# ── 2. Band-limited noise ────────────────────────────────────────────────────

def add_bandlimited_noise(
    signal: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
    f_low: float = 5.0,
    f_high: float = 15.0,
    fs: int = 2000,
) -> np.ndarray:
    """Add band-limited noise filtered to [f_low, f_high] Hz at the specified SNR."""
    white = rng.normal(0, 1, len(signal))
    nyq = fs / 2.0
    sos = butter(4, [f_low / nyq, f_high / nyq], btype="band", output="sos")
    filtered = sosfilt(sos, white)
    return signal + _scale_noise_to_snr(signal, filtered, snr_db)


# ── 3. DC drift ──────────────────────────────────────────────────────────────

def add_dc_drift(signal: np.ndarray, drift_rate: float = 0.001) -> np.ndarray:
    """Add a linear DC drift ramp to the signal."""
    ramp = np.linspace(0, drift_rate * len(signal), len(signal))
    return signal + ramp


# ── 4. Harmonic interference ─────────────────────────────────────────────────

def add_harmonic_interference(
    signal: np.ndarray,
    fs: int = 2000,
    harmonics: list[float] | None = None,
    amplitudes: list[float] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add sinusoidal harmonic interference tones with random phase."""
    if harmonics is None:
        harmonics = [7.5, 8.5, 16.0]
    if amplitudes is None:
        amplitudes = [0.3, 0.3, 0.2]
    if rng is None:
        rng = np.random.default_rng()
    t = np.arange(len(signal)) / fs
    interference = np.zeros_like(signal, dtype=np.float64)
    for freq, amp in zip(harmonics, amplitudes):
        phase = rng.uniform(0, 2 * np.pi)
        interference += amp * np.sin(2 * np.pi * freq * t + phase)
    return signal + interference


# ── 5. Variable amplitude sentence ───────────────────────────────────────────

def make_variable_amplitude_sentence(
    sentence_type: str,
    rng: np.random.Generator,
    amp_range: tuple[float, float] = (0.1, 5.0),
    snr_db: float | None = None,
) -> dict:
    """Like make_sentence() but each terminal gets a random amplitude multiplier."""
    if sentence_type == "impulse":
        directions = [+1, -1, +1, -1, +1, -1, +1, -1, +1]
        base_amplitudes = [A_UP, A_DOWN, A_UP, A_DOWN, A_UP, A_DOWN, A_UP, A_DOWN, A_UP]
        labels = list(IMPULSE_LABELS)
        phrase_terminal_ranges = [(1, 4), (5, 8)]
    elif sentence_type == "correction":
        directions = [-1, +1, -1]
        base_amplitudes = [A_DOWN, A_UP, A_DOWN]
        labels = list(CORRECTION_LABELS)
        phrase_terminal_ranges = []
    else:
        raise ValueError(f"Unknown sentence type: {sentence_type}")

    segments, durations = [], []
    for direction, base_amp in zip(directions, base_amplitudes):
        d = int(rng.integers(D_MIN, D_MAX))
        multiplier = rng.uniform(amp_range[0], amp_range[1])
        durations.append(d)
        segments.append(make_terminal(direction, d, base_amp * multiplier, F_FINE, FS))

    cumulative = np.cumsum([0] + durations)
    terminal_starts = cumulative[:-1].astype(np.int32)
    terminal_ends = cumulative[1:].astype(np.int32)

    phrase_starts_list, phrase_ends_list = [], []
    for start_idx, end_idx in phrase_terminal_ranges:
        phrase_starts_list.append(int(terminal_starts[start_idx]))
        phrase_ends_list.append(int(terminal_ends[end_idx - 1]))

    signal = np.concatenate(segments).astype(np.float64)

    if snr_db is not None and snr_db != float("inf"):
        sig_rms = np.sqrt(np.mean(signal ** 2))
        if sig_rms > 1e-12:
            noise_std = sig_rms / (10 ** (snr_db / 20))
            signal = signal + rng.normal(0, noise_std, len(signal))

    return {
        "label": sentence_type,
        "signal": signal,
        "terminal_starts": terminal_starts,
        "terminal_ends": terminal_ends,
        "terminal_labels": labels,
        "phrase_starts": np.array(phrase_starts_list, dtype=np.int32),
        "phrase_ends": np.array(phrase_ends_list, dtype=np.int32),
    }


# ── 6. Short segment sentence ────────────────────────────────────────────────

def make_short_segment_sentence(
    sentence_type: str,
    rng: np.random.Generator,
    d_min: int = 20,
    d_max: int = 100,
    snr_db: float | None = None,
) -> dict:
    """Like make_sentence() but with overridden (shorter) segment durations."""
    if sentence_type == "impulse":
        directions = [+1, -1, +1, -1, +1, -1, +1, -1, +1]
        amplitudes = [A_UP, A_DOWN, A_UP, A_DOWN, A_UP, A_DOWN, A_UP, A_DOWN, A_UP]
        labels = list(IMPULSE_LABELS)
        phrase_terminal_ranges = [(1, 4), (5, 8)]
    elif sentence_type == "correction":
        directions = [-1, +1, -1]
        amplitudes = [A_DOWN, A_UP, A_DOWN]
        labels = list(CORRECTION_LABELS)
        phrase_terminal_ranges = []
    else:
        raise ValueError(f"Unknown sentence type: {sentence_type}")

    segments, durations = [], []
    for direction, amplitude in zip(directions, amplitudes):
        d = int(rng.integers(d_min, d_max))
        durations.append(d)
        segments.append(make_terminal(direction, d, amplitude, F_FINE, FS))

    cumulative = np.cumsum([0] + durations)
    terminal_starts = cumulative[:-1].astype(np.int32)
    terminal_ends = cumulative[1:].astype(np.int32)

    phrase_starts_list, phrase_ends_list = [], []
    for start_idx, end_idx in phrase_terminal_ranges:
        phrase_starts_list.append(int(terminal_starts[start_idx]))
        phrase_ends_list.append(int(terminal_ends[end_idx - 1]))

    signal = np.concatenate(segments).astype(np.float64)

    if snr_db is not None and snr_db != float("inf"):
        sig_rms = np.sqrt(np.mean(signal ** 2))
        if sig_rms > 1e-12:
            noise_std = sig_rms / (10 ** (snr_db / 20))
            signal = signal + rng.normal(0, noise_std, len(signal))

    return {
        "label": sentence_type,
        "signal": signal,
        "terminal_starts": terminal_starts,
        "terminal_ends": terminal_ends,
        "terminal_labels": labels,
        "phrase_starts": np.array(phrase_starts_list, dtype=np.int32),
        "phrase_ends": np.array(phrase_ends_list, dtype=np.int32),
    }


# ── 7. Extreme length ratio sentence ─────────────────────────────────────────

def make_extreme_ratio_sentence(
    sentence_type: str,
    rng: np.random.Generator,
    ratio: int = 20,
    snr_db: float | None = None,
) -> dict:
    """Like make_sentence() but alternating very short and very long terminals."""
    if sentence_type == "impulse":
        directions = [+1, -1, +1, -1, +1, -1, +1, -1, +1]
        amplitudes = [A_UP, A_DOWN, A_UP, A_DOWN, A_UP, A_DOWN, A_UP, A_DOWN, A_UP]
        labels = list(IMPULSE_LABELS)
        phrase_terminal_ranges = [(1, 4), (5, 8)]
    elif sentence_type == "correction":
        directions = [-1, +1, -1]
        amplitudes = [A_DOWN, A_UP, A_DOWN]
        labels = list(CORRECTION_LABELS)
        phrase_terminal_ranges = []
    else:
        raise ValueError(f"Unknown sentence type: {sentence_type}")

    d_short = max(20, D_MIN // ratio)
    d_long = d_short * ratio

    segments, durations = [], []
    for i, (direction, amplitude) in enumerate(zip(directions, amplitudes)):
        d = d_short if i % 2 == 0 else d_long
        durations.append(d)
        segments.append(make_terminal(direction, d, amplitude, F_FINE, FS))

    cumulative = np.cumsum([0] + durations)
    terminal_starts = cumulative[:-1].astype(np.int32)
    terminal_ends = cumulative[1:].astype(np.int32)

    phrase_starts_list, phrase_ends_list = [], []
    for start_idx, end_idx in phrase_terminal_ranges:
        phrase_starts_list.append(int(terminal_starts[start_idx]))
        phrase_ends_list.append(int(terminal_ends[end_idx - 1]))

    signal = np.concatenate(segments).astype(np.float64)

    if snr_db is not None and snr_db != float("inf"):
        sig_rms = np.sqrt(np.mean(signal ** 2))
        if sig_rms > 1e-12:
            noise_std = sig_rms / (10 ** (snr_db / 20))
            signal = signal + rng.normal(0, noise_std, len(signal))

    return {
        "label": sentence_type,
        "signal": signal,
        "terminal_starts": terminal_starts,
        "terminal_ends": terminal_ends,
        "terminal_labels": labels,
        "phrase_starts": np.array(phrase_starts_list, dtype=np.int32),
        "phrase_ends": np.array(phrase_ends_list, dtype=np.int32),
    }


# ── All-in-one config ────────────────────────────────────────────────────────

@dataclass
class AdversarialConfig:
    """Configuration for adversarial signal generation."""

    noise_type: str = "white"  # white, pink, brown, burst, bandlimited
    snr_db: float | None = 20.0
    dc_drift_rate: float = 0.0
    harmonic_interference: bool = False
    amp_range: tuple[float, float] = (1.0, 1.0)
    d_min: int = D_MIN
    d_max: int = D_MAX
    length_ratio: float | None = None


_NOISE_FUNCTIONS = {
    "pink": add_pink_noise,
    "brown": add_brown_noise,
    "burst": add_burst_noise,
    "bandlimited": add_bandlimited_noise,
}


def apply_adversarial(
    signal: np.ndarray, config: AdversarialConfig, rng: np.random.Generator
) -> np.ndarray:
    """Apply all configured adversarial conditions to a clean signal."""
    out = signal.copy()

    # Noise
    if config.snr_db is not None and config.snr_db != float("inf"):
        if config.noise_type == "white":
            sig_rms = np.sqrt(np.mean(out ** 2))
            if sig_rms > 1e-12:
                noise_std = sig_rms / (10 ** (config.snr_db / 20))
                out = out + rng.normal(0, noise_std, len(out))
        elif config.noise_type in _NOISE_FUNCTIONS:
            out = _NOISE_FUNCTIONS[config.noise_type](out, config.snr_db, rng)

    # DC drift
    if config.dc_drift_rate != 0.0:
        out = add_dc_drift(out, config.dc_drift_rate)

    # Harmonic interference
    if config.harmonic_interference:
        out = add_harmonic_interference(out, rng=rng)

    return out


def make_adversarial_sentence(
    sentence_type: str, rng: np.random.Generator, config: AdversarialConfig
) -> dict:
    """Generate a sentence with adversarial conditions (both structural and signal distortion)."""
    # Structural variations
    if config.length_ratio is not None:
        result = make_extreme_ratio_sentence(
            sentence_type, rng, ratio=int(config.length_ratio), snr_db=None,
        )
    elif config.amp_range != (1.0, 1.0):
        result = make_variable_amplitude_sentence(
            sentence_type, rng, amp_range=config.amp_range, snr_db=None,
        )
    elif config.d_min != D_MIN or config.d_max != D_MAX:
        result = make_short_segment_sentence(
            sentence_type, rng, d_min=config.d_min, d_max=config.d_max, snr_db=None,
        )
    else:
        result = make_sentence(sentence_type, rng, snr_db=None)

    # Signal distortion
    result["signal"] = apply_adversarial(result["signal"], config, rng)
    return result
