#!/usr/bin/env python3
"""BIDMC PhysioNet data loader with patient profiling and train/val/test splits.

Loads impedance pneumography signals from 53 ICU patients, extracts
ground-truth exhale troughs between annotated inhale peaks, profiles each
patient for adversarial characteristics, and provides reproducible splits.

Usage:
    python scripts/data_loader.py          # profile all patients, print table, save JSON
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt

BIDMC_FS = 125
BIDMC_N_PATIENTS = 53

# LP filter for ground-truth trough extraction (2 Hz cutoff at 125 Hz)
_B_LP, _A_LP = butter(4, 2.0 / (BIDMC_FS / 2), "low")

# Adversarial thresholds
_DRIFT_THRESH = 0.10
_CV_THRESH = 0.15
_SIGH_THRESH = 0.10

# Hard-coded reproducible split (seed=42, stratified by adversarial flag).
# bidmc05 excluded: only 3 troughs — unusable for training.
_SPLITS: dict[str, list[str]] = {
    "train": [
        "bidmc01", "bidmc02", "bidmc03", "bidmc04", "bidmc06", "bidmc07",
        "bidmc08", "bidmc09", "bidmc10", "bidmc18", "bidmc20", "bidmc21",
        "bidmc22", "bidmc23", "bidmc26", "bidmc28", "bidmc30", "bidmc32",
        "bidmc33", "bidmc34", "bidmc35", "bidmc37", "bidmc38", "bidmc39",
        "bidmc41", "bidmc42", "bidmc43", "bidmc45", "bidmc47", "bidmc48",
        "bidmc49", "bidmc50", "bidmc51", "bidmc52",
    ],
    "val": [
        "bidmc12", "bidmc14", "bidmc16", "bidmc17", "bidmc31", "bidmc36",
        "bidmc40", "bidmc53",
    ],
    "test": [
        "bidmc11", "bidmc13", "bidmc15", "bidmc19", "bidmc24", "bidmc25",
        "bidmc27", "bidmc29", "bidmc44", "bidmc46",
    ],
    "excluded": ["bidmc05"],
}


def patient_ids() -> list[str]:
    """All 53 patient IDs in order."""
    return [f"bidmc{i:02d}" for i in range(1, BIDMC_N_PATIENTS + 1)]


def load_patient(patient_id: str) -> dict:
    """Load a single BIDMC patient record.

    Downloads on first access via ``pn_dir='bidmc'``.

    Returns dict with keys:
        signal    – raw impedance pneumography (channel 0), float64
        resp_lp   – LP-filtered (2 Hz) demeaned signal, contiguous float64
        peaks     – inhale peak sample indices (every 2nd annotation)
        troughs   – exhale trough sample indices (argmin between peaks)
        fs        – sampling frequency (125)
        patient_id
    """
    import wfdb

    rec = wfdb.rdrecord(patient_id, pn_dir="bidmc")
    ann = wfdb.rdann(patient_id, "breath", pn_dir="bidmc")

    signal = np.asarray(rec.p_signal[:, 0], dtype=np.float64)
    fs = int(rec.fs)

    peaks = ann.sample[::2]

    resp_lp = np.ascontiguousarray(
        filtfilt(_B_LP, _A_LP, signal - signal.mean())
    )

    troughs = _extract_troughs(resp_lp, peaks)

    return {
        "signal": signal,
        "resp_lp": resp_lp,
        "peaks": peaks,
        "troughs": troughs,
        "fs": fs,
        "patient_id": patient_id,
    }


def _extract_troughs(resp_lp: np.ndarray, peaks: np.ndarray) -> list[int]:
    """Argmin between consecutive inhale peaks on LP-filtered signal."""
    troughs: list[int] = []
    for i in range(len(peaks) - 1):
        p1, p2 = int(peaks[i]), int(peaks[i + 1])
        if p2 <= p1:
            continue
        trough = p1 + int(np.argmin(resp_lp[p1:p2]))
        troughs.append(trough)
    return troughs


def get_gt_troughs(patient_data: dict) -> list[int]:
    """Extract ground truth trough positions from loaded patient data."""
    return list(patient_data["troughs"])


def profile_patient(patient_data: dict) -> dict:
    """Compute adversarial profile metrics for a patient.

    Returns dict with:
        n_breaths, n_troughs, mean_ibi_samples, mean_rr_bpm, cv_ibi,
        drift_cv, sigh_count, sigh_fraction, snr_db, signal_length,
        median_breath_dur, is_adversarial
    """
    signal = patient_data["signal"]
    peaks = patient_data["peaks"]
    troughs = patient_data["troughs"]
    resp_lp = patient_data["resp_lp"]
    fs = patient_data["fs"]

    n_breaths = len(peaks) - 1 if len(peaks) > 1 else 0
    n_troughs = len(troughs)

    # Inter-beat intervals (peak-to-peak)
    ibis = np.diff(peaks).astype(np.float64) if len(peaks) > 1 else np.array([1.0])
    mean_ibi = float(ibis.mean())
    cv_ibi = float(ibis.std() / mean_ibi) if mean_ibi > 0 else 0.0
    mean_rr_bpm = 60.0 * fs / mean_ibi if mean_ibi > 0 else 0.0

    # Drift severity: difference of thirds normalised by RMS
    n = len(signal)
    third = n // 3
    rms = float(np.sqrt(np.mean(signal ** 2))) + 1e-12
    drift_cv = float(abs(signal[:third].mean() - signal[-third:].mean()) / rms)

    # Sigh detection: breath amplitude > 2× median
    amplitudes = np.array([
        abs(resp_lp[int(peaks[i])] - resp_lp[int(troughs[i])])
        for i in range(min(n_breaths, n_troughs))
    ]) if n_troughs > 0 and n_breaths > 0 else np.array([0.0])
    med_amp = float(np.median(amplitudes)) if len(amplitudes) > 0 else 1.0
    sigh_count = int(np.sum(amplitudes > 2.0 * med_amp)) if med_amp > 0 else 0
    sigh_fraction = sigh_count / max(n_breaths, 1)

    # SNR estimate (signal power / noise power after LP)
    noise = signal - resp_lp
    sig_pow = float(np.mean(resp_lp ** 2)) + 1e-20
    noise_pow = float(np.mean(noise ** 2)) + 1e-20
    snr_db = 10.0 * np.log10(sig_pow / noise_pow)

    median_breath_dur = float(np.median(ibis))

    is_adversarial = (
        drift_cv > _DRIFT_THRESH
        or cv_ibi > _CV_THRESH
        or sigh_fraction > _SIGH_THRESH
    )

    return {
        "n_breaths": n_breaths,
        "n_troughs": n_troughs,
        "mean_ibi_samples": mean_ibi,
        "mean_rr_bpm": mean_rr_bpm,
        "cv_ibi": cv_ibi,
        "drift_cv": drift_cv,
        "sigh_count": sigh_count,
        "sigh_fraction": sigh_fraction,
        "snr_db": snr_db,
        "signal_length": len(signal),
        "median_breath_dur": median_breath_dur,
        "is_adversarial": is_adversarial,
    }


def get_splits(seed: int = 42) -> dict[str, list[str]]:
    """Return hard-coded reproducible train/val/test splits.

    The split was computed once with stratification by adversarial profile
    and then frozen for reproducibility.  ``seed`` is accepted for API
    compatibility but ignored (the split is deterministic).
    """
    return {k: list(v) for k, v in _SPLITS.items()}


def load_all_patients(verbose: bool = True) -> dict[str, dict]:
    """Load all 53 BIDMC patients, keyed by patient_id.

    Prints progress to stderr when *verbose* is True.
    """
    all_patients: dict[str, dict] = {}
    ids = patient_ids()
    for i, pid in enumerate(ids, 1):
        if verbose:
            print(
                f"\r  Loading {pid} [{i:2d}/{len(ids)}] ...",
                end="",
                flush=True,
                file=sys.stderr,
            )
        try:
            all_patients[pid] = load_patient(pid)
        except Exception as exc:
            print(
                f"\n  WARNING: failed to load {pid}: {exc}",
                file=sys.stderr,
            )
    if verbose:
        print(file=sys.stderr)
    return all_patients


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    print("Loading all BIDMC patients ...")
    patients = load_all_patients(verbose=True)
    print(f"Loaded {len(patients)} patients.\n")

    # Profile
    profiles: dict[str, dict] = {}
    for pid in sorted(patients):
        profiles[pid] = profile_patient(patients[pid])

    # Table header
    hdr = (
        f"{'Patient':<10} {'Breaths':>7} {'Troughs':>7} {'RR bpm':>7} "
        f"{'CV_IBI':>7} {'Drift':>7} {'Sighs':>7} {'Adv':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for pid in sorted(profiles):
        p = profiles[pid]
        adv_flag = " *" if p.get("is_adversarial", False) else ""
        print(
            f"{pid:<10} {p['n_breaths']:>7d} {p['n_troughs']:>7d} "
            f"{p['mean_rr_bpm']:>7.1f} {p['cv_ibi']:>7.3f} "
            f"{p['drift_cv']:>7.3f} {p['sigh_fraction']:>7.3f}{adv_flag}"
        )

    # Splits
    splits = get_splits()
    print(f"\nSplits  — train: {len(splits['train'])}, "
          f"val: {len(splits['val'])}, test: {len(splits['test'])}, "
          f"excluded: {len(splits.get('excluded', []))}")
    for split_name in ("train", "val", "test", "excluded"):
        ids_in_split = splits.get(split_name, [])
        n_adv = sum(
            1 for pid in ids_in_split
            if profiles.get(pid, {}).get("is_adversarial", False)
        )
        print(f"  {split_name:>8}: {len(ids_in_split):2d} patients, "
              f"{n_adv} adversarial")

    # Save profiles (without is_adversarial to match existing schema)
    save_profiles = {}
    for pid, p in profiles.items():
        save_profiles[pid] = {k: v for k, v in p.items() if k != "is_adversarial"}
    out_path = results_dir / "patient_profiles.json"
    with open(out_path, "w") as f:
        json.dump(save_profiles, f, indent=2)
    print(f"\nSaved profiles → {out_path}")


if __name__ == "__main__":
    _main()
