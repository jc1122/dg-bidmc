"""Shared constants, paths, IO helpers, and metric functions for the DG constituency benchmark."""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

# ── SCFG grammar constants (from SPEC.md) ────────────────────────────────────

FS = 2000           # samples per second
SEED = 42           # fully deterministic
N_SENTENCES = 500   # full dataset (250 Impulse + 250 Correction)
N_SMOKE = 10        # tiny debug dataset for local tests
D_MIN = 200         # minimum terminal duration (bars)
D_MAX = 600         # maximum terminal duration (bars)
A_UP = 1.0          # upswing amplitude
A_DOWN = 0.7        # downswing amplitude
F_FINE = 8.0        # Hz — fine oscillation frequency inside terminals
SNR_DB = 20.0       # default signal-to-noise ratio in dB (20 dB = moderate noise)

# Impulse: 9 terminals arranged Up Down Up Down Up Down Up Down Up
IMPULSE_LABELS = ["Up", "Down", "Up", "Down", "Up", "Down", "Up", "Down", "Up"]
# Correction: 3 terminals arranged Down Up Down
CORRECTION_LABELS = ["Down", "Up", "Down"]

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
DATASET_PATH = RESULTS_DIR / "dataset.npz"
METRICS_PATH = RESULTS_DIR / "metrics.json"
COMPARISON_PLOT_PATH = RESULTS_DIR / "comparison_plot.png"
PARSE_OVERLAY_PATH = RESULTS_DIR / "parse_tree_examples.png"


def ensure_results_dir() -> Path:
    """Create the results directory if it does not exist."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def dataset_path() -> Path:
    return DATASET_PATH


def metrics_path() -> Path:
    return METRICS_PATH


# ── Dataset IO ────────────────────────────────────────────────────────────────

def save_dataset(
    signals: list[np.ndarray],
    labels: list[str],
    terminal_starts: list[np.ndarray],
    terminal_ends: list[np.ndarray],
    terminal_labels: list[list[str]],
    phrase_starts: list[np.ndarray],
    phrase_ends: list[np.ndarray],
    path: Path | None = None,
) -> Path:
    """Save generated dataset to NPZ.

    Each sentence can have a different length, so we store signals and per-sentence
    arrays as object arrays and metadata as JSON-encoded strings.
    """
    path = path or DATASET_PATH
    ensure_results_dir()
    np.savez(
        path,
        signals=np.array(signals, dtype=object),
        labels=np.array(labels),
        terminal_starts=np.array(terminal_starts, dtype=object),
        terminal_ends=np.array(terminal_ends, dtype=object),
        terminal_labels=np.array([json.dumps(tl) for tl in terminal_labels]),
        phrase_starts=np.array(phrase_starts, dtype=object),
        phrase_ends=np.array(phrase_ends, dtype=object),
    )
    return path


def load_dataset(path: Path | None = None) -> dict[str, Any]:
    """Load dataset from NPZ, returning a dict with native Python lists."""
    path = path or DATASET_PATH
    data = np.load(path, allow_pickle=True)
    n = len(data["labels"])
    return {
        "signals": [data["signals"][i] for i in range(n)],
        "labels": list(data["labels"]),
        "terminal_starts": [data["terminal_starts"][i] for i in range(n)],
        "terminal_ends": [data["terminal_ends"][i] for i in range(n)],
        "terminal_labels": [json.loads(s) for s in data["terminal_labels"]],
        "phrase_starts": [data["phrase_starts"][i] for i in range(n)],
        "phrase_ends": [data["phrase_ends"][i] for i in range(n)],
    }


# ── Metrics IO ────────────────────────────────────────────────────────────────

def save_metrics(metrics: dict[str, Any], path: Path | None = None) -> Path:
    """Write benchmark metrics to JSON."""
    path = path or METRICS_PATH
    ensure_results_dir()
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    return path


def load_metrics(path: Path | None = None) -> dict[str, Any]:
    """Read benchmark metrics from JSON."""
    path = path or METRICS_PATH
    with open(path) as f:
        return json.load(f)


# ── Metric functions ──────────────────────────────────────────────────────────

def sentence_accuracy(y_true: list[str], y_pred: list[str]) -> float:
    """Fraction of correctly classified sentences."""
    if not y_true:
        return 0.0
    return sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true)


def _match_boundaries(
    pred_starts: np.ndarray,
    true_starts: np.ndarray,
    true_durations: np.ndarray,
    tolerance_frac: float = 0.10,
    tolerance_floor: int = 20,
) -> int:
    """Greedy bipartite matching of predicted to true boundary starts.

    Pairs are sorted by distance and matched closest-first so that
    both precision and recall use the same optimal-greedy alignment.

    Returns the number of matched pairs.
    """
    if len(pred_starts) == 0 or len(true_starts) == 0:
        return 0

    pairs = []
    for i, (ts, dur) in enumerate(zip(true_starts, true_durations)):
        tol = max(tolerance_frac * dur, tolerance_floor)
        for j, ps in enumerate(pred_starts):
            dist = abs(float(ps) - float(ts))
            if dist <= tol:
                pairs.append((dist, i, j))

    pairs.sort()
    used_true: set[int] = set()
    used_pred: set[int] = set()
    matched = 0
    for _, i, j in pairs:
        if i not in used_true and j not in used_pred:
            matched += 1
            used_true.add(i)
            used_pred.add(j)

    return matched


def boundary_recall(
    pred_starts: np.ndarray,
    pred_ends: np.ndarray,
    true_starts: np.ndarray,
    true_ends: np.ndarray,
    true_durations: np.ndarray | None = None,
    tolerance_frac: float = 0.10,
    tolerance_floor: int = 20,
) -> float:
    """Fraction of true boundaries matched by a predicted boundary.

    A predicted boundary start is correct if it falls within
    ±max(tolerance_frac * true_terminal_duration, tolerance_floor) of
    the corresponding true boundary start.
    """
    if len(true_starts) == 0:
        return 0.0
    if len(pred_starts) == 0:
        return 0.0
    if true_durations is None:
        true_durations = true_ends - true_starts
    matched = _match_boundaries(
        pred_starts, true_starts, true_durations, tolerance_frac, tolerance_floor,
    )
    return matched / len(true_starts)


# Backward-compatible alias — boundary_accuracy was recall-only before.
boundary_accuracy = boundary_recall


def boundary_precision(
    pred_starts: np.ndarray,
    pred_ends: np.ndarray,
    true_starts: np.ndarray,
    true_ends: np.ndarray,
    true_durations: np.ndarray | None = None,
    tolerance_frac: float = 0.10,
    tolerance_floor: int = 20,
) -> float:
    """Fraction of predicted boundaries that match a true boundary."""
    if len(pred_starts) == 0:
        return 0.0
    if len(true_starts) == 0:
        return 0.0
    if true_durations is None:
        true_durations = true_ends - true_starts
    matched = _match_boundaries(
        pred_starts, true_starts, true_durations, tolerance_frac, tolerance_floor,
    )
    return matched / len(pred_starts)


def boundary_f1(
    pred_starts: np.ndarray,
    pred_ends: np.ndarray,
    true_starts: np.ndarray,
    true_ends: np.ndarray,
    true_durations: np.ndarray | None = None,
    tolerance_frac: float = 0.10,
    tolerance_floor: int = 20,
) -> float:
    """Boundary F1 score: harmonic mean of precision and recall.

    Uses greedy distance-sorted matching with per-terminal tolerance
    (±max(tolerance_frac * duration, tolerance_floor)).
    """
    if len(true_starts) == 0 or len(pred_starts) == 0:
        return 0.0
    if true_durations is None:
        true_durations = true_ends - true_starts
    matched = _match_boundaries(
        pred_starts, true_starts, true_durations, tolerance_frac, tolerance_floor,
    )
    if matched == 0:
        return 0.0
    precision = matched / len(pred_starts)
    recall = matched / len(true_starts)
    return 2 * precision * recall / (precision + recall)


def label_accuracy(pred_labels: list[str], true_labels: list[str]) -> float:
    """Fraction of correctly predicted terminal labels (position-aligned)."""
    if not true_labels:
        return 0.0
    n = min(len(pred_labels), len(true_labels))
    if n == 0:
        return 0.0
    return sum(p == t for p, t in zip(pred_labels[:n], true_labels[:n])) / len(true_labels)


# ── Timing helper ─────────────────────────────────────────────────────────────

@contextmanager
def timer():
    """Context manager that records wall-clock seconds in .elapsed attribute."""
    class _Timer:
        elapsed: float = 0.0
    t = _Timer()
    start = time.perf_counter()
    try:
        yield t
    finally:
        t.elapsed = time.perf_counter() - start


# ── Method result schema ──────────────────────────────────────────────────────

def make_method_result(
    name: str,
    sentence_preds: list[str] | None = None,
    boundary_preds: list[dict] | None = None,
    label_preds: list[list[str]] | None = None,
    timing_sec: float = 0.0,
    diagnostics: dict | None = None,
) -> dict[str, Any]:
    """Canonical result dict every benchmark method must produce."""
    return {
        "name": name,
        "sentence_preds": sentence_preds,
        "boundary_preds": boundary_preds,
        "label_preds": label_preds,
        "timing_sec": timing_sec,
        "diagnostics": diagnostics or {},
    }
