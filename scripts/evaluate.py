"""Rigorous evaluation harness for DG-GNN respiratory boundary detection.

Reports:
- graph-macro boundary F1 at multiple temporal tolerances
- patient-macro boundary F1
- event precision / recall / F1
- matched-boundary timing error
- respiratory-rate MAE / RMSE / bias / LoA / correlation

This keeps the legacy ``boundary_f1_600ms`` and ``rate_mae_bpm`` keys for
backward compatibility, but adds stricter and more clinically standard metrics.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from graph_features import compute_feature_dims
from model import build_model
from train import _apply_post_processing, _collect_predictions, evaluate as eval_model


DEFAULT_TOLERANCES = {"600ms": 75, "300ms": 38}


def _patient_id_from_graph_id(graph_id: str) -> str:
    return graph_id.rsplit("_w", 1)[0]


def _load_graphs_with_ids(data_dir: str) -> tuple[list, list[str]]:
    files = sorted(Path(data_dir).glob("*.pt"))
    graphs = [torch.load(path, weights_only=False) for path in files]
    graph_ids = [path.stem for path in files]
    return graphs, graph_ids


def _safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(values))


def _compute_rate_bpm(troughs: np.ndarray, fs: int = 125) -> float | None:
    troughs = np.asarray(troughs, dtype=np.float64)
    if troughs.size < 2:
        return None
    intervals = np.diff(np.sort(troughs)) / float(fs)
    mean_interval = float(np.mean(intervals))
    if mean_interval <= 0:
        return None
    return 60.0 / mean_interval


def _match_events(
    pred_troughs: np.ndarray | list[float],
    gt_troughs: np.ndarray | list[float],
    tol_samples: int,
) -> list[tuple[float, float, float]]:
    pred = np.asarray(pred_troughs, dtype=np.float64)
    gt = np.asarray(gt_troughs, dtype=np.float64)
    if pred.size == 0 or gt.size == 0:
        return []

    pairs: list[tuple[float, int, int]] = []
    for gt_idx, gt_val in enumerate(gt):
        for pred_idx, pred_val in enumerate(pred):
            dist = abs(float(pred_val) - float(gt_val))
            if dist <= tol_samples:
                pairs.append((dist, pred_idx, gt_idx))
    pairs.sort()

    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matches: list[tuple[float, float, float]] = []
    for dist, pred_idx, gt_idx in pairs:
        if pred_idx in used_pred or gt_idx in used_gt:
            continue
        used_pred.add(pred_idx)
        used_gt.add(gt_idx)
        matches.append((float(pred[pred_idx]), float(gt[gt_idx]), float(dist)))
    return matches


def compute_boundary_event_metrics(
    pred_troughs: np.ndarray | list[float],
    gt_troughs: np.ndarray | list[float],
    tol_samples: int,
) -> dict:
    """Compute event-based detection and timing metrics for one graph/window."""
    pred = np.asarray(pred_troughs, dtype=np.float64)
    gt = np.asarray(gt_troughs, dtype=np.float64)
    matches = _match_events(pred, gt, tol_samples)
    abs_errors = [dist for _, _, dist in matches]
    matched = len(matches)
    false_positives = int(pred.size - matched)
    false_negatives = int(gt.size - matched)

    precision = matched / pred.size if pred.size > 0 else 0.0
    recall = matched / gt.size if gt.size > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "matched": matched,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "abs_errors_samples": abs_errors,
        "mean_abs_error_samples": _safe_mean(abs_errors),
        "median_abs_error_samples": _safe_median(abs_errors),
    }


def compute_rate_error_metrics(
    pred_rates: np.ndarray | list[float],
    gt_rates: np.ndarray | list[float],
) -> dict:
    """Compute standard RR estimation metrics over a set of windows."""
    pred = np.asarray(pred_rates, dtype=np.float64)
    gt = np.asarray(gt_rates, dtype=np.float64)
    if pred.size != gt.size:
        raise ValueError("pred_rates and gt_rates must have the same length")

    finite = np.isfinite(pred) & np.isfinite(gt)
    pred = pred[finite]
    gt = gt[finite]
    if pred.size == 0:
        return {
            "n": 0,
            "mae_bpm": None,
            "rmse_bpm": None,
            "bias_bpm": None,
            "loa_low_bpm": None,
            "loa_high_bpm": None,
            "pearson_r": None,
        }

    diffs = pred - gt
    mae = float(np.mean(np.abs(diffs)))
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs))
    loa_low = float(bias - 1.96 * sd)
    loa_high = float(bias + 1.96 * sd)

    if pred.size >= 2 and np.std(pred) > 0 and np.std(gt) > 0:
        pearson_r = float(np.corrcoef(pred, gt)[0, 1])
    else:
        pearson_r = None

    return {
        "n": int(pred.size),
        "mae_bpm": mae,
        "rmse_bpm": rmse,
        "bias_bpm": bias,
        "loa_low_bpm": loa_low,
        "loa_high_bpm": loa_high,
        "pearson_r": pearson_r,
    }


def evaluate_graph_predictions(
    graph_predictions: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    graph_ids: list[str],
    threshold: float,
    nms_dist: int,
    top_n: int,
    fs: int = 125,
    tolerances: dict[str, int] | None = None,
) -> dict:
    """Evaluate already-collected per-graph predictions for one split."""
    if len(graph_predictions) != len(graph_ids):
        raise ValueError("graph_predictions and graph_ids must have the same length")

    tolerances = tolerances or DEFAULT_TOLERANCES
    per_graph_f1s: dict[str, list[float]] = {name: [] for name in tolerances}
    per_patient_f1s: dict[str, dict[str, list[float]]] = {
        name: defaultdict(list) for name in tolerances
    }
    total_counts: dict[str, dict[str, int]] = {
        name: {"matched": 0, "false_positives": 0, "false_negatives": 0}
        for name in tolerances
    }
    total_abs_errors: dict[str, list[float]] = {name: [] for name in tolerances}

    pred_rates: list[float] = []
    gt_rates: list[float] = []
    by_patient_rates: dict[str, list[float]] = defaultdict(list)
    by_patient_gt_rates: dict[str, list[float]] = defaultdict(list)

    for graph_id, entry in zip(graph_ids, graph_predictions):
        g_scores = np.asarray(entry[0], dtype=np.float64)
        g_bars = np.asarray(entry[1], dtype=np.float64)
        g_labels = np.asarray(entry[2], dtype=np.float64)
        g_feats = np.asarray(entry[3], dtype=np.float64) if len(entry) > 3 else None

        pred_troughs = _apply_post_processing(
            g_scores,
            g_bars,
            threshold,
            nms_dist,
            top_n,
            g_features=g_feats,
        )
        gt_troughs = g_bars[g_labels > 0.5]
        patient_id = _patient_id_from_graph_id(graph_id)

        for tol_name, tol_samples in tolerances.items():
            metrics = compute_boundary_event_metrics(pred_troughs, gt_troughs, tol_samples)
            per_graph_f1s[tol_name].append(metrics["f1"])
            per_patient_f1s[tol_name][patient_id].append(metrics["f1"])
            total_counts[tol_name]["matched"] += metrics["matched"]
            total_counts[tol_name]["false_positives"] += metrics["false_positives"]
            total_counts[tol_name]["false_negatives"] += metrics["false_negatives"]
            total_abs_errors[tol_name].extend(metrics["abs_errors_samples"])

        pred_rate = _compute_rate_bpm(pred_troughs, fs)
        gt_rate = _compute_rate_bpm(gt_troughs, fs)
        if pred_rate is not None and gt_rate is not None:
            pred_rates.append(pred_rate)
            gt_rates.append(gt_rate)
            by_patient_rates[patient_id].append(pred_rate)
            by_patient_gt_rates[patient_id].append(gt_rate)

    results: dict[str, object] = {
        "n_graphs": len(graph_predictions),
        "n_patients": len({_patient_id_from_graph_id(gid) for gid in graph_ids}),
        "threshold": float(threshold),
        "nms_dist": int(nms_dist),
        "top_n": int(top_n),
    }

    for tol_name, _ in tolerances.items():
        matched = total_counts[tol_name]["matched"]
        false_positives = total_counts[tol_name]["false_positives"]
        false_negatives = total_counts[tol_name]["false_negatives"]

        event_precision = (
            matched / (matched + false_positives)
            if (matched + false_positives) > 0
            else 0.0
        )
        event_recall = (
            matched / (matched + false_negatives)
            if (matched + false_negatives) > 0
            else 0.0
        )
        if event_precision + event_recall == 0:
            event_f1 = 0.0
        else:
            event_f1 = 2.0 * event_precision * event_recall / (event_precision + event_recall)

        patient_macro = _safe_mean(
            [float(np.mean(values)) for values in per_patient_f1s[tol_name].values()]
        )
        mae_samples = _safe_mean(total_abs_errors[tol_name])
        med_samples = _safe_median(total_abs_errors[tol_name])

        results[f"boundary_f1_{tol_name}"] = float(np.mean(per_graph_f1s[tol_name])) if per_graph_f1s[tol_name] else 0.0
        results[f"graph_macro_f1_{tol_name}"] = results[f"boundary_f1_{tol_name}"]
        results[f"patient_macro_f1_{tol_name}"] = patient_macro
        results[f"event_precision_{tol_name}"] = float(event_precision)
        results[f"event_recall_{tol_name}"] = float(event_recall)
        results[f"event_f1_{tol_name}"] = float(event_f1)
        results[f"matched_events_{tol_name}"] = matched
        results[f"false_positives_{tol_name}"] = false_positives
        results[f"false_negatives_{tol_name}"] = false_negatives
        results[f"boundary_mae_samples_{tol_name}"] = mae_samples
        results[f"boundary_mae_ms_{tol_name}"] = None if mae_samples is None else float(mae_samples * 1000.0 / fs)
        results[f"boundary_median_ms_{tol_name}"] = None if med_samples is None else float(med_samples * 1000.0 / fs)

    rate_metrics = compute_rate_error_metrics(pred_rates, gt_rates)
    results["rate_mae_bpm"] = rate_metrics["mae_bpm"]
    results["rate_rmse_bpm"] = rate_metrics["rmse_bpm"]
    results["rate_bias_bpm"] = rate_metrics["bias_bpm"]
    results["rate_loa_low_bpm"] = rate_metrics["loa_low_bpm"]
    results["rate_loa_high_bpm"] = rate_metrics["loa_high_bpm"]
    results["rate_pearson_r"] = rate_metrics["pearson_r"]
    results["rate_eval_windows"] = rate_metrics["n"]

    by_patient: dict[str, dict[str, float | int | None]] = {}
    for patient_id in sorted({_patient_id_from_graph_id(gid) for gid in graph_ids}):
        patient_row: dict[str, float | int | None] = {
            "n_graphs": len(per_patient_f1s[next(iter(tolerances))][patient_id]),
        }
        for tol_name in tolerances:
            patient_row[f"boundary_f1_{tol_name}"] = _safe_mean(per_patient_f1s[tol_name][patient_id])
        patient_rate = compute_rate_error_metrics(
            by_patient_rates.get(patient_id, []),
            by_patient_gt_rates.get(patient_id, []),
        )
        patient_row["rate_mae_bpm"] = patient_rate["mae_bpm"]
        by_patient[patient_id] = patient_row
    results["by_patient"] = by_patient

    return results


def _load_benchmark_targets(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text())


def evaluate_model_on_split(
    model_path: str,
    config: dict,
    data_dir: str,
    device: str = "cuda",
    threshold: float = 0.5,
    nms_dist: int = 0,
    top_n: int = 0,
    fs: int = 125,
    tolerances: dict[str, int] | None = None,
) -> dict:
    """Load a saved model and evaluate it on one split."""
    in_dim, edge_dim = compute_feature_dims(config)
    model_config = {**config, "in_dim": in_dim, "edge_dim": edge_dim}
    model = build_model(model_config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    graphs, graph_ids = _load_graphs_with_ids(data_dir)
    loader = DataLoader(graphs, batch_size=32)

    # Preserve a legacy-compatible loss surface for quick smoke comparison.
    legacy = eval_model(
        model,
        loader,
        device,
        threshold=threshold,
        tol_samples=(tolerances or DEFAULT_TOLERANCES)["600ms"],
        fs=fs,
        nms_dist=nms_dist,
        top_n=top_n,
    )
    graph_predictions = _collect_predictions(model, loader, device)
    results = evaluate_graph_predictions(
        graph_predictions=graph_predictions,
        graph_ids=graph_ids,
        threshold=threshold,
        nms_dist=nms_dist,
        top_n=top_n,
        fs=fs,
        tolerances=tolerances,
    )
    results["loss"] = legacy["loss"]
    return results


def evaluate_all(
    model_path: str,
    config: dict,
    test_dir: str,
    test_adv_dir: str | None = None,
    device: str = "cuda",
    threshold: float = 0.5,
    nms_dist: int = 0,
    top_n: int = 0,
    fs: int = 125,
    tolerances: dict[str, int] | None = None,
    benchmark_targets_path: str | None = None,
) -> dict:
    """Full evaluation with rigorous split-level metrics and optional targets."""
    tolerances = tolerances or DEFAULT_TOLERANCES

    results = {
        "protocol": {
            "selection_metric": "boundary_f1_600ms on validation splits",
            "final_report_metric": "held-out test metrics",
            "tolerances": tolerances,
        }
    }
    results["test"] = evaluate_model_on_split(
        model_path,
        config,
        test_dir,
        device=device,
        threshold=threshold,
        nms_dist=nms_dist,
        top_n=top_n,
        fs=fs,
        tolerances=tolerances,
    )
    if test_adv_dir:
        results["test_adversarial"] = evaluate_model_on_split(
            model_path,
            config,
            test_adv_dir,
            device=device,
            threshold=threshold,
            nms_dist=nms_dist,
            top_n=top_n,
            fs=fs,
            tolerances=tolerances,
        )
        results["aggregate_f1_600ms"] = (
            0.55 * results["test"]["boundary_f1_600ms"]
            + 0.45 * results["test_adversarial"]["boundary_f1_600ms"]
        )
        results["aggregate_patient_macro_f1_600ms"] = (
            0.55 * results["test"]["patient_macro_f1_600ms"]
            + 0.45 * results["test_adversarial"]["patient_macro_f1_600ms"]
        )
    else:
        results["aggregate_f1_600ms"] = results["test"]["boundary_f1_600ms"]
        results["aggregate_patient_macro_f1_600ms"] = results["test"]["patient_macro_f1_600ms"]

    benchmark_targets = _load_benchmark_targets(benchmark_targets_path)
    if benchmark_targets is not None:
        results["benchmark_targets"] = benchmark_targets

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model state dict")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--test-dir", required=True)
    parser.add_argument("--test-adv-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--nms", type=int, default=0)
    parser.add_argument("--top-n", type=int, default=0)
    parser.add_argument("--benchmark-targets", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    results = evaluate_all(
        model_path=args.model,
        config=config,
        test_dir=args.test_dir,
        test_adv_dir=args.test_adv_dir,
        device=args.device,
        threshold=args.threshold,
        nms_dist=args.nms,
        top_n=args.top_n,
        benchmark_targets_path=args.benchmark_targets,
    )

    print(json.dumps(results, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2))
