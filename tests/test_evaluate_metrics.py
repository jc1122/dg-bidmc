from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import (
    compute_boundary_event_metrics,
    compute_rate_error_metrics,
    evaluate_graph_predictions,
)


def test_compute_boundary_event_metrics_reports_detection_and_timing():
    metrics = compute_boundary_event_metrics(
        pred_troughs=[10, 50],
        gt_troughs=[12, 30],
        tol_samples=3,
    )

    assert metrics["matched"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["mean_abs_error_samples"] == pytest.approx(2.0)
    assert metrics["median_abs_error_samples"] == pytest.approx(2.0)


def test_compute_rate_error_metrics_reports_mae_rmse_bias_loa_and_correlation():
    metrics = compute_rate_error_metrics(
        pred_rates=[12.0, 14.0, 16.0],
        gt_rates=[10.0, 15.0, 20.0],
    )

    diffs = [2.0, -1.0, -4.0]
    expected_bias = sum(diffs) / len(diffs)
    expected_mae = sum(abs(d) for d in diffs) / len(diffs)
    expected_rmse = (sum(d * d for d in diffs) / len(diffs)) ** 0.5

    assert metrics["n"] == 3
    assert metrics["mae_bpm"] == pytest.approx(expected_mae)
    assert metrics["rmse_bpm"] == pytest.approx(expected_rmse)
    assert metrics["bias_bpm"] == pytest.approx(expected_bias)
    assert metrics["loa_low_bpm"] < metrics["bias_bpm"] < metrics["loa_high_bpm"]
    assert metrics["pearson_r"] == pytest.approx(1.0)


def test_evaluate_graph_predictions_reports_multi_tolerance_and_patient_macro():
    graph_predictions = [
        (
            [0.1, 0.95, 0.2],
            [100, 200, 300],
            [0.0, 1.0, 0.0],
        ),
        (
            [0.1, 0.9, 0.2, 0.1],
            [100, 200, 300, 400],
            [0.0, 1.0, 0.0, 1.0],
        ),
    ]

    metrics = evaluate_graph_predictions(
        graph_predictions=graph_predictions,
        graph_ids=["bidmc01_w0000", "bidmc02_w0000"],
        threshold=0.5,
        nms_dist=0,
        top_n=0,
        fs=125,
        tolerances={"600ms": 75, "300ms": 38},
    )

    expected_graph_macro_f1 = (1.0 + (2.0 / 3.0)) / 2.0

    assert metrics["n_graphs"] == 2
    assert metrics["n_patients"] == 2
    assert metrics["boundary_f1_600ms"] == pytest.approx(expected_graph_macro_f1)
    assert metrics["boundary_f1_300ms"] == pytest.approx(expected_graph_macro_f1)
    assert metrics["patient_macro_f1_600ms"] == pytest.approx(expected_graph_macro_f1)
    assert metrics["event_precision_600ms"] == pytest.approx(1.0)
    assert metrics["event_recall_600ms"] == pytest.approx(2.0 / 3.0)
    assert metrics["event_f1_600ms"] == pytest.approx(0.8)
    assert metrics["boundary_mae_ms_600ms"] == pytest.approx(0.0)
