#!/usr/bin/env python3
"""P99: Multi-Scale Temporal Fusion DegreeGraphs experiment.

Tests whether building DG graphs at multiple LP cutoffs (0.5, 1.0, 4.0 Hz)
in addition to the primary 2.0 Hz graph improves raw model generalization.

Configs:
  baseline     — standard 6 features (control)
  ms_3scales   — 3 auxiliary scales: [0.5, 1.0, 4.0] Hz → 18 features
  ms_2scales   — 2 auxiliary scales: [0.5, 1.0] Hz → 14 features (coarse only)
  ms_fine      — 2 auxiliary scales: [1.0, 4.0] Hz → 14 features (fine focus)

Each config × 3 seeds = 12 total runs.
"""

import json
import os
import sys
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from graph_features import cache_split_graphs, compute_feature_dims, DEFAULT_FEATURE_CONFIG
from train import train, load_cached_graphs, evaluate, optimize_threshold, build_model
from torch_geometric.loader import DataLoader as PyGLoader

SEEDS = [42, 123, 456]
ROBUST_PP = {"threshold": 0.15, "nms_dist": 250}

CONFIGS = {
    "baseline": {
        **DEFAULT_FEATURE_CONFIG,
        "multi_scale_cutoffs": None,
    },
    "ms_3scales": {
        **DEFAULT_FEATURE_CONFIG,
        "multi_scale_cutoffs": [0.5, 1.0, 4.0],
    },
    "ms_2scales_coarse": {
        **DEFAULT_FEATURE_CONFIG,
        "multi_scale_cutoffs": [0.5, 1.0],
    },
    "ms_fine": {
        **DEFAULT_FEATURE_CONFIG,
        "multi_scale_cutoffs": [1.0, 4.0],
    },
}


def load_patients():
    """Load patient data from ground truth and splits."""
    with open("results/ground_truth.json") as f:
        gt = json.load(f)
    with open("results/splits.json") as f:
        splits = json.load(f)
    with open("results/patient_profiles.json") as f:
        profiles = json.load(f)

    import wfdb

    patients = {}
    all_ids = splits["train"] + splits["val"] + splits["test"]
    for pid in all_ids:
        rec = wfdb.rdrecord(pid, pn_dir="bidmc")
        resp = rec.p_signal[:, 0]
        troughs = [b["sample"] for b in gt[pid]]
        patients[pid] = {
            "signal": resp,
            "troughs": troughs,
            "profile": profiles.get(pid, {}),
        }

    return patients, splits


def run_trial(name, feat_config, seed, patients, splits):
    """Run a single training trial."""
    print(f"\n{'=' * 60}")
    print(f"TRIAL: {name}_s{seed}")
    print(f"{'=' * 60}")

    in_dim, edge_dim = compute_feature_dims(feat_config)
    print(f"Features: in_dim={in_dim}, edge_dim={edge_dim}")
    print(f"Multi-scale cutoffs: {feat_config.get('multi_scale_cutoffs')}")

    # Cache graphs per split into separate directories
    base_dir = f"data/graphs/p99_{name}"
    train_dir = os.path.join(base_dir, "train")
    val_dir = os.path.join(base_dir, "val")
    test_dir = os.path.join(base_dir, "test")

    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)

    cache_split_graphs(patients, splits["train"], train_dir, config=feat_config)
    cache_split_graphs(patients, splits["val"], val_dir, config=feat_config)
    cache_split_graphs(patients, splits["test"], test_dir, config=feat_config)

    # Train config
    train_config = {
        **feat_config,
        "arch": "transformer",
        "in_dim": in_dim,
        "edge_dim": edge_dim,
        "hidden_dim": 128,
        "n_heads": 4,
        "n_layers": 5,
        "dropout": 0.1,
        "residual": True,
        "batch_norm": True,
        "batch_size": 32,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 100,
        "patience": 15,
        "pos_weight": 7.0,
        "seed": seed,
        "loss_fn": "bce",
        "add_self_loops": False,
    }

    result = train(
        config=train_config,
        train_dir=train_dir,
        val_dir=val_dir,
        device="cuda",
    )

    # --- Test evaluation ---
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_model.pt")
    model_config = {**train_config, "in_dim": in_dim, "edge_dim": edge_dim}
    model = build_model(model_config)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.eval()
    device = "cpu"

    val_data = load_cached_graphs(val_dir)
    test_data = load_cached_graphs(test_dir)
    val_loader = PyGLoader(val_data, batch_size=32)
    test_loader = PyGLoader(test_data, batch_size=32)

    # Val-optimized PP
    pp = optimize_threshold(model, val_loader, device, tol_samples=75)
    val_opt_thresh, val_opt_nms = pp[0], pp[1]

    val_r = evaluate(model, val_loader, device, tol_samples=75,
                     threshold=val_opt_thresh, nms_dist=val_opt_nms)
    test_r = evaluate(model, test_loader, device, tol_samples=75,
                      threshold=val_opt_thresh, nms_dist=val_opt_nms)
    test_robust = evaluate(model, test_loader, device, tol_samples=75,
                           threshold=ROBUST_PP["threshold"],
                           nms_dist=ROBUST_PP["nms_dist"])
    test_raw = evaluate(model, test_loader, device, tol_samples=75,
                        threshold=0.5, nms_dist=0)

    val_pp = val_r.get("boundary_f1_600ms", 0)
    test_valpp = test_r.get("boundary_f1_600ms", 0)
    test_robust_f1 = test_robust.get("boundary_f1_600ms", 0)
    test_raw_f1 = test_raw.get("boundary_f1_600ms", 0)

    print(f"  RESULT: val_pp={val_pp:.4f} test_valpp={test_valpp:.4f} "
          f"test_robust={test_robust_f1:.4f} test_raw={test_raw_f1:.4f}")

    return {
        "name": f"{name}_s{seed}",
        "config": name,
        "seed": seed,
        "in_dim": in_dim,
        "edge_dim": edge_dim,
        "val_pp": val_pp,
        "test_valpp": test_valpp,
        "test_robust": test_robust_f1,
        "test_raw": test_raw_f1,
    }


def main():
    print("P99: Multi-Scale Temporal Fusion DegreeGraphs")
    print("Loading patients...")
    patients, splits = load_patients()
    print(f"Loaded {len(patients)} patients")

    results = []
    for config_name, feat_config in CONFIGS.items():
        for seed in SEEDS:
            trial_name = f"{config_name}_s{seed}"
            try:
                r = run_trial(config_name, feat_config, seed, patients, splits)
                results.append(r)
            except Exception as e:
                print(f"ERROR in {trial_name}: {e}")
                traceback.print_exc()
                results.append({"name": trial_name, "error": str(e)})

    # Summary
    print("\n" + "=" * 80)
    print("P99 SUMMARY")
    print("=" * 80)
    for config_name in CONFIGS:
        config_results = [r for r in results if r.get("config") == config_name and "error" not in r]
        if config_results:
            test_robust = [r["test_robust"] for r in config_results]
            test_raw = [r["test_raw"] for r in config_results]
            print(f"  {config_name:20s}: test_robust={np.mean(test_robust):.4f}±{np.std(test_robust):.4f} "
                  f"test_raw={np.mean(test_raw):.4f}±{np.std(test_raw):.4f}")
        else:
            print(f"  {config_name:20s}: ALL FAILED")

    # Save results
    os.makedirs("results/aorus", exist_ok=True)
    with open("results/aorus/p99_multiscale.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/aorus/p99_multiscale.json")
    print("DONE")


if __name__ == "__main__":
    main()
